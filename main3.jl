using JLD2, CSV, DataFrames, Dates, Plots, Peaks
using DataFrames, SavitzkyGolay, DSP,LinearAlgebra

# Load the data from the JLD2 file
df = load("./data_out/covid_data_interpolated.jld2")["df"]

println("Data loaded successfully from JLD2 file.")
println("DataFrame size: ", size(df))
println("First 5 rows of the DataFrame:")
first(df, 5) |> println
# Optionally, you can also check the column names and types
println("Column names: ", names(df))
println("Column types: ", eltype.(eachcol(df)))

# ============================================================
# ИДЕНТИФИКАЦИЯ ВОЛН ЗАБОЛЕВАЕМОСТИ
# ============================================================

# ─────────────────────────────────────────────────────────────
# Общий хелпер: нормализованная свёртка с частичным окном на краях
# Работает для любого симметричного ядра
# ─────────────────────────────────────────────────────────────
function _convolve_with_edges(y::Vector{Float64}, kernel::Vector{Float64})
    n      = length(y)
    half   = length(kernel) ÷ 2
    result = similar(y)

    for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)

        # Вырезаем соответствующую часть ядра (края — неполное окно)
        k_lo = half - (i - lo) + 1
        k_hi = half + (hi - i) + 1
        w = kernel[k_lo:k_hi]

        result[i] = dot(w, y[lo:hi]) / sum(w)   # нормировка по реальным весам
    end

    return result
end

# ─────────────────────────────────────────────────────────────
# 1. Savitzky-Golay
#    window  — ширина окна в днях (нечётное число)
#    polyorder — степень полинома (обычно 2–4)
#    out_col — имя новой колонки (по умолчанию: <col>_sg)
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# SG: коэффициенты через least-squares полином в окне
# ─────────────────────────────────────────────────────────────
function _sg_kernel(window::Int, polyorder::Int)
    half = window ÷ 2
    xs   = -half:half                           # [-h, ..., 0, ..., h]
    # Матрица Вандермонда: каждая строка — [1, x, x², ..., x^p]
    V    = [x^p for x in xs, p in 0:polyorder]
    # Коэффициенты фильтра = строка 0 из (VᵀV)⁻¹Vᵀ, т.е. центральная точка
    return (V' * V \ V')[1, :]                  # вектор длины window
end

function _convolve_with_edges(y::Vector{Float64}, kernel::Vector{Float64})
    n    = length(y)
    half = length(kernel) ÷ 2
    map(1:n) do i
        lo   = max(1, i - half)
        hi   = min(n, i + half)
        k_lo = half - (i - lo) + 1
        k_hi = half + (hi - i) + 1
        w    = kernel[k_lo:k_hi]
        dot(w, y[lo:hi]) / sum(w)
    end
end

# ─────────────────────────────────────────────────────────────
# 1. Savitzky-Golay
# ─────────────────────────────────────────────────────────────
function smooth_sg!(df::DataFrame, col::Symbol;
                    window::Int     = 15,
                    polyorder::Int  = 3,
                    out_col::Symbol = Symbol(col, :_sg))

    isodd(window)       || throw(ArgumentError("window должен быть нечётным"))
    window > polyorder  || throw(ArgumentError("window > polyorder"))

    kernel         = _sg_kernel(window, polyorder)
    df[!, out_col] = _convolve_with_edges(Float64.(df[!, col]), kernel)
    return df
end

# ─────────────────────────────────────────────────────────────
# Сглаженная первая производная
#   out_col  — имя колонки для результата
#   sigma    — ширина гауссиана для сглаживания
#   trunc    — количество σ для обрезки ядра
# ─────────────────────────────────────────────────────────────
function add_smoothed_derivative!(df::DataFrame, col::Symbol;
                                  out_col::Symbol = Symbol(col, :deriv_smooth),
                                  sign_col::Symbol = Symbol(out_col, :sign),
                                  sigma::Float64   = 5.0,
                                  trunc::Float64   = 3.0)

    y = Float64.(df[!, col])
    n = length(y)
    n >= 2 || throw(ArgumentError("Нужно как минимум два значения для вычисления производной"))

    dy = similar(y)
    dy[1] = y[2] - y[1]
    for i in 2:n-1
        dy[i] = (y[i + 1] - y[i - 1]) / 2
    end
    dy[n] = y[n] - y[n - 1]

    half = round(Int, trunc * sigma)
    kernel = exp.(-0.5 .* ((-half:half) ./ sigma).^2)

    df[!, out_col] = _convolve_with_edges(dy, kernel)
    df[!, sign_col] = sign.(df[!, out_col])
    return df
end

function fill_derivative_signs(signs::AbstractVector{<:Integer})
    s = [v > 0 ? 1 : v < 0 ? -1 : 0 for v in signs]
    last = 0
    for i in 1:length(s)
        if s[i] == 0
            s[i] = last
        else
            last = s[i]
        end
    end
    first_nonzero = findfirst(!=(0), s)
    if first_nonzero !== nothing
        for i in 1:first_nonzero-1
            s[i] = s[first_nonzero]
        end
    end
    return s
end

function detect_waves_by_derivative_sign!(df::DataFrame, sign_col::Symbol;
                                          out_col::Symbol = :wave)
    n = nrow(df)
    if n == 0
        df[!, out_col] = Int[]
        return df
    end

    signs = fill_derivative_signs([ismissing(v) ? 0 : Int(v) for v in df[!, sign_col]])
    wave = zeros(Int, n)
    current_wave = 0
    current_state = :idle

    for i in 1:n
        s = signs[i]
        if s == 1
            if current_state != :up
                current_wave += 1
                current_state = :up
            end
            wave[i] = current_wave
        elseif s == -1
            if current_state == :up
                current_state = :down
            end
            if current_wave > 0
                wave[i] = current_wave
            end
        else
            wave[i] = current_wave
        end
    end

    df[!, out_col] = wave
    return df
end

# ─────────────────────────────────────────────────────────────
# 2. Gaussian convolution
# ─────────────────────────────────────────────────────────────
function smooth_gauss!(df::DataFrame, col::Symbol;
                        sigma::Float64   = 5.0,
                        trunc::Float64   = 3.0,
                        out_col::Symbol  = Symbol(col, :_gauss))

    half           = round(Int, trunc * sigma)
    kernel         = exp.(-0.5 .* ((-half:half) ./ sigma).^2)
    df[!, out_col] = _convolve_with_edges(Float64.(df[!, col]), kernel)
    return df
end
# ─────────────────────────────────────────────────────────────
# Использование
# ─────────────────────────────────────────────────────────────
smooth_sg!(df, :daily_interp; window = 125, polyorder = 3)        # → df.daily_interp_sg
smooth_gauss!(df, :daily_interp; sigma = 15.0)                    # → df.daily_interp_gauss
add_smoothed_derivative!(df, :daily_interp;
    out_col = :daily_interp_deriv_smooth,
    sign_col = :daily_interp_deriv_smooth_sign,
    sigma = 15.0,
    trunc = 3.0)
detect_waves_by_derivative_sign!(df, :daily_interp_deriv_smooth_sign; out_col = :wave)

plot(df.date, [df.daily_interp_sg, df.daily_interp_gauss],
     labels = ["Savitzky-Golay" "Gaussian"],
     title = "Сравнение методов сглаживания",
     xlabel = "Дата", ylabel = "Значение",
     legend = :topright)


# ─────────────────────────────────────────────────────────────
# Детектирование пиков в колонке DataFrame
# Возвращает DataFrame с найденными пиками и их характеристиками
# ─────────────────────────────────────────────────────────────
function detect_peaks(df::DataFrame, col::Symbol;
                       time_col::Symbol             = :date,
                       min_prominence_frac::Float64 = 0.15,
                       min_distance::Int            = 14)

    y    = Float64.(df[!, col])
    span = maximum(y) - minimum(y)

    # Шаг 1: все локальные максимумы с минимальным расстоянием между ними
    idxs, vals = findmaxima(y, min_distance)

    # Шаг 2: prominence для каждого пика
    proms = peakproms(idxs, y)[2]

    # Шаг 3: фильтр по prominence
    mask  = proms .>= min_prominence_frac * span
    idxs, vals, proms = idxs[mask], vals[mask], proms[mask]

    return DataFrame(
        index      = idxs,
        time       = df[idxs, time_col],
        value      = vals,
        prominence = proms,
    )
end

peaks_df = detect_peaks(df, :daily_interp_gauss;
                         time_col            = :date,
                         min_prominence_frac = 0.003,
                         min_distance        = 8)


# ─────────────────────────────────────────────────────────────
# Подробная информация по пикам:
# старт/конец волны через watershed, AUC, ширина на полувысоте (FWHM)
# ─────────────────────────────────────────────────────────────
function peak_info(df::DataFrame, col::Symbol;
                    time_col::Symbol             = :date,
                    min_prominence_frac::Float64 = 0.15,
                    min_distance::Int            = 21,
                    baseline_frac::Float64       = 0.10)

    y    = Float64.(df[!, col])
    n    = length(y)
    span = maximum(y) - minimum(y)
    base = minimum(y) + baseline_frac * span

    result = findmaxima(y, min_distance)
    idxs = result.indices
    proms = peakproms(idxs, y)[2]

    # Filter by prominence
    mask = proms .>= min_prominence_frac * span
    idxs = idxs[mask]
    proms = proms[mask]

    isempty(idxs) && return DataFrame()

    # Watershed: каждая точка → ближайший пик (расстояние / высота пика)
    assignments = map(1:n) do i
        idxs[argmin([abs(i - p) / y[p] for p in idxs])]
    end

    rows = map(idxs) do p
        owned = filter(i -> assignments[i] == p && y[i] >= base, 1:n)
        isempty(owned) && return nothing

        i_start = minimum(owned)
        i_stop  = maximum(owned)
        half    = y[p] / 2.0

        # FWHM: ищем пересечения с половиной высоты пика слева и справа
        fwhm_l = findlast(i -> y[i] <= half, i_start:p)
        fwhm_r = findfirst(i -> y[i] <= half, p:i_stop)
        fwhm   = (isnothing(fwhm_l) || isnothing(fwhm_r)) ? missing :
                  (p + fwhm_r - 1) - (i_start + fwhm_l - 1)

        (
            peak_idx        = p,
            peak_time       = df[p, time_col],
            peak_value      = y[p],
            prominence      = proms[findfirst(==(p), idxs)],
            start_idx       = i_start,
            start_time      = df[i_start, time_col],
            stop_idx        = i_stop,
            stop_time       = df[i_stop, time_col],
            duration_days   = i_stop - i_start + 1,
            auc             = sum(y[i_start:i_stop]),
            auc_above_base  = sum(y[i_start:i_stop] .- base),
            fwhm_days       = fwhm,
            peak_to_start   = p - i_start,       # дней от старта до пика
            peak_to_end     = i_stop - p,         # дней от пика до конца
            asymmetry       = (p - i_start) / max(i_stop - p, 1),  # >1 → правый хвост длиннее
        )
    end |> x -> filter(!isnothing, x)

    return DataFrame(rows)
end

# ─────────────────────────────────────────────────────────────
# График: данные + маркеры пиков
# ─────────────────────────────────────────────────────────────
function plot_peaks(df::DataFrame, col::Symbol, info::DataFrame;
                     time_col::Symbol = :date,
                     title::String    = "Peak Detection")

    t = df[!, time_col]
    y = Float64.(df[!, col])

    fig = plot(t, y;
               label     = string(col),
               color     = :steelblue,
               linewidth = 2,
               title,
               xlabel    = string(time_col),
               ylabel    = string(col),
               legend    = :topright,
               size      = (1200, 500),
               grid      = true,
               gridalpha = 0.3)

    # Старт волны — зелёные треугольники вверх
    scatter!(fig, t[info.start_idx], y[info.start_idx];
             label      = "start",
             color      = :green,
             markersize = 7,
             markershape = :utriangle)

    # Пик — красные звёзды
    scatter!(fig, t[info.peak_idx], y[info.peak_idx];
             label       = "peak",
             color       = :red,
             markersize  = 9,
             markershape = :star5)

    # Конец волны — синие треугольники вниз
    scatter!(fig, t[info.stop_idx], y[info.stop_idx];
             label       = "end",
             color       = :royalblue,
             markersize  = 7,
             markershape = :dtriangle)

    # Аннотации над пиками: номер волны + значение
    for (i, row) in enumerate(eachrow(info))
        annotate!(fig, t[row.peak_idx], row.peak_value * 1.04,
                  text("W$i\n$(round(Int, row.peak_value))", 8, :red, :center))
    end

    return fig
end

function contiguous_segments(mask::AbstractVector{Bool})
    segments = Vector{Tuple{Int,Int}}()
    i = 1
    n = length(mask)
    while i <= n
        if mask[i]
            j = i
            while j < n && mask[j + 1]
                j += 1
            end
            push!(segments, (i, j))
            i = j + 1
        else
            i += 1
        end
    end
    return segments
end

function plot_cases_with_waves(df::DataFrame, cases_col::Symbol;
                                date_col::Symbol = :date,
                                wave_col::Symbol = :wave,
                                title::String    = "Заболеваемость и волны")
    t = df[!, date_col]
    y = Float64.(df[!, cases_col])

    fig = plot(t, y;
               label     = string(cases_col),
               color     = :black,
               linewidth = 2,
               title     = title,
               xlabel    = string(date_col),
               ylabel    = string(cases_col),
               legend    = :topright,
               size      = (1200, 500),
               grid      = true,
               gridalpha = 0.25)

    ymin = minimum(y)
    ymax = maximum(y)
    wave_colors = [RGBA(0.2, 0.4, 0.8, 0.16),
                   RGBA(0.2, 0.7, 0.2, 0.16),
                   RGBA(0.9, 0.4, 0.1, 0.16),
                   RGBA(0.7, 0.2, 0.8, 0.16),
                   RGBA(0.1, 0.6, 0.6, 0.16),
                   RGBA(0.8, 0.2, 0.3, 0.16),
                   RGBA(0.4, 0.5, 0.1, 0.16)]

    for wave in sort(unique(skipmissing(df[!, wave_col])))
        color = wave_colors[mod1(wave, length(wave_colors))]
        mask = df[!, wave_col] .== wave
        for (lo, hi) in contiguous_segments(mask)
            x_rect = [t[lo], t[lo], t[hi], t[hi]]
            y_rect = [ymin, ymax, ymax, ymin]
            plot!(fig, x_rect, y_rect;
                  seriestype  = :shape,
                  fillcolor   = color,
                  linecolor   = :transparent,
                  label       = false)
            annotate!(fig, t[clamp(div(lo + hi, 2), 1, length(t))],
                      ymax * 0.92,
                      text("W$(wave)", 9, :blue, :center))
        end
    end

    return fig
end

function plot_cases_with_sign(df::DataFrame, cases_col::Symbol, sign_col::Symbol;
                               date_col::Symbol = :date,
                               title::String    = "Заболеваемость и знак производной")
    t = df[!, date_col]
    y = Float64.(df[!, cases_col])
    fig = plot(t, y;
               label     = string(cases_col),
               color     = :black,
               linewidth = 2,
               title     = title,
               xlabel    = string(date_col),
               ylabel    = string(cases_col),
               legend    = :topright,
               size      = (1200, 500),
               grid      = true,
               gridalpha = 0.25)

    colors = Dict(-1 => RGBA(1, 0.2, 0.2, 0.12),
                   0  => RGBA(0.6, 0.6, 0.6, 0.12),
                   1  => RGBA(0.2, 0.7, 0.2, 0.12))

    ymin = minimum(y)
    ymax = maximum(y)
    for sign in sort(unique(skipmissing(df[!, sign_col])))
        mask = df[!, sign_col] .== sign
        for (lo, hi) in contiguous_segments(mask)
            x_rect = [t[lo], t[lo], t[hi], t[hi]]
            y_rect = [ymin, ymax, ymax, ymin]
            plot!(fig, x_rect, y_rect;
                  seriestype  = :shape,
                  fillcolor   = get(colors, sign, RGBA(0,0,0,0.1)),
                  linecolor   = :transparent,
                  label       = false)
        end
    end

    hline!(fig, [0]; color = :gray, linestyle = :dash, linewidth = 1, label = "zero")
    return fig
end

info = peak_info(df, :daily_interp_gauss,
                  time_col            = :date,
                  min_prominence_frac = 0.005,
                  min_distance        = 11,
                  baseline_frac       = 0.002)

# Красивый вывод таблицы
select(info, :peak_time, :peak_value, :prominence,
             :start_time, :stop_time,
             :duration_days, :fwhm_days,
             :auc_above_base, :asymmetry)

fig = plot_peaks(df, :daily_interp_gauss, info; time_col = :date,
                  title = "ВП — пики заболеваемости")
savefig(fig, "figures/covid/peaks.png")
savefig(fig, "figures/covid/peaks.svg")
println("График с пиками сохранён: figures/covid/peaks.png/svg")

sign_col = :daily_interp_deriv_smooth_sign 
if sign_col !== nothing
    fig_sign = plot_cases_with_sign(df, :daily_interp_gauss, sign_col;
                                    date_col = :date,
                                    title = "Заболеваемость и знак производной\n
зеленый - рост, красный - спад")
    display(fig_sign)
    savefig(fig_sign, "figures/covid/cases_derivative_sign.png")
    savefig(fig_sign, "figures/covid/cases_derivative_sign.svg")
    println("График со знаком производной сохранён: figures/covid/cases_derivative_sign.png/svg")
else
    println("Колонка знака производной не найдена: ни :daily_interp_deriv_smooth_sign, ни :deriv_sign")
end

wave_col = :wave
if wave_col !== nothing
    fig_waves = plot_cases_with_waves(df, :daily_interp_gauss; wave_col = wave_col,
                                      date_col = :date,
                                      title = "Заболеваемость и волны")
    display(fig_waves)
    savefig(fig_waves, "figures/covid/cases_waves.png")
    savefig(fig_waves, "figures/covid/cases_waves.svg")
    println("График с волнами сохранён: figures/covid/cases_waves.png/svg")
else
    println("Колонка волны не найдена: ни :wave, ни :wave_number")
end

# ============================================================
# Сохраняем датафрейм с новой информацией о волнах и производных, включая csv

function save_data(df::DataFrame, filename::String, out_dir::String = "data_out")
    jld2file = joinpath(out_dir, filename * ".jld2")
    JLD2.save(jld2file, "df", df)

    parquetfile = joinpath(out_dir, filename * ".parquet")
    df_parquet = copy(df)
    for name in names(df_parquet)
        col = df_parquet[!, name]
        if nonmissingtype(eltype(col)) <: Date
            df_parquet[!, name] = string.(col)
        end
    end
    Parquet.write_parquet(parquetfile, df_parquet)

    csvfile = joinpath(out_dir, filename * ".csv")
    CSV.write(csvfile, df)

    @info "Data saved to: $jld2file, $parquetfile and $csvfile"
end

save_data(df, "covid_data_with_waves_and_derivatives")