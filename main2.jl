using JLD2, DataFrames

# Load the data from the JLD2 file
df = load("./data_out/covid_data_translated.jld2")["df"]

println("Data loaded successfully from JLD2 file.")
println("DataFrame size: ", size(df))
println("First 5 rows of the DataFrame:")
first(df, 5) |> println

using Interpolations
using Interpolations: linear_interpolation, Line

function natural_cubic_spline(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    if n == 2
        return t -> y[1] + (y[2] - y[1]) * (t - x[1]) / (x[2] - x[1])
    end

    h = [x[i+1] - x[i] for i in 1:n-1]
    alpha = vcat([0.0], [3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]) for i in 2:n-1], [0.0])

    l = ones(Float64, n)
    mu = zeros(Float64, n)
    z = zeros(Float64, n)
    c = zeros(Float64, n)
    b = zeros(Float64, n-1)
    d = zeros(Float64, n-1)

    for i in 2:n-1
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    end

    for j in n-1:-1:1
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2*c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    end

    return function(t)
        if t <= x[1]
            return y[1]
        elseif t >= x[end]
            return y[end]
        end
        j = clamp(searchsortedlast(x, t), 1, n-1)
        dx = t - x[j]
        return y[j] + b[j] * dx + c[j] * dx^2 + d[j] * dx^3
    end
end

function interpolate_missing(col::Vector, method::String="linear")
    valid_idx = findall(!ismissing, col)
    
    if length(valid_idx) < 2
        return col
    end
    
    result = Vector{Union{Float64, Missing}}(col)
    valid_vals = Float64.(col[valid_idx])
    
    if method == "linear"
        itp = linear_interpolation(valid_idx, valid_vals, extrapolation_bc=Line())
        for i in eachindex(col)
            if ismissing(col[i])
                result[i] = itp(i)
            end
        end
        
    elseif method == "forward_fill"
        for i in eachindex(col)
            if ismissing(col[i]) && i > 1 && !ismissing(result[i-1])
                result[i] = result[i-1]
            end
        end
        
    elseif method == "backward_fill"
        for i in reverse(eachindex(col))
            if ismissing(col[i]) && i < length(col) && !ismissing(result[i+1])
                result[i] = result[i+1]
            end
        end
        
    elseif method == "cubic"
        spline = natural_cubic_spline(valid_idx, valid_vals)
        for i in eachindex(col)
            if ismissing(col[i])
                if i < valid_idx[1]
                    result[i] = 0.0
                elseif i > valid_idx[end]
                    result[i] = valid_vals[end]
                else
                    result[i] = spline(i)
                end
            end
        end
    else
        error("Неизвестный метод: $method. Используйте: linear, forward_fill, backward_fill, cubic")
    end

    # Сделать результат ненубывающим, чтобы total не уменьшался
    last_val = 0.0
    for i in eachindex(result)
        if ismissing(result[i])
            continue
        end
        if result[i] < last_val
            result[i] = last_val
        else
            last_val = result[i]
        end
    end
    
    return result
end

function first_difference(col::Vector)
    n = length(col)
    result = Vector{Union{Missing, Float64}}(undef, n)
    result[1] = missing
    for i in 2:n
        if ismissing(col[i]) || ismissing(col[i-1])
            result[i] = missing
        else
            result[i] = col[i] - col[i-1]
        end
    end
    return result
end

# Примеры использования:
#= 
df.total_interp_linear = interpolate_missing(df.total, "linear")
df.total_interp_cubic = interpolate_missing(df.total, "cubic")
df.total_interp_ff = interpolate_missing(df.total, "forward_fill")
df.total_interp_bf = interpolate_missing(df.total, "backward_fill") 
=#

df.total_interp = interpolate_missing(df.total, "cubic")
df.total_interp_diff = first_difference(df.total_interp)
df.total_interp_diff[1] = 0.0

println("Интерполяция завершена.")
println("Первые 5 строк с интерполяцией и первой разностью:")
first(df[!, [:total, :total_interp, :total_interp_diff]], 5) |> println

# === Графики ===
using Plots

# 1. Совмещённый график: date vs total и total_interp
p_total = plot(df.date, df.total, label="total", legend=:topleft,
               xlabel="Date", ylabel="Total", title="Корличество заболевших (наблюдаемое vs интерполированное)",
               # уменьшить размер шрифта для осей и заголовка
               titlefontsize=10
               )
plot!(p_total, df.date, df.total_interp, label="total_interp", linestyle=:dash)
savefig(p_total, "./figures/covid/total_vs_interp.png")
savefig(p_total, "./figures/covid/total_vs_interp.svg")
println("График сохранён: ./figures/covid/total_vs_interp.png")

# 2. Совмещённый график: date vs daily и total_interp_diff
p_daily = plot(df.date, df.daily, label="daily", legend=:topleft,
               xlabel="Date", ylabel="Count", title="Daily: observed vs interpolated diff",
               titlefontsize=10
               )
plot!(p_daily, df.date, df.total_interp_diff, label="total_interp_diff", linestyle=:dash)
savefig(p_daily, "./figures/covid/daily_vs_interp_diff.png")
savefig(p_daily, "./figures/covid/daily_vs_interp_diff.svg")
println("График сохранён: ./figures/covid/daily_vs_interp_diff.png")

rename!(df, :total_interp_diff => :daily_interp)

# Окно сглаживания для daily_interp
using Statistics, StatsBase
window_size = 14
df.daily_interp_smooth = [mean(skipmissing(df.daily_interp[max(1, i-window_size+1):i])) for i in 1:nrow(df)]    

# 2. Совмещённый график: date vs daily и total_interp_diff
p_daily = plot(df.date, df.daily, label="daily", legend=:topleft,
               xlabel="Date", ylabel="Count", title="Заболевшие в день: наблюдаемое vs интерполированное (дифференцированное)",
               titlefontsize=10
               )
# 2. Совмещённый график: date vs daily и daily_interp и daily_interp_smooth
p_daily1 = plot(df.date, df.daily, label="daily", legend=:topleft,
               xlabel="Date", ylabel="Count", title="Заболевшие в день:\n наблюдаемое vs интерполированное и сглаженное",
               titlefontsize=10
               )
plot!(p_daily1, df.date, df.daily_interp, label="daily_interp", linestyle=:dash)
plot!(p_daily1, df.date, df.daily_interp_smooth, label="daily_interp_smooth", 
    linestyle=:solid, linewidth=2, color=:red)
savefig(p_daily1, "./figures/covid/daily_vs_interp_diff.svg")
savefig(p_daily1, "./figures/covid/daily_vs_interp_diff.png")
println("График сохранён: ./figures/covid/daily_vs_interp_diff.svg")

# Сохранение DataFrame в JLD2 файл
@save "./data_out/covid_data_interpolated.jld2" df
@save "./covid_data_interpolated.jld2" df