# ============================================================
# config/utils.jl — Конфигурация и утилиты графиков
# ============================================================

using Dates
using Plots
using CSV
using DataFrames

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
const PROJECT_ROOT = dirname(@__DIR__)

const PATHS = (
    root      = PROJECT_ROOT,
    csv_covid = joinpath(PROJECT_ROOT, "csv", "covid19_daily.csv"),
    csv_vp    = joinpath(PROJECT_ROOT, "csv", "vp_daily.csv"),
    figures   = joinpath(PROJECT_ROOT, "figures"),
    figures_covid = joinpath(PROJECT_ROOT, "figures", "covid"),
    figures_vp    = joinpath(PROJECT_ROOT, "figures", "vp"),
    data_out  = joinpath(PROJECT_ROOT, "data_out"),
)

const PLOT_CFG = (
    dpi      = 350,
    size     = (1920, 1080),  # 16:9
    font     = "Calibri",
    font_alt = "Aptos",
    fontsize = 14,
    linewidth = 2,
    marker_size = 3,
    fmts     = [:png, :svg],
)

# ============================================================
# ИНИЦИАЛИЗАЦИЯ ПРОЕКТА
# ============================================================
"""
    setup_project()

Создаёт необходимые папки, проверяет наличие CSV-файлов.
"""
function setup_project()
    for dir_path in (PATHS.figures, PATHS.figures_covid, PATHS.figures_vp, PATHS.data_out)
        mkpath(dir_path)
        @info "Папка создана: $dir_path"
    end

    for (name, path) in (; :covid => PATHS.csv_covid, :vp => PATHS.csv_vp)
        if isfile(path)
            @info "Найден файл данных: $name → $path"
        else
            @warn "Файл данных НЕ найден: $name → $path"
        end
    end

    return PATHS
end

# ============================================================
# УТИЛИТЫ ГРАФИКОВ
# ============================================================
"""
    apply_theme!(p::Plots.Plot)

Применяет тему оформления к графику: шрифты, размеры, DPI.
"""
function apply_theme!(p::Plots.Plot)
    default(
        fontfamily = PLOT_CFG.font,
        fontsize   = PLOT_CFG.fontsize,
        linewidth  = PLOT_CFG.linewidth,
        markersize = PLOT_CFG.marker_size,
        dpi        = PLOT_CFG.dpi,
    )
    return p
end

"""
    save_figure(p::Plots.Plot, basename::String; dir = PATHS.figures_covid)

Сохраняет график в PNG и SVG с высоким разрешением.
"""
function save_figure(p::Plots.Plot, basename::String; dir = PATHS.figures_covid)
    mkpath(dir)
    for fmt in PLOT_CFG.formats
        out_path = joinpath(dir, "$basename.$fmt")
        savefig(p, out_path)
        @info "Сохранён график: $out_path"
    end
    return nothing
end

"""
    format_axis_labels(labels::Vector{String}; max_labels::Int = 8)

Прореживает подписи оси X для читаемости.
"""
function format_axis_labels(labels::Vector{String}; max_labels::Int = 8)
    n = length(labels)
    if n <= max_labels
        return labels
    end
    step = cld(n, max_labels)
    result = String[i % step == 0 || i == 1 ? labels[i] : "" for i in 1:n]
    return result
end

"""
    nice_date_range(dates::Vector{Date})

Возвращает красиво отформатированную строку диапазона дат.
"""
function nice_date_range(dates::Vector{Date})::String
    d1 = first(dates)
    d2 = last(dates)
    return "$(dayname(d1)), $(Dates.value(d1)) — $(dayname(d2)), $(Dates.value(d2))"
end

"""
    set_gr_backend!()

Устанавливает GR бэкенд с настройками по умолчанию.
"""
function set_gr_backend!()
    gr()
    default(
        fontfamily = PLOT_CFG.font,
        fontsize   = PLOT_CFG.fontsize,
        linewidth  = PLOT_CFG.linewidth,
        markersize = PLOT_CFG.marker_size,
        dpi        = PLOT_CFG.dpi,
        size       = PLOT_CFG.size,
        legend     = :topleft,
        grid       = true,
        gridalpha  = 0.3,
        framestyle = :semi,
    )
    @info "GR бэкенд настроен: dpi=$(PLOT_CFG.dpi), size=$(PLOT_CFG.size), font=$(PLOT_CFG.font)"
    return nothing
end
