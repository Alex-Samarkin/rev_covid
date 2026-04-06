# ============================================================
# main.jl — Главный скрипт: загрузка, анализ, визуализация COVID-19
# ============================================================
#
# Зависимости (раскомментировать при первом запуске):
#   using Pkg
#   Pkg.add(["Plots", "CSV", "DataFrames", "GR"])
#
# Запуск:
#   julia main.jl
# ============================================================

using Pkg

# --- Установка зависимостей (первый запуск) -----------------
const REQUIRED_PACKAGES = ["Plots", "CSV", "DataFrames", "GR", "Dates", "Statistics"]
const MISSING_PKGS = filter(p -> !any(haskey.(Ref(Pkg.dependencies()), Base.UUID.(keys(Pkg.dependencies())))), REQUIRED_PACKAGES)

if !isempty(MISSING_PKGS)
    @info "Установка пакетов: $MISSING_PKGS"
    Pkg.add(MISSING_PKGS)
end

using Plots
using CSV
using DataFrames
using Dates
using Statistics

# --- Подключение модулей проекта ----------------------------
include(joinpath(@__DIR__, "config", "utils.jl"))
include(joinpath(@__DIR__, "analysis", "covid_analysis.jl"))

# ============================================================
# ГРАФИКИ
# ============================================================

"""
    plot_daily_cases(df::DataFrame)

График заболеваний за сутки + 7-дневное скользящее среднее.
"""
function plot_daily_cases(df::DataFrame)
    @info "Построение графика: ежедневные случаи"

    p = plot(
        df[!, :дата],
        df[!, :за_сутки],
        seriestype = :bar,
        label      = "За сутки",
        color      = :steelblue,
        alpha      = 0.7,
        ylabel     = "Число случаев",
        xlabel     = "Дата",
        title      = "Заболевания COVID-19 за сутки",
    )

    # Скользящее среднее
    if :за_сутки_ma7 in names(df)
        plot!(
            p,
            df[!, :дата],
            df[!, :за_сутки_ma7],
            seriestype  = :line,
            label       = "Скользящее среднее (7 дн.)",
            color       = :red,
            linewidth   = 3,
        )
    end

    # Прореживание подписей оси X
    n = nrow(df)
    step = cld(n, 12)
    xticks!(
        p,
        df[!, :дата][1:step:end],
        format_axis_labels(string.(df[!, :дата])[1:step:end]; max_labels = 12),
    )

    xrotation!(p, 45)
    apply_theme!(p)

    save_figure(p, "covid_daily_cases"; dir = PATHS.figures_covid)
    return p
end

"""
    plot_cumulative_cases(df::DataFrame)

График накопительных заболеваний.
"""
function plot_cumulative_cases(df::DataFrame)
    @info "Построение графика: накопительные случаи"

    df_clean = dropmissing(df, :всего_накоп_)

    p = plot(
        df_clean[!, :дата],
        df_clean[!, :всего_накоп_],
        seriestype = :line,
        label      = "Всего (накоп.)",
        color      = :navy,
        linewidth  = 3,
        fill       = (0, 0.15, :navy),
        ylabel     = "Число случаев",
        xlabel     = "Дата",
        title      = "Накопительная заболеваемость COVID-19",
    )

    # Прореживание подписей оси X
    n = nrow(df_clean)
    step = cld(n, 12)
    xticks!(
        p,
        df_clean[!, :дата][1:step:end],
        format_axis_labels(string.(df_clean[!, :дата])[1:step:end]; max_labels = 12),
    )

    xrotation!(p, 45)
    apply_theme!(p)

    save_figure(p, "covid_cumulative"; dir = PATHS.figures_covid)
    return p
end

"""
    plot_deaths_daily(df::DataFrame)

График летальных случаев за сутки.
"""
function plot_deaths_daily(df::DataFrame)
    @info "Построение графика: летальные случаи за сутки"

    p = plot(
        df[!, :дата],
        df[!, :летальных_за_сутки],
        seriestype = :bar,
        label      = "Летальных за сутки",
        color      = :crimson,
        alpha      = 0.7,
        ylabel     = "Число случаев",
        xlabel     = "Дата",
        title      = "Летальные случаи COVID-19 за сутки",
    )

    n = nrow(df)
    step = cld(n, 12)
    xticks!(
        p,
        df[!, :дата][1:step:end],
        format_axis_labels(string.(df[!, :дата])[1:step:end]; max_labels = 12),
    )

    xrotation!(p, 45)
    apply_theme!(p)

    save_figure(p, "covid_deaths_daily"; dir = PATHS.figures_covid)
    return p
end

"""
    plot_combined_panel(df::DataFrame)

Комбинированная панель: 2x2 графика.
"""
function plot_combined_panel(df::DataFrame)
    @info "Построение комбинированной панели 2×2"

    df_clean = dropmissing(df, :всего_накоп_)

    # --- 1. За сутки + скользящее среднее ---
    p1 = plot(
        df[!, :дата], df[!, :за_сутки],
        seriestype = :bar, label = "За сутки", color = :steelblue, alpha = 0.7,
        ylabel = "Случаев/день", title = "A) За сутки",
    )
    if :за_сутки_ma7 in names(df)
        plot!(p1, df[!, :дата], df[!, :за_сутки_ma7],
              seriestype = :line, label = "MA(7)", color = :red, linewidth = 2)
    end

    # --- 2. Накопительные ---
    p2 = plot(
        df_clean[!, :дата], df_clean[!, :всего_накоп_],
        seriestype = :line, label = "Всего", color = :navy, linewidth = 3,
        fill = (0, 0.15, :navy), ylabel = "Всего случаев", title = "B) Накопительно",
    )

    # --- 3. Летальные за сутки ---
    p3 = plot(
        df[!, :дата], df[!, :летальных_за_сутки],
        seriestype = :bar, label = "Летальных/день", color = :crimson, alpha = 0.7,
        ylabel = "Летальных/день", xlabel = "Дата", title = "C) Летальные за сутки",
    )

    # --- 4. Летальные накопительно ---
    p4 = plot(
        df_clean[!, :дата], df_clean[!, :летальных_всего],
        seriestype = :line, label = "Летальных всего", color = :darkred, linewidth = 3,
        fill = (0, 0.1, :darkred), ylabel = "Летальных всего", xlabel = "Дата",
        title = "D) Летальные накопительно",
    )

    # --- Прореживание X-подписей ---
    for p_obj in (p1, p2, p3, p4)
        n = length(p_obj[:xaxis][:ticks][1])
        if n > 10
            step = cld(n, 8)
            xticks!(p_obj, p_obj[:xaxis][:ticks][1][1:step:end],
                    p_obj[:xaxis][:ticks][2][1:step:end])
        end
        xrotation!(p_obj, 45)
    end

    apply_theme!(p1); apply_theme!(p2); apply_theme!(p3); apply_theme!(p4)

    # Компоновка 2×2
    panel = plot(p1, p2, p3, p4, layout = (2, 2), size = PLOT_CFG.size, dpi = PLOT_CFG.dpi)

    save_figure(panel, "covid_panel_2x2"; dir = PATHS.figures_covid)
    return panel
end

# ============================================================
# ОСНОВНОЙ ПРОЦЕСС
# ============================================================
function main()
    println("\n", "=" ^ 70)
    println("  АНАЛИЗ ДАННЫХ COVID-19")
    println("=" ^ 70)

    # 1. Инициализация
    setup_project()
    set_gr_backend!()

    # 2. Загрузка данных
    df = load_covid_data()

    # 3. Расчёт производных метрик
    df = compute_metrics(df)

    # 4. Сводная статистика
    print_summary(df)

    # 5. Графики
    println("\nПостроение графиков...")

    plot_daily_cases(df)
    println("  ✓ covid_daily_cases")

    plot_cumulative_cases(df)
    println("  ✓ covid_cumulative")

    plot_deaths_daily(df)
    println("  ✓ covid_deaths_daily")

    plot_combined_panel(df)
    println("  ✓ covid_panel_2x2")

    println("\n" * "=" ^ 70)
    println("  ГОТОВО! Графики сохранены в: $(PATHS.figures_covid)")
    println("=" ^ 70, "\n")

    return df
end

# Запуск
main()
