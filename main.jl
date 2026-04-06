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

    n = nrow(df)
    step = max(1, cld(n, 50))  # прореживание для баров
    idx = 1:step:n

    p = bar(
        idx,
        df[!, :за_сутки][idx],
        label      = "За сутки",
        color      = :steelblue,
        alpha      = 0.7,
        ylabel     = "Число случаев",
        xlabel     = "Дата",
        title      = "Заболевания COVID-19 за сутки",
        markersize = 0,
        xrotation  = 45,
    )

    # Скользящее среднее
    if :за_сутки_ma7 in names(df)
        plot!(
            p,
            idx,
            df[!, :за_сутки_ma7][idx],
            seriestype  = :line,
            label       = "Скользящее среднее (7 дн.)",
            color       = :red,
            linewidth   = 3,
        )
    end

    # Подписи дат на оси X
    tick_idx = collect(1:step:n)
    n_labels = min(12, length(tick_idx))
    label_step = max(1, cld(length(tick_idx), n_labels))
    label_idx = tick_idx[1:label_step:end]
    xticks!(p, label_idx, string.(df[!, :дата])[label_idx])
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

    df_clean = dropmissing(df, :всего_накоп)

    p = plot(
        df_clean[!, :дата],
        df_clean[!, :всего_накоп],
        seriestype = :line,
        label      = "Всего (накоп.)",
        color      = :navy,
        linewidth  = 3,
        fill       = (0, 0.15, :navy),
        ylabel     = "Число случаев",
        xlabel     = "Дата",
        title      = "Накопительная заболеваемость COVID-19",
        xrotation  = 45,
    )

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

    n = nrow(df)
    step = max(1, cld(n, 50))
    idx = 1:step:n

    p = bar(
        idx,
        df[!, :летальных_за_сутки][idx],
        label      = "Летальных за сутки",
        color      = :crimson,
        alpha      = 0.7,
        ylabel     = "Число случаев",
        xlabel     = "Дата",
        title      = "Летальные случаи COVID-19 за сутки",
        markersize = 0,
        xrotation  = 45,
    )

    tick_idx = collect(1:step:n)
    n_labels = min(12, length(tick_idx))
    label_step = max(1, cld(length(tick_idx), n_labels))
    label_idx = tick_idx[1:label_step:end]
    xticks!(p, label_idx, string.(df[!, :дата])[label_idx])
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

    df_clean = dropmissing(df, :всего_накоп)

    # --- 1. За сутки + скользящее среднее ---
    n = nrow(df)
    step = max(1, cld(n, 50))
    idx = 1:step:n
    p1 = bar(
        idx, df[!, :за_сутки][idx],
        label = "За сутки", color = :steelblue, alpha = 0.7,
        ylabel = "Случаев/день", title = "A) За сутки", xrotation = 45, markersize = 0,
    )
    if :за_сутки_ma7 in names(df)
        plot!(p1, idx, df[!, :за_сутки_ma7][idx],
              seriestype = :line, label = "MA(7)", color = :red, linewidth = 2)
    end

    # --- 2. Накопительные ---
    p2 = plot(
        df_clean[!, :дата], df_clean[!, :всего_накоп],
        seriestype = :line, label = "Всего", color = :navy, linewidth = 3,
        fill = (0, 0.15, :navy), ylabel = "Всего случаев", title = "B) Накопительно",
        xrotation = 45,
    )

    # --- 3. Летальные за сутки ---
    n3 = nrow(df)
    step3 = max(1, cld(n3, 50))
    idx3 = 1:step3:n3
    p3 = bar(
        idx3, df[!, :летальных_за_сутки][idx3],
        label = "Летальных/день", color = :crimson, alpha = 0.7,
        ylabel = "Летальных/день", xlabel = "Дата", title = "C) Летальные за сутки",
        xrotation = 45, markersize = 0,
    )

    # --- 4. Летальные накопительно ---
    p4 = plot(
        df_clean[!, :дата], df_clean[!, :летальных_всего],
        seriestype = :line, label = "Летальных всего", color = :darkred, linewidth = 3,
        fill = (0, 0.1, :darkred), ylabel = "Летальных всего", xlabel = "Дата",
        title = "D) Летальные накопительно", xrotation = 45,
    )

    # --- Прореживание X-подписей (авто-тики Plots + поворот) ---
    # xrotation задан при создании каждого subplot
    for p_obj in (p1, p2, p3, p4)
        apply_theme!(p_obj)
    end

    # Компоновка 2×2
    panel = plot(p1, p2, p3, p4, layout = (2, 2), size = PLOT_CFG.size)

    # GR нужен display перед сохранением layout
    display(panel)
    yield()  # дать GR отрендерить
    sleep(1)

    save_figure(panel, "covid_panel_2x2"; dir = PATHS.figures_covid)
    return panel
end

#= Написать функцию, которая заменяет в df русские названия колонок нра их простые английскиее эквиваленты  =#
using JLD2, Parquet
function save_data(df::DataFrame, filename::String, out_dir::String = "data_out")
    # combine out_dir, filename and extension "jld2"
    jld2file = joinpath(out_dir, filename * ".jld2")
    jld2file2 = joinpath("./", filename * ".jld2")
    JLD2.save(jld2file, "df", df)
    JLD2.save(jld2file2, "df", df)
    # also save as parquet
   
    # Копируем df и конвертируем даты в String
    df_parquet = copy(df)
    for col in names(df_parquet)
        if eltype(df_parquet[!, col]) <: Date
            df_parquet[!, col] = string.(df_parquet[!, col])
        end
    end
    
    parquetfile = joinpath(out_dir, filename * ".parquet")
    Parquet.write_parquet(parquetfile, df_parquet)

    @info "Data saved to: $jld2file and $parquetfile"
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

    # 6. Перевод колонок на английский
    println("\nПеревод колонок на английский...")
    translate_columns!(df)
    println("  Колонки: $(names(df))")

    println("\n" * "=" ^ 70)
    println("  ГОТОВО! Графики сохранены в: $(PATHS.figures_covid)")
    println("=" ^ 70, "\n")

    save_data(df, "covid_data_translated", PATHS.data_out)

    return df
end

# Запуск
main()
