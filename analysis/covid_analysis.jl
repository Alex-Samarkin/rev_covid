# ============================================================
# analysis/covid_analysis.jl — Процедуры анализа данных COVID-19
# ============================================================

using Dates
using CSV
using DataFrames
using Statistics

include(joinpath(@__DIR__, "..", "config", "utils.jl"))

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
"""
    load_covid_data(path::String = PATHS.csv_covid)::DataFrame

Загружает CSV COVID-19, парсит даты, очищает от пропусков.
"""
function load_covid_data(path::String = PATHS.csv_covid)::DataFrame
    if !isfile(path)
        error("Файл данных не найден: $path")
    end

    df = CSV.read(path, DataFrame, dateformat = "yyyy-mm-dd")
    rename!(df, lowercase.(names(df)))  # унификация имён колонок

    # Парсинг дат
    df[!, :дата] = Date.(df[!, :дата], dateformat"yyyy-mm-dd")

    # Преобразование строковых "MD" → missing, а числа → Float64
    for col in names(df)
        if col == :дата
            continue
        end
        df[!, col] = allowmissing(df[!, col])
        for i in 1:nrow(df)
            val = df[i, col]
            if val isa AbstractString && val == "MD"
                df[i, col] = missing
            elseif val isa AbstractString
                try
                    df[i, col] = parse(Float64, val)
                catch
                    df[i, col] = missing
                end
            end
        end
    end

    n_total = nrow(df)
    n_valid = count(.!ismissing.(df[!, :всего_накоп_]))
    @info "COVID данные загружены: $n_total строк, $n_valid с данными"

    return df
end

# ============================================================
# ПРОИЗВОДНЫЕ МЕТРИКИ
# ============================================================
"""
    compute_metrics(df::DataFrame)::DataFrame

Добавляет производные метрики: 7-дневное скользящее среднее.
"""
function compute_metrics(df::DataFrame)::DataFrame
    out = copy(df)

    # Скользящее среднее (7 дней) для "за сутки"
    col_daily = :за_сутки
    col_smooth = :за_сутки_ma7

    out[!, col_smooth] = allowmissing(out[!, col_smooth])

    daily_vals = out[!, col_daily]
    n = length(daily_vals)

    for i in 1:n
        window_start = max(1, i - 6)
        window = [daily_vals[j] for j in window_start:i if !ismissing(daily_vals[j])]
        if !isempty(window)
            out[i, col_smooth] = mean(window)
        else
            out[i, col_smooth] = missing
        end
    end

    @info "Добавлено скользящее среднее (7 дней)"
    return out
end

# ============================================================
# АГРЕГАЦИЯ ПО ПЕРИОДАМ
# ============================================================
"""
    aggregate_monthly(df::DataFrame)::DataFrame

Агрегация данных по месяцам.
"""
function aggregate_monthly(df::DataFrame)::DataFrame
    df_clean = dropmissing(df, :всего_накоп_)
    df_clean[!, :год] = year.(df_clean[!, :дата])
    df_clean[!, :месяц] = month.(df_clean[!, :дата])

    monthly = combine(
        groupby(df_clean, [:год, :месяц]),
        :за_сутки => (x -> sum(skipmissing(x))) => :за_месяц,
        :летальных_за_сутки => (x -> sum(skipmissing(x))) => :летальных_за_месяц,
    )

    sort!(monthly, [:год, :месяц])
    @info "Агрегация по месяцам: $(nrow(monthly)) периодов"
    return monthly
end

# ============================================================
# СТАТИСТИКА
# ============================================================
"""
    print_summary(df::DataFrame)

Выводит сводную статистику по данным COVID-19.
"""
function print_summary(df::DataFrame)
    println("\n", "=" ^ 70)
    println("  СВОДНАЯ СТАТИСТИКА COVID-19")
    println("=" ^ 70)

    dates = df[!, :дата]
    println("  Период: $(first(dates)) — $(last(dates))")
    println("  Всего дней: $(nrow(df))")

    daily = skipmissing(df[!, :за_сутки])
    daily_vec = collect(daily)

    if !isempty(daily_vec)
        println("\n  Заболевания за сутки:")
        println("    Среднее:  $(round(mean(daily_vec), digits=1))")
        println("    Медиана:  $(round(median(daily_vec), digits=1))")
        println("    Максимум: $(round(maximum(daily_vec), digits=1)) ($(findmax(daily_vec)[2])-й день)")
        println("    Минимум:  $(round(minimum(daily_vec), digits=1))")
    end

    deaths = skipmissing(df[!, :летальных_за_сутки])
    deaths_vec = collect(deaths)

    if !isempty(deaths_vec)
        println("\n  Летальные случаи за сутки:")
        println("    Среднее:  $(round(mean(deaths_vec), digits=1))")
        println("    Медиана:  $(round(median(deaths_vec), digits=1))")
        println("    Максимум: $(maximum(deaths_vec))")
    end

    last_total = df[end, :всего_накоп_]
    if !ismissing(last_total)
        println("\n  Итого заболевших (накоп.): $(round(Int, last_total))")
    end

    last_deaths = df[end, :летальных_всего]
    if !ismissing(last_deaths)
        println("  Итого летальных (накоп.): $(round(Int, last_deaths))")
        lethality = round(last_deaths / last_total * 100, digits=2)
        println("  Летальность: $(lethality)%")
    end

    println("=" ^ 70, "\n")
    return nothing
end
