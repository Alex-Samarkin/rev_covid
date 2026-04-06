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

    df = CSV.read(path, DataFrame, dateformat = "yyyy-mm-dd", missingstring = "MD")

    # Переименование колонок в удобные символы
    rename!(df, Dict(
        "Дата"            => :дата,
        "Всего (накоп.)"  => :всего_накоп,
        "За сутки"        => :за_сутки,
        "Летальных всего" => :летальных_всего,
        "Летальных за сутки" => :летальных_за_сутки,
    ))

    n_total = nrow(df)
    n_valid = count(.!ismissing.(df[!, :всего_накоп]))
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
    out[!, :за_сутки_ma7] = Vector{Union{Missing, Float64}}(missing, nrow(out))

    daily_vals = out[!, :за_сутки]
    n = length(daily_vals)

    for i in 1:n
        window_start = max(1, i - 6)
        window = [daily_vals[j] for j in window_start:i if !ismissing(daily_vals[j])]
        if !isempty(window)
            out[i, :за_сутки_ma7] = mean(window)
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
    df_clean = dropmissing(df, :всего_накоп)
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

    last_total = df[end, :всего_накоп]
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

# ============================================================
# ПЕРЕВОД КОЛОНОК
# ============================================================
"""
    translate_columns!(df::DataFrame)::DataFrame

Заменяет русские названия колонок на простые английские эквиваленты.
"""
function translate_columns!(df::DataFrame)::DataFrame
    RU_TO_EN = Dict(
        :дата              => :date,
        :всего_накоп       => :total,
        :за_сутки          => :daily,
        :летальных_всего   => :deaths_total,
        :летальных_за_сутки => :deaths_daily,
        :за_сутки_ma7      => :daily_ma7,
        :год               => :year,
        :месяц             => :month,
        :за_месяц          => :monthly,
        :летальных_за_месяц => :deaths_monthly,
    )

    renames = Dict{Symbol, Symbol}()
    for col in names(df)
        sym = Symbol(col)
        if haskey(RU_TO_EN, sym)
            renames[sym] = RU_TO_EN[sym]
        end
    end

    if !isempty(renames)
        rename!(df, renames)
        @info "Переименованы колонки: $(collect(keys(renames))) → $(collect(values(renames)))"
    end

    return df
end
