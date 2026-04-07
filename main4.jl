using JLD2, CSV, DataFrames, Dates, Plots, Peaks
using DataFrames, SavitzkyGolay, DSP,LinearAlgebra

# Load the data from the JLD2 file
df = load("./data_out/covid_data_with_waves_and_derivatives.jld2")["df"]

println("Data loaded successfully from JLD2 file.")
println("DataFrame size: ", size(df))
println("First 5 rows of the DataFrame:")
first(df, 5) |> println
# Optionally, you can also check the column names and types
println("Column names: ", names(df))
println("Column types: ", eltype.(eachcol(df)))

# Load stamms from CSV file
stamms = CSV.read("csv/covid19_seird_params.csv", DataFrame)
# and save to JLD2 for later use 
@save "./data_out/covid19_seird_params.jld2" stamms

#plot timeline of stamms based on strain_id, dom_start_adj and dom_end_adj and with R0_avg
p1 = plot(stamms.dom_start_adj, 
          stamms.R0_avg, 
          label="R0_avg", 
          xlabel="Date", ylabel="R0_avg",
          title="Timeline of R0_avg for COVID-19 Strains",
          legend=:topright)

# вертикальные линии
for i in 1:nrow(stamms)
    vline!(p1, [stamms.dom_start_adj[i]], color=:green, linestyle=:dash, label=nothing)
    vline!(p1, [stamms.dom_end_adj[i]], color=:red, linestyle=:dash, label=nothing)
end

# создаём вторую ось
p2 = twinx(p1)

plot!(p2, df.date, df.daily_interp_smooth,
      label="Total Cases",
      color=:blue,
      ylabel="Total Cases")

display(p1)
savefig(p1, "./figures/covid/timeline_R0_avg_and_total_cases.png")
savefig(p1, "./figures/covid/timeline_R0_avg_and_total_cases.svg")

println("Timeline plot of R0_avg and Total Cases saved successfully.")

#========================================================================#
# Панель параметров вирусов во времени (тольок средние значения), несколько графиков, расположенных по вертикали
# с маркерами волн и с вертикальными линиями, обозначающими начало и конец доминирования штамма
#========================================================================#
p3 = plot(stamms.dom_start_adj, 
          stamms.R0_avg, 
          label="R0_avg", 
          xlabel="Date", ylabel="R0_avg",
          title="Timeline of R0_avg for COVID-19 Strains",
          legend=:topright)
for i in 1:nrow(stamms)
    vline!(p3, [stamms.dom_start_adj[i]], color=:green, linestyle=:dash, label=nothing)
    vline!(p3, [stamms.dom_end_adj[i]], color=:red, linestyle=:dash, label=nothing)
end
p4 = plot(stamms.dom_start_adj, 
          stamms.T_infect_avg_days, 
          label="Infectious Period", 
          xlabel="Date", ylabel="Infectious Period (days)",
          title="Timeline of Infectious Period for COVID-19 Strains",
          legend=:topright)
for i in 1:nrow(stamms)
    vline!(p4, [stamms.dom_start_adj[i]], color=:green, linestyle=:dash, label=nothing)
    vline!(p4, [stamms.dom_end_adj[i]], color=:red, linestyle=:dash, label=nothing)
end
p5 = plot(stamms.dom_start_adj, 
          stamms.T_incub_avg_days, 
          label="Latent Period", 
          xlabel="Date", ylabel="Latent Period (days)",
          title="Timeline of Latent Period for COVID-19 Strains",
          legend=:topright) 
for i in 1:nrow(stamms)
    vline!(p5, [stamms.dom_start_adj[i]], color=:green, linestyle=:dash, label=nothing)
    vline!(p5, [stamms.dom_end_adj[i]], color=:red, linestyle=:dash, label=nothing)
end
p6 = plot(stamms.dom_start_adj, 
          stamms.CFR_pct, 
          label="Case Fatality Rate", 
          xlabel="Date", ylabel="Case Fatality Rate (%)",
          title="Timeline of Case Fatality Rate for COVID-19 Strains",
          legend=:topright)
for i in 1:nrow(stamms)
    vline!(p6, [stamms.dom_start_adj[i]], color=:green, linestyle=:dash, label=nothing)
    vline!(p6, [stamms.dom_end_adj[i]], color=:red, linestyle=:dash, label=nothing)
end
# Объединяем графики в одну панель  
p_combined = plot(p3, p4, p5, p6, layout=(4, 1), size=(800, 1200))
display(p_combined)
savefig(p_combined, "./figures/covid/timeline_strain_parameters.png")
savefig(p_combined, "./figures/covid/timeline_strain_parameters.svg")
println("Timeline plot of strain parameters saved successfully.")

#========================================================================#
# Теперь соединяем данные и календарь шшттаммов, чтобы показать, как параметры штаммов влияют на динамику случаев
#========================================================================#
# --- 1. Нормализуем даты в календаре штаммов
stamms_annot = copy(stamms)

if !(eltype(stamms_annot.dom_start_adj) <: Date)
    stamms_annot.dom_start_adj = Date.(stamms_annot.dom_start_adj)
end

if !(eltype(stamms_annot.dom_end_adj) <: Date)
    stamms_annot.dom_end_adj = Date.(stamms_annot.dom_end_adj)
end

# На случай сортируем по началу интервала
sort!(stamms_annot, :dom_start_adj)

# --- 2. Выберем поля штамма, которые нужны для SEIRD и анализа
strain_cols = [
    :strain_id,
    :num,
    :strain_name,
    :pango_lineage,
    :dom_start_adj,
    :dom_end_adj,
    :R0_min,
    :R0_avg,
    :R0_max,
    :T_incub_avg_days,
    :sigma_avg_per_day,
    :T_infect_avg_days,
    :gamma_avg_per_day,
    :IFR_avg,
    :mu_avg_per_day,
    :beta_min_per_day,
    :beta_avg_per_day,
    :beta_max_per_day,
    :T_gen_avg_days,
    :severity,
    :immune_escape
]

# Оставляем только реально существующие колонки
strain_cols = filter(c -> c in names(stamms_annot), strain_cols)

# --- 3. Построим "дневной календарь штаммов":
# одна дата -> один активный штамм и его свойства
daily_rows = NamedTuple[]

for row in eachrow(stamms_annot)
    d1 = row.dom_start_adj
    d2 = row.dom_end_adj

    for d in d1:Day(1):d2
        push!(daily_rows, (
            date = d,
            strain_id = getproperty(row, :strain_id),
            strain_num = (:num in propertynames(row)) ? getproperty(row, :num) : missing,
            strain_name = (:strain_name in propertynames(row)) ? getproperty(row, :strain_name) : missing,
            pango_lineage = (:pango_lineage in propertynames(row)) ? getproperty(row, :pango_lineage) : missing,
            strain_dom_start = d1,
            strain_dom_end = d2,
            R0_min = (:R0_min in propertynames(row)) ? getproperty(row, :R0_min) : missing,
            R0_avg = (:R0_avg in propertynames(row)) ? getproperty(row, :R0_avg) : missing,
            R0_max = (:R0_max in propertynames(row)) ? getproperty(row, :R0_max) : missing,
            T_incub_avg_days = (:T_incub_avg_days in propertynames(row)) ? getproperty(row, :T_incub_avg_days) : missing,
            sigma_avg_per_day = (:sigma_avg_per_day in propertynames(row)) ? getproperty(row, :sigma_avg_per_day) : missing,
            T_infect_avg_days = (:T_infect_avg_days in propertynames(row)) ? getproperty(row, :T_infect_avg_days) : missing,
            gamma_avg_per_day = (:gamma_avg_per_day in propertynames(row)) ? getproperty(row, :gamma_avg_per_day) : missing,
            IFR_avg = (:IFR_avg in propertynames(row)) ? getproperty(row, :IFR_avg) : missing,
            mu_avg_per_day = (:mu_avg_per_day in propertynames(row)) ? getproperty(row, :mu_avg_per_day) : missing,
            beta_min_per_day = (:beta_min_per_day in propertynames(row)) ? getproperty(row, :beta_min_per_day) : missing,
            beta_avg_per_day = (:beta_avg_per_day in propertynames(row)) ? getproperty(row, :beta_avg_per_day) : missing,
            beta_max_per_day = (:beta_max_per_day in propertynames(row)) ? getproperty(row, :beta_max_per_day) : missing,
            T_gen_avg_days = (:T_gen_avg_days in propertynames(row)) ? getproperty(row, :T_gen_avg_days) : missing,
            severity = (:severity in propertynames(row)) ? getproperty(row, :severity) : missing,
            immune_escape = (:immune_escape in propertynames(row)) ? getproperty(row, :immune_escape) : missing
        ))
    end
end

strain_daily = DataFrame(daily_rows)

# --- 4. Если интервалы штаммов пересекаются, на одну дату может прийтись >1 штамма.
# Тогда выбираем один по правилу:
# "побеждает штамм с более поздним dom_start_adj" (более новый доминирующий вариант).
if nrow(strain_daily) > 0
    g = groupby(strain_daily, :date)

    strain_daily_resolved = combine(g) do sdf
        sort!(sdf, [:strain_dom_start, :strain_num], rev = true)

        chosen = sdf[1, :]
        nmatch = nrow(sdf)

        DataFrame(
            date = [chosen.date],
            strain_id = [chosen.strain_id],
            strain_num = [chosen.strain_num],
            strain_name = [chosen.strain_name],
            pango_lineage = [chosen.pango_lineage],
            strain_dom_start = [chosen.strain_dom_start],
            strain_dom_end = [chosen.strain_dom_end],
            R0_min = [chosen.R0_min],
            R0_avg = [chosen.R0_avg],
            R0_max = [chosen.R0_max],
            T_incub_avg_days = [chosen.T_incub_avg_days],
            sigma_avg_per_day = [chosen.sigma_avg_per_day],
            T_infect_avg_days = [chosen.T_infect_avg_days],
            gamma_avg_per_day = [chosen.gamma_avg_per_day],
            IFR_avg = [chosen.IFR_avg],
            mu_avg_per_day = [chosen.mu_avg_per_day],
            beta_min_per_day = [chosen.beta_min_per_day],
            beta_avg_per_day = [chosen.beta_avg_per_day],
            beta_max_per_day = [chosen.beta_max_per_day],
            T_gen_avg_days = [chosen.T_gen_avg_days],
            severity = [chosen.severity],
            immune_escape = [chosen.immune_escape],
            n_matching_strains = [nmatch],
            strain_match_type = [nmatch == 1 ? "exact" : "overlap_resolved"]
        )
    end
else
    strain_daily_resolved = DataFrame(
        date = Date[],
        strain_id = String[],
        strain_num = Int[],
        strain_name = String[],
        pango_lineage = String[],
        strain_dom_start = Date[],
        strain_dom_end = Date[],
        R0_min = Float64[],
        R0_avg = Float64[],
        R0_max = Float64[],
        T_incub_avg_days = Float64[],
        sigma_avg_per_day = Float64[],
        T_infect_avg_days = Float64[],
        gamma_avg_per_day = Float64[],
        IFR_avg = Float64[],
        mu_avg_per_day = Float64[],
        beta_min_per_day = Float64[],
        beta_avg_per_day = Float64[],
        beta_max_per_day = Float64[],
        T_gen_avg_days = Float64[],
        severity = String[],
        immune_escape = String[],
        n_matching_strains = Int[],
        strain_match_type = String[]
    )
end

# Сохраняем результат для дальнейшего использования в JLD2 файле и CSV
@save "./data_out/strain_daily_resolved.jld2" strain_daily_resolved
CSV.write("./data_out/strain_daily_resolved.csv", strain_daily_resolved)
println("Daily resolved strain calendar created and saved successfully.")

# --- 5. Присоединяем календарь штаммов к основному df по date
df_enriched = leftjoin(df, strain_daily_resolved, on = :date)

# Сохраняем результат для дальнейшего использования в JLD2 файле и CSV
@save "./data_out/covid_data_enriched_with_strains.jld2" df_enriched
CSV.write("./data_out/covid_data_enriched_with_strains.csv", df_enriched)
println("Enriched COVID data with strain information created and saved successfully.")

# --- 6. Строим график: динамика ежедневных случаев (или других метрик) с аннотациями по штаммам
p7 = plot(df_enriched.date, df_enriched.daily_interp_smooth,
      label="Total Cases",
      color=:blue,
      xlabel="Date", ylabel="Total Cases",
      title="Daily Cases with Strain Annotations")
for i in 1:nrow(stamms)
    vline!(p7, [stamms.dom_start_adj[i]], color=:green, linestyle=:dash, label=nothing)
    vline!(p7, [stamms.dom_end_adj[i]], color=:red, linestyle=:dash, label=nothing)
end
# Добавь по правой оси параметр штамма (не R0_avg, а, например, severity или immune_escape) для наглядности
p8 = twinx(p7)
plot!(p8, df_enriched.date, df_enriched.severity,
      label="Strain Severity",
      color=:orange,
      ylabel="Strain Severity")

display(p7)
savefig(p7, "./figures/covid/daily_cases_with_strain_annotations.png")
savefig(p7, "./figures/covid/daily_cases_with_strain_annotations.svg")
println("Daily cases plot with strain annotations saved successfully.")