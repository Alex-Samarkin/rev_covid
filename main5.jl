using Revise, JLD2, CSV, DataFrames, Dates, Plots, Peaks
using DataFrames, SavitzkyGolay, DSP,LinearAlgebra
using Printf

# Load the data from the JLD2 file
df = load("./data_out/covid_data_enriched_with_strains.jld2")["df_enriched"]

println("Data loaded successfully from JLD2 file.")
println("DataFrame size: ", size(df))
println("First 5 rows of the DataFrame:")
first(df, 5) |> println
# Optionally, you can also check the column names and types
println("Column names: ", names(df))
println("Column types: ", eltype.(eachcol(df)))

#========================================================================#
# Step 1: choose a wave number, filter data, and plot cases for that wave
#========================================================================#
wave_num = 1

if !("wave" in names(df))
    error("Column :wave was not found in df. Available columns: $(names(df))")
end

df_wave = filter("wave" => ==(wave_num), df)
sort!(df_wave, "date")

if nrow(df_wave) == 0
    error("No data found for wave = $wave_num. Available waves: $(sort(unique(df.wave)))")
end

println("Filtered wave = $wave_num")
println("Rows in wave subset: ", nrow(df_wave))
println("Date range: ", minimum(df_wave.date), " .. ", maximum(df_wave.date))




p_wave = plot(
    df_wave.date,
    df_wave.daily,
    label = "Daily cases (raw)",
    color = :steelblue,
    alpha = 0.6,
    linewidth = 1.5,
    xlabel = "Date",
    ylabel = "Cases",
    title = "COVID cases: wave $wave_num",
    legend = :topright,
    size = (1980, 1114),
    titlefontsize = 28,
    guidefontsize = 22,
    tickfontsize = 18,
    legendfontsize = 18
)

plot!(
    p_wave,
    df_wave.date,
    df_wave.daily_interp_smooth,
    label = "Daily cases (smoothed)",
    color = :navy,
    linewidth = 2.0
)

# добавь вертикальную линию для пика волны
peak_idx = findmax(df_wave.daily_interp_smooth)[2]
peak_date = df_wave.date[peak_idx]
vline!(p_wave, [peak_date], label = "Peak date", color = :red, linestyle = :dash)
# добавь аннотацию для пика
annotate!(p_wave, peak_date, df_wave.daily_interp_smooth[peak_idx], text("Peak: $(peak_date)", :red, :left, 20))    

# добавь полосы со штаммамми
strains = unique(df_wave.strain_id)
colors = [:orange, :green, :purple, :cyan, :magenta]
for (i, strain) in enumerate(strains)
    df_strain = filter("strain_id" => ==(strain), df_wave)
    if nrow(df_strain) > 0
        plot!( 
            p_wave,
            df_strain.date,
            df_strain.daily_interp_smooth,
            label = "Strain: $strain",
            color = colors[mod1(i, length(colors))],
            linewidth = 1.5*30,
            alpha = 0.25
        )
    end
end



display(p_wave)
savefig(p_wave, "./figures/covid/wave_$(wave_num)_cases.png")
savefig(p_wave, "./figures/covid/wave_$(wave_num)_cases.svg")
println("Wave plot saved: ./figures/covid/wave_$(wave_num)_cases.(png|svg)")

#========================================================================#
# Step 2: Подготить данные для расчета этой волны как реализации SEIRD модели
#========================================================================#
# Здесь мы будем использовать данные о ежедневных случаях (daily_interp) и ежедневных смертях (deaths_daily или 0 если данные отсутствуют) для выбранной волны. 
# Мы также будем использовать сглаженные данные (daily_interp_smooth) для анализа.
# Создадим новый DataFrame для SEIRD модели
# "R0_avg", "T_incub_avg_days", "sigma_avg_per_day", "T_infect_avg_days", "gamma_avg_per_day", "IFR_avg", "mu_avg_per_day", "beta_avg_per_day", "T_gen_avg_days", "severity", "immune_escape"
df_seird = DataFrame(
    date = df_wave.date,
    daily_cases = df_wave.daily_interp,
    daily_deaths = coalesce.(df_wave.deaths_daily, 0), # заменаем пропущенные значения смертей на 0
    daily_cases_smooth = df_wave.daily_interp_smooth,
    strain_id = df_wave.strain_id,
    R0_avg = df_wave.R0_avg,
    T_incub_avg_days = df_wave.T_incub_avg_days,
    sigma_avg_per_day = df_wave.sigma_avg_per_day,
    T_infect_avg_days = df_wave.T_infect_avg_days,
    gamma_avg_per_day = df_wave.gamma_avg_per_day,
    IFR_avg = df_wave.IFR_avg,
    mu_avg_per_day = df_wave.mu_avg_per_day,
    beta_avg_per_day = df_wave.beta_avg_per_day,
    T_gen_avg_days = df_wave.T_gen_avg_days,
    severity = df_wave.severity,
    immune_escape = df_wave.immune_escape
)
println("SEIRD DataFrame prepared:")
println("Columns: ", names(df_seird))
println("First 5 rows:")
first(df_seird, 5) |> println

# Теперь у нас есть DataFrame df_seird, который содержит все необходимые данные для анализа волны с помощью SEIRD модели.
# Мы можем использовать эти данные для дальнейшего анализа, например, для оценки параметров модели или для сравнения с реальными данными.
# В следующем шаге мы можем приступить к расчету параметров SEIRD модели на основе этих данных.
# используя сведения о штамме

using DifferentialEquations
using Optim


# ============================================================
# main6.jl — SEIRD-модель для одной волны COVID-19
#
# Зависимости (первый запуск):
#   Pkg.add(["DifferentialEquations", "Optim", "Plots", "JLD2",
#            "DataFrames", "CSV", "Dates"])
#
# Структура:
#   1. Загрузка данных волны (df_seird из main5)
#   2. Инициализация параметров из календаря штаммов
#   3. SEIRD ODE (правые части + Jacobian для rigid-решателей)
#   4. Прямое решение с параметрами штамма (prior run)
#   5. Подгонка параметров: β, σ, γ, μ, E₀, I₀ через Optim/NelderMead
#   6. Визуализация: данные vs модель, сходимость потерь
#   7. Сохранение результатов
# ============================================================

using JLD2, CSV, DataFrames, Dates
using DifferentialEquations          # Tsit5 (адаптивный RK4/5 Dormand-Prince)
using Optim                          # NelderMead + LBFGS для подгонки
using Plots
using Statistics: mean
using LinearAlgebra

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 1: Загрузка данных
# ─────────────────────────────────────────────────────────────

df_wave = CSV.read("./data_out/covid_data_enriched_with_strains.csv", DataFrame)

# Фильтруем нужную волну
WAVE_NUM = 1

df_wave = filter(:wave => ==(WAVE_NUM), df_wave)
sort!(df_wave, :date)
nrow(df_wave) > 0 || error("Волна $WAVE_NUM не найдена в данных")

println("Волна $WAVE_NUM: $(nrow(df_wave)) дней, " *
        "$(minimum(df_wave.date)) .. $(maximum(df_wave.date))")

# Наблюдаемые временные ряды (в абсолютных случаях)
# daily_interp — интерполированный поток новых заражений в день
obs_cases  = Float64.(coalesce.(df_wave.daily_interp, 0.0))
obs_deaths = Float64.(coalesce.(df_wave.deaths_daily, 0.0))
T          = nrow(df_wave)          # длина волны в днях

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 2: Константы и prior-параметры штамма
#
# Параметры читаем из первой строки df_wave (для волны они постоянны
# или слабо меняются — берём среднее за период доминирования).
# ─────────────────────────────────────────────────────────────

const N_POP = 100_000.0   # население Пскова (для нормировки S/N)

# Средние параметры штамма за период волны
σ_prior = mean(skipmissing(df_wave.sigma_avg_per_day))  # 1/T_incub
γ_prior = mean(skipmissing(df_wave.gamma_avg_per_day))  # 1/T_infect
μ_prior = mean(skipmissing(df_wave.mu_avg_per_day))     # IFR·γ ≈ смертность
β_prior = mean(skipmissing(df_wave.beta_avg_per_day))   # R0·γ

println("\n=== Prior-параметры штамма ===")
@printf "  β  = %.5f  (R0·γ)\n" β_prior
@printf "  σ  = %.5f  (1/T_incub)\n" σ_prior
@printf "  γ  = %.5f  (1/T_infect)\n" γ_prior
@printf "  μ  = %.6f  (death rate)\n" μ_prior

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 3: SEIRD ODE
#
#   dS/dt = -β·S·I / N                      (убыль восприимчивых)
#   dE/dt =  β·S·I / N  -  σ·E              (инкубация)
#   dI/dt =  σ·E  -  (γ+μ)·I               (инфекционный период)
#   dR/dt =  γ·I                             (выздоровление)
#   dD/dt =  μ·I                             (смерти)
#
# Состояние u = [S, E, I, R, D]  (абсолютные числа)
# Параметры p = (β, σ, γ, μ)
# ─────────────────────────────────────────────────────────────

function seird!(du, u, p, t)
    S, E, I, R, D = u
    β, σ, γ, μ    = p

    # Новые экспозиции за шаг dt
    new_exp = β * S * I / N_POP

    du[1] = -new_exp                   # dS
    du[2] =  new_exp  - σ * E          # dE
    du[3] =  σ * E    - (γ + μ) * I   # dI
    du[4] =  γ * I                     # dR
    du[5] =  μ * I                     # dD
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 4: Вспомогательные функции
# ─────────────────────────────────────────────────────────────

"""
    run_seird(p_vec, u0; tspan=(0., T-1)) -> solution

Решает SEIRD с заданными параметрами p_vec=(β,σ,γ,μ)
и начальными условиями u0=[S,E,I,R,D].
Возвращает DifferentialEquations.Solution, сохранённую в целые дни.
"""
function run_seird(p_vec, u0; tspan = (0.0, Float64(T - 1)))
    prob = ODEProblem(seird!, u0, tspan, p_vec)
    # Tsit5 — адаптивный явный RK4(5), хорошо работает для гладких эпидемических ODE
    solve(prob, Tsit5();
          saveat    = 1.0,        # сохраняем каждый день
          abstol    = 1e-8,
          reltol    = 1e-6,
          maxiters  = 1_000_000)
end

"""
    model_observables(sol) -> (new_cases, new_deaths)

Извлекает наблюдаемые потоки из решения ODE:
  - новые случаи ≈ σ·E (поток E→I)
  - смерти       ≈ μ·I (поток I→D)
"""
function model_observables(sol, p_vec)
    σ = p_vec[2]
    μ = p_vec[4]
    # sol[k, t] — k-й компонент в момент t
    E = sol[2, :]   # вектор E по времени
    I = sol[3, :]   # вектор I по времени
    return σ .* E, μ .* I
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 5: Начальные условия
#
# E₀ и I₀ — параметры подгонки; S₀ выводится из баланса.
# R₀_init, D₀_init — малы в начале волны, оставляем как ~0
# (при желании можно оценить из накопленных данных до волны).
# ─────────────────────────────────────────────────────────────

# Грубая оценка: I₀ ≈ первый день сглаженных случаев / (γ+μ)
I0_prior = obs_cases[1] / (γ_prior + μ_prior + 1e-9) /4.0  # делим на 4, чтобы не переоценить старт (эвристика)
E0_prior = I0_prior * 2.0        # E чуть больше I на старте (эвристика)

function make_u0(E0, I0)
    S0 = max(0.0, N_POP - E0 - I0)   # все остальные — восприимчивы
    [S0, E0, I0, 0.0, 0.0]
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 6: Функция потерь (MSE по случаям + смертям)
#
# θ = log-пространство параметров, чтобы обеспечить θ > 0 без ограничений
# θ = [log β, log σ, log γ, log μ, log E₀, log I₀]
# ─────────────────────────────────────────────────────────────

# Веса потерь: deaths_weight > 1 компенсирует, что смертей мало
const DEATHS_WEIGHT = 10.0

function loss(θ::Vector{Float64})
    # Обратное преобразование из log-пространства
    β, σ, γ, μ, E0, I0 = exp.(θ)

    # Численная защита: убираем вырожденные конфигурации
    (β < 1e-9 || σ < 1e-9 || γ < 1e-9 || μ < 1e-9) && return 1e18
    (E0 < 1.0 || I0 < 1.0)                           && return 1e18

    p_vec = (β, σ, γ, μ)
    u0    = make_u0(E0, I0)

    sol = run_seird(p_vec, u0)

    # Если решатель не сошёлся — возвращаем большую потерю
    sol.retcode !== ReturnCode.Success && return 1e18

    mod_cases, mod_deaths = model_observables(sol, p_vec)

    # Обрезаем до длины наблюдений (могут не совпасть на 1 точку)
    n = min(length(mod_cases), T)

    # Нормализованный MSE (делим на максимум, чтобы шкалы совпали)
    mse_c = mean((mod_cases[1:n]  .- obs_cases[1:n]).^2)  / (maximum(obs_cases)^2 + 1)
    mse_d = mean((mod_deaths[1:n] .- obs_deaths[1:n]).^2) / (maximum(obs_deaths)^2 + 1)

    return mse_c + DEATHS_WEIGHT * mse_d
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 7: Prior run — модель с параметрами штамма (без подгонки)
# ─────────────────────────────────────────────────────────────

println("\n=== Prior run (параметры штамма) ===")
p_prior = (β_prior, σ_prior, γ_prior, μ_prior)
u0_prior = make_u0(E0_prior, I0_prior)
sol_prior = run_seird(p_prior, u0_prior)
println("  Статус решателя: $(sol_prior.retcode)")

cases_prior, deaths_prior = model_observables(sol_prior, collect(p_prior))

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 8: Оптимизация
#
# Шаг 1: NelderMead — грубый глобальный поиск, не требует градиентов
# Шаг 2: LBFGS    — уточнение в найденном минимуме (квази-Ньютон)
# ─────────────────────────────────────────────────────────────

# Начальная точка — prior параметры штамма в log-пространстве
θ0 = log.([β_prior, σ_prior, γ_prior, μ_prior, E0_prior, I0_prior])

println("\n=== Оптимизация параметров SEIRD ===")
println("  Начальная потеря (prior): $(round(loss(θ0); sigdigits=5))")

# --- Шаг 1: NelderMead (робастный старт) ---
res_nm = optimize(loss, θ0,
                  NelderMead(),
                  Optim.Options(
                      iterations  = 25_000,
                      show_trace  = false,
                      g_tol       = 1e-7))

θ_nm = Optim.minimizer(res_nm)
println("  После NelderMead : loss = $(round(Optim.minimum(res_nm); sigdigits=5))")

# --- Шаг 2: LBFGS уточнение (конечно-разностный градиент) ---
res_lbfgs = optimize(loss, θ_nm,
                     LBFGS(),
                     Optim.Options(
                         iterations  = 20_000,
                         show_trace  = false,
                         g_tol       = 1e-9))

θ_fit = Optim.minimizer(res_lbfgs)
β_fit, σ_fit, γ_fit, μ_fit, E0_fit, I0_fit = exp.(θ_fit)

println("  После LBFGS      : loss = $(round(Optim.minimum(res_lbfgs); sigdigits=5))")

println("\n=== Подобранные параметры ===")
@printf "  β_fit  = %.5f  (prior: %.5f, Δ=%.2f%%)\n" β_fit β_prior  100*(β_fit-β_prior)/β_prior
@printf "  σ_fit  = %.5f  (prior: %.5f, Δ=%.2f%%)\n" σ_fit σ_prior  100*(σ_fit-σ_prior)/σ_prior
@printf "  γ_fit  = %.5f  (prior: %.5f, Δ=%.2f%%)\n" γ_fit γ_prior  100*(γ_fit-γ_prior)/γ_prior
@printf "  μ_fit  = %.6f  (prior: %.6f, Δ=%.2f%%)\n" μ_fit μ_prior  100*(μ_fit-μ_prior)/μ_prior
@printf "  R0_fit = %.3f   (β/γ)\n"  β_fit / γ_fit
@printf "  E₀_fit = %.0f\n" E0_fit
@printf "  I₀_fit = %.0f\n" I0_fit

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 9: Финальное решение с подобранными параметрами
# ─────────────────────────────────────────────────────────────

p_fit = (β_fit, σ_fit, γ_fit, μ_fit)
u0_fit = make_u0(E0_fit, I0_fit)
sol_fit = run_seird(p_fit, u0_fit)

println("\n  Статус финального решения: $(sol_fit.retcode)")

cases_fit, deaths_fit = model_observables(sol_fit, collect(p_fit))
n_plot = min(length(cases_fit), T)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 10: Визуализация
# ─────────────────────────────────────────────────────────────

dates = df_wave.date[1:n_plot]

# --- График 1: ежедневные случаи (данные vs prior vs fit) ---
p_cases = plot(dates, obs_cases[1:n_plot];
               label     = "Наблюдения",
               color     = :steelblue,
               linewidth = 2,
               alpha     = 0.8,
               xlabel    = "Дата",
               ylabel    = "Новых случаев / день",
               title     = "SEIRD — волна $WAVE_NUM: ежедневные случаи",
               legend    = :topright,
               size      = (1200, 500),
               grid      = true,
               gridalpha = 0.3)

plot!(p_cases, dates, cases_prior[1:n_plot];
      label     = "Модель (prior штамма)",
      color     = :orange,
      linewidth = 1.5,
      linestyle = :dash)

plot!(p_cases, dates, cases_fit[1:n_plot];
      label     = "Модель (подобранные θ)",
      color     = :crimson,
      linewidth = 2)

# --- График 2: ежедневные смерти ---
p_deaths = plot(dates, obs_deaths[1:n_plot];
                label     = "Наблюдения",
                color     = :gray50,
                linewidth = 2,
                alpha     = 0.8,
                xlabel    = "Дата",
                ylabel    = "Смертей / день",
                title     = "SEIRD — волна $WAVE_NUM: ежедневные смерти",
                legend    = :topright,
                size      = (1200, 500),
                grid      = true,
                gridalpha = 0.3)

plot!(p_deaths, dates, deaths_prior[1:n_plot];
      label     = "Модель (prior)",
      color     = :orange,
      linewidth = 1.5,
      linestyle = :dash)

plot!(p_deaths, dates, deaths_fit[1:n_plot];
      label     = "Модель (fit)",
      color     = :darkred,
      linewidth = 2)

# --- График 3: все SEIRD-компартменты модели (fit) ---
S_fit = sol_fit[1, 1:n_plot]
E_fit = sol_fit[2, 1:n_plot]
I_fit = sol_fit[3, 1:n_plot]
R_fit = sol_fit[4, 1:n_plot]
D_fit = sol_fit[5, 1:n_plot]

p_compartments = plot(dates, [S_fit E_fit I_fit R_fit D_fit];
                      label     = ["S" "E" "I" "R" "D"],
                      linewidth = 2,
                      xlabel    = "Дата",
                      ylabel    = "Численность",
                      title     = "SEIRD — компартменты, волна $WAVE_NUM",
                      legend    = :right,
                      size      = (1200, 500),
                      grid      = true,
                      gridalpha = 0.3)

# --- Панель 2×2 ---
panel = plot(p_cases, p_deaths, p_compartments;
             layout = (3, 1),
             size   = (1200, 1500))

display(panel)
savefig(panel, "./figures/covid/seird_wave$(WAVE_NUM)_fit.png")
savefig(panel, "./figures/covid/seird_wave$(WAVE_NUM)_fit.svg")
println("\nГрафики сохранены: figures/covid/seird_wave$(WAVE_NUM)_fit.(png|svg)")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 11: Сохранение результатов
# ─────────────────────────────────────────────────────────────

# Датафрейм с траекториями модели
df_model = DataFrame(
    date          = df_wave.date[1:n_plot],
    obs_cases     = obs_cases[1:n_plot],
    obs_deaths    = obs_deaths[1:n_plot],
    fit_cases     = cases_fit[1:n_plot],
    fit_deaths    = deaths_fit[1:n_plot],
    prior_cases   = cases_prior[1:n_plot],
    S             = S_fit,
    E             = E_fit,
    I             = I_fit,
    R             = R_fit,
    D             = D_fit,
)

# Датафрейм с подобранными параметрами
df_params = DataFrame(
    param       = ["β",    "σ",    "γ",    "μ",    "E0",   "I0",  "R0"],
    prior_value = [β_prior, σ_prior, γ_prior, μ_prior, E0_prior, I0_prior, β_prior/γ_prior],
    fit_value   = [β_fit,   σ_fit,   γ_fit,   μ_fit,   E0_fit,   I0_fit,  β_fit/γ_fit],
    delta_pct   = 100 .* ([β_fit, σ_fit, γ_fit, μ_fit, E0_fit, I0_fit, β_fit/γ_fit] .-
                           [β_prior, σ_prior, γ_prior, μ_prior, E0_prior, I0_prior, β_prior/γ_prior]) ./
                           [β_prior, σ_prior, γ_prior, μ_prior, E0_prior, I0_prior, β_prior/γ_prior]
)

@save "./data_out/seird_wave$(WAVE_NUM)_results.jld2" df_model df_params
CSV.write("./data_out/seird_wave$(WAVE_NUM)_model_trajectories.csv", df_model)
CSV.write("./data_out/seird_wave$(WAVE_NUM)_fitted_params.csv", df_params)

println("\n=== Подобранные параметры ===")
println(df_params)
println("\nДанные сохранены в ./data_out/seird_wave$(WAVE_NUM)_*.csv")
println("Готово ✓")
