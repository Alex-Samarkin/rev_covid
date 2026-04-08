using JLD2, CSV, DataFrames, Dates, Plots, Peaks
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
#             Город ~200 000 жителей
#
# Стратегия подбора параметров:
#   σ, γ  — фиксируются из календаря штаммов (биологические константы)
#   β     — оценивается из скорости роста данных, затем уточняется
#   μ     — подбирается по смертям (или фиксируется из IFR если смертей мало)
#   E₀, I₀ — подбираются вместе с β
#
# Итого 4 свободных параметра: [β, μ, E₀, I₀]
# Это резко улучшает обусловленность задачи.
# ============================================================

using JLD2, CSV, DataFrames, Dates, Printf, Random
using DifferentialEquations
using Optim
using Plots, Statistics

# ─────────────────────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────

# const WAVE_NUM      = 4
const N_POP         = 200_000.0
const N_STARTS      = 40
const SEED          = 42

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 1: Загрузка данных
# ─────────────────────────────────────────────────────────────

df_all  = CSV.read("./data_out/covid_data_enriched_with_strains.csv", DataFrame)
df_wave = sort(filter(:wave => ==(WAVE_NUM), df_all), :date)
nrow(df_wave) > 0 || error("Волна $WAVE_NUM не найдена")

obs_cases  = Float64.(coalesce.(df_wave.daily_interp_smooth,  0.0))
obs_deaths = Float64.(coalesce.(df_wave.deaths_daily,  0.0))
T          = nrow(df_wave)

println("=" ^ 60)
println("  Волна $WAVE_NUM | $T дней | N = $(Int(N_POP))")
println("  $(minimum(df_wave.date)) .. $(maximum(df_wave.date))")
@printf "  cases  | max=%.2f  sum=%.0f\n" maximum(obs_cases)  sum(obs_cases)
@printf "  deaths | max=%.2f  sum=%.0f\n" maximum(obs_deaths) sum(obs_deaths)
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 2: Параметры штамма
# ─────────────────────────────────────────────────────────────

σ_fix = mean(skipmissing(df_wave.sigma_avg_per_day))   # ФИКСИРОВАНО
γ_fix = mean(skipmissing(df_wave.gamma_avg_per_day))   # ФИКСИРОВАНО
μ_prior = mean(skipmissing(df_wave.mu_avg_per_day))
β_prior = mean(skipmissing(df_wave.beta_avg_per_day))

println("\n=== Фиксированные параметры штамма ===")
@printf "  σ = %.5f  (T_incub  = %.1f д.) — ФИКСИРОВАНО\n" σ_fix 1/σ_fix
@printf "  γ = %.5f  (T_infect = %.1f д.) — ФИКСИРОВАНО\n" γ_fix 1/γ_fix

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 3: Умная оценка β из данных
#
# В начале волны (фаза экспоненциального роста) кривая cases(t) ~ exp(r·t)
# Для SEIR-системы скорость роста r связана с β уравнением:
#   (r + σ)(r + γ + μ) = β · σ
# Откуда: β = (r + σ)(r + γ + μ) / σ
#
# r оцениваем линейной регрессией log(cases) на первых ~20% волны
# (до пика, пока рост ещё экспоненциальный)
# ─────────────────────────────────────────────────────────────

function estimate_growth_rate(y::Vector{Float64}; frac::Float64=0.25)
    # Ищем пик
    peak_idx = argmax(y)
    # Берём первую четверть до пика (или минимум 7 точек)
    n_fit = max(7, round(Int, peak_idx * frac))
    n_fit = min(n_fit, peak_idx - 1)
    n_fit < 3 && return nothing

    # Только положительные точки для log-регрессии
    seg = y[1:n_fit]
    valid = seg .> 0.5
    sum(valid) < 3 && return nothing

    xs = Float64.(findall(valid))
    ys = log.(seg[valid])

    # МНК: y = a + r·x
    x̄, ȳ = mean(xs), mean(ys)
    r = sum((xs .- x̄) .* (ys .- ȳ)) / (sum((xs .- x̄).^2) + 1e-12)
    return r
end

r_est = estimate_growth_rate(obs_cases)

if r_est !== nothing && r_est > 0
    # β из дисперсионного соотношения SEIR
    β_data = (r_est + σ_fix) * (r_est + γ_fix + μ_prior) / σ_fix
    @printf "\n  Оценка из данных: r = %.4f/д → β_data = %.5f  (R0 = %.3f)\n" r_est β_data β_data/γ_fix
else
    β_data = β_prior
    @printf "\n  Рост не определён из данных, используем β_prior = %.5f\n" β_prior
end

# Стартовые начальные условия
peak_val    = maximum(obs_cases)
I0_est      = max(1.0, obs_cases[1] / (γ_fix + μ_prior + 1e-12))
E0_est      = max(1.0, I0_est * (1 + σ_fix / (γ_fix + μ_prior)))

println("  Стартовые IC: E₀=$(round(E0_est,digits=2))  I₀==$(round(I0_est,digits=2))")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 4: SEIRD ODE (σ, γ — константы модуля)
# ─────────────────────────────────────────────────────────────

function seird!(du, u, p, t)
    S, E, I, R, D = u
    β, μ          = p          # σ и γ фиксированы

    λ = β * I / N_POP
    du[1] = -λ * S
    du[2] =  λ * S  - σ_fix * E
    du[3] =  σ_fix * E  - (γ_fix + μ) * I
    du[4] =  γ_fix * I
    du[5] =  μ * I
end

function run_seird(β, μ, E0, I0)
    S0   = max(0.0, N_POP - E0 - I0)
    u0   = [S0, E0, I0, 0.0, 0.0]
    prob = ODEProblem(seird!, u0, (0.0, Float64(T-1)), (β, μ))
    # AutoVern7(Rodas5P): быстрый RK на нежёстких участках,
    # автопереключение на неявный Rodas5P при жёсткости
    solve(prob, AutoVern7(Rodas5P());
          saveat=1.0, abstol=1e-8, reltol=1e-6, maxiters=1_000_000)
end

observables(sol, μ) = σ_fix .* sol[2, :],   # σ·E → новые случаи
                      μ     .* sol[3, :]     # μ·I → смерти

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 5: Функция потерь — log-MSE
#
# θ = [log β, log μ, log E₀, log I₀]  ∈ ℝ⁴
# ─────────────────────────────────────────────────────────────

const log_obs_c = log1p.(obs_cases)
const log_obs_d = log1p.(obs_deaths)
const DEATHS_W  = sum(obs_deaths) > 5.0 ? 3.0 : 0.0

function loss(θ::Vector{Float64})
    β, μ, E0, I0 = exp.(θ)

    (β < 1e-9 || μ < 1e-9)  && return 1e18
    (E0 + I0) >= N_POP       && return 1e18

    sol = run_seird(β, μ, E0, I0)
    sol.retcode !== ReturnCode.Success && return 1e18

    mc, md = observables(sol, μ)
    n = min(length(mc), T)

    lc = mean((log1p.(max.(mc[1:n], 0.0)) .- log_obs_c[1:n]).^2)
    ld = mean((log1p.(max.(md[1:n], 0.0)) .- log_obs_d[1:n]).^2)
    return lc + DEATHS_W * ld
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 6: Prior run
# ─────────────────────────────────────────────────────────────

sol_prior    = run_seird(β_prior, μ_prior, E0_est, I0_est)
cases_prior, deaths_prior = sol_prior.retcode == ReturnCode.Success ?
    observables(sol_prior, μ_prior) :
    (zeros(T), zeros(T))

θ0 = log.([β_data, μ_prior, E0_est, I0_est])
println("\n=== Prior run ===")
@printf "  retcode: %s   loss(start) = %.6f\n" sol_prior.retcode loss(θ0)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 7: Мульти-старт оптимизация
#
# Возмущаем стартовую точку в log-пространстве.
# β-возмущение небольшое (±0.5) — у нас хорошая оценка из данных.
# IC-возмущение шире (±1.5) — они менее определены.
# ─────────────────────────────────────────────────────────────

println("\n=== Мульти-старт ($N_STARTS стартов) ===")
rng = MersenneTwister(SEED)

# Матрица возмущений: [log β, log μ, log E₀, log I₀]
perturb_σ = [0.5, 1.0, 1.5, 1.5]

nm_results = map(1:N_STARTS) do i
    δ = perturb_σ .* randn(rng, 4)
    θ_start = θ0 .+ δ
    res = optimize(loss, θ_start, NelderMead(),
                   Optim.Options(iterations=4_000, show_trace=false, g_tol=1e-7))
    (loss=Optim.minimum(res), θ=Optim.minimizer(res))
end

sort!(nm_results, by=x->x.loss)
@printf "  Топ-3 NelderMead: %.6f  %.6f  %.6f\n" nm_results[1].loss nm_results[2].loss nm_results[3].loss

# LBFGS от топ-3 стартов, берём лучший
lbfgs_results = map(nm_results[1:3]) do nm
    r = optimize(loss, nm.θ, LBFGS(),
                 Optim.Options(iterations=10_000, show_trace=false, g_tol=1e-14))
    (loss=Optim.minimum(r), θ=Optim.minimizer(r))
end
sort!(lbfgs_results, by=x->x.loss)
best = lbfgs_results[1]

β_f, μ_f, E0_f, I0_f = exp.(best.θ)

println("\n=== Подобранные параметры ===")
@printf "  β   start=%.5f  fit=%.5f  Δ=%+.1f%%\n" β_data β_f 100*(β_f-β_data)/β_data
@printf "  σ   = %.5f  (фиксировано)\n" σ_fix
@printf "  γ   = %.5f  (фиксировано)\n" γ_fix
@printf "  μ   start=%.6f  fit=%.6f  Δ=%+.1f%%\n" μ_prior μ_f 100*(μ_f-μ_prior)/μ_prior
@printf "  R0  start=%.3f   fit=%.3f\n" β_data/γ_fix β_f/γ_fix
@printf "  E₀=%.1f  I₀=%.1f\n" E0_f I0_f
@printf "  Финальная потеря: %.6f\n" best.loss

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 8: Финальное решение + графики
# ─────────────────────────────────────────────────────────────

sol_fit           = run_seird(β_f, μ_f, E0_f, I0_f)
cases_fit, deaths_fit = observables(sol_fit, μ_f)
n_plt = min(length(cases_fit), T)
dates = df_wave.date[1:n_plt]

# А) Случаи линейная
pA = plot(dates, obs_cases[1:n_plt];
          label="данные", color=:steelblue, lw=2, alpha=0.8,
          xlabel="Дата", ylabel="Случаев / день",
          title="SEIRD волна $WAVE_NUM — случаи",
          legend=:topright, size=(1100,380), grid=true, gridalpha=0.3)
# plot!(pA, dates, cases_prior[1:n_plt]; label="prior", color=:orange, lw=1.5, ls=:dash)
plot!(pA, dates, cases_fit[1:n_plt];   label="fit  (loss=$(round(best.loss,sigdigits=3)))",
      color=:crimson, lw=2.5)

# Б) Случаи log — так видит оптимизатор
pB = plot(dates, log1p.(obs_cases[1:n_plt]);
          label="данные", color=:steelblue, lw=2, alpha=0.8,
          xlabel="Дата", ylabel="log(случаев + 1)",
          title="SEIRD волна $WAVE_NUM — log-шкала",
          legend=:topright, size=(1100,380), grid=true, gridalpha=0.3)
plot!(pB, dates, log1p.(max.(cases_prior[1:n_plt],0.0)); label="prior", color=:orange, lw=1.5, ls=:dash)
plot!(pB, dates, log1p.(max.(cases_fit[1:n_plt], 0.0)); label="fit",   color=:crimson, lw=2.5)

# В) Компартменты
S_v = sol_fit[1,1:n_plt]; E_v = sol_fit[2,1:n_plt]
I_v = sol_fit[3,1:n_plt]; R_v = sol_fit[4,1:n_plt]; D_v = sol_fit[5,1:n_plt]

pC = plot(dates, [S_v E_v I_v R_v D_v];
          label=["S" "E" "I" "R" "D"], lw=2,
          xlabel="Дата", ylabel="Человек",
          title="Компартменты — волна $WAVE_NUM  (N=$(Int(N_POP)))",
          legend=:right, size=(1100,380), grid=true, gridalpha=0.3)

panel = plot(pA, pB, pC; layout=(3,1), size=(1100,1200))
display(panel)
savefig(panel, "./figures/covid/seird_wave$(WAVE_NUM)_fit.png")
savefig(panel, "./figures/covid/seird_wave$(WAVE_NUM)_fit.svg")
println("\nГрафики → figures/covid/seird_wave$(WAVE_NUM)_fit.(png|svg)")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 9: Сохранение
# ─────────────────────────────────────────────────────────────

df_traj = DataFrame(
    date=df_wave.date[1:n_plt],
    obs_cases=obs_cases[1:n_plt], obs_deaths=obs_deaths[1:n_plt],
    fit_cases=cases_fit[1:n_plt], fit_deaths=deaths_fit[1:n_plt],
    prior_cases=cases_prior[1:n_plt],
    S=S_v, E=E_v, I=I_v, R=R_v, D=D_v,
)
df_par = DataFrame(
    param       = ["β",    "σ",    "γ",    "μ",   "E0",   "I0",   "R0"],
    fixed       = [false,  true,   true,   false, false,  false,  false],
    prior_value = [β_data, σ_fix,  γ_fix,  μ_prior, E0_est, I0_est, β_data/γ_fix],
    fit_value   = [β_f,    σ_fix,  γ_fix,  μ_f,   E0_f,   I0_f,   β_f/γ_fix],
)
df_par.delta_pct = ifelse.(df_par.fixed, 0.0,
    100 .* (df_par.fit_value .- df_par.prior_value) ./ (df_par.prior_value .+ 1e-12))

@save "./data_out/seird_wave$(WAVE_NUM)_results.jld2" df_traj df_par
CSV.write("./data_out/seird_wave$(WAVE_NUM)_trajectories.csv", df_traj)
CSV.write("./data_out/seird_wave$(WAVE_NUM)_params.csv", df_par)

println("\n=== Итог ===")
println(df_par)
println("Готово ✓")