# ============================================================
# main6.jl — SEIRD-модель для одной волны COVID-19
#
# КЛЮЧЕВАЯ ПРОБЛЕМА, которую исправляет эта версия:
# ─────────────────────────────────────────────────────────────
# С фиксированным N=200 000 и суммарными случаями ~несколько сотен
# истощение восприимчивых S практически отсутствует → SEIRD даёт
# только экспоненциальный рост, пика нет → оптимизатор не может подогнать.
#
# Решение: N_eff — эффективный размер восприимчивой популяции —
# подбирается вместе с остальными параметрами.
# Физический смысл: не все 200 000 жителей участвуют в конкретной волне
# (разный иммунный статус, изоляция, возрастные группы).
#
# Аналитическое предобусловливание (перед оптимизацией):
#   1. N_eff оценивается из уравнения финального размера эпидемии
#   2. β оценивается из скорости роста в начале волны
#   3. E₀, I₀ — из первых значений ряда
#   4. 1D-сканирование N_eff для попадания в наблюдаемый максимум
#
# Свободные параметры θ = [log β, log μ, log E₀, log I₀, log N_eff]
# ============================================================

using JLD2, CSV, DataFrames, Dates, Printf, Random
using DifferentialEquations
using Optim
using Plots, Statistics

# ─────────────────────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────

WAVE_NUM  = 10

const N_CITY    = 200_000.0   # полное население города (верхняя граница для N_eff)
const N_STARTS  = 40
const SEED      = 42

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 1: Загрузка данных
# ─────────────────────────────────────────────────────────────

df_all  = CSV.read("./data_out/covid_data_enriched_with_strains.csv", DataFrame)
df_wave = sort(filter(:wave => ==(WAVE_NUM), df_all), :date)
nrow(df_wave) > 0 || error("Волна $WAVE_NUM не найдена")

#=====#
obs_cases  = Float64.(coalesce.(df_wave.daily_interp_smooth,  0.0))
obs_deaths = Float64.(coalesce.(df_wave.deaths_daily,  0.0))
T          = nrow(df_wave)

peak_cases  = maximum(obs_cases)
peak_idx    = argmax(obs_cases)
total_cases = sum(obs_cases)

println("=" ^ 60)
println("  Волна $WAVE_NUM | $T дней | N_city = $(Int(N_CITY))")
println("  $(minimum(df_wave.date)) .. $(maximum(df_wave.date))")
@printf "  cases  : max=%.2f (день %d)  sum=%.0f\n" peak_cases peak_idx total_cases
@printf "  deaths : max=%.2f  sum=%.0f\n" maximum(obs_deaths) sum(obs_deaths)
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 2: Фиксированные параметры штамма (σ, γ)
# ─────────────────────────────────────────────────────────────

σ_fix   = mean(skipmissing(df_wave.sigma_avg_per_day))
γ_fix   = mean(skipmissing(df_wave.gamma_avg_per_day))
μ_prior = mean(skipmissing(df_wave.mu_avg_per_day))
β_prior = mean(skipmissing(df_wave.beta_avg_per_day))

println("\n=== Параметры штамма ===")
@printf "  σ=%.5f (T_incub=%.1fд)  γ=%.5f (T_infect=%.1fд)  — ФИКСИРОВАНО\n" σ_fix 1/σ_fix γ_fix 1/γ_fix
@printf "  β_prior=%.5f  μ_prior=%.6f  R0_prior=%.3f\n" β_prior μ_prior β_prior/γ_fix

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 3: SEIRD ODE — N_eff передаётся как параметр
#
# Теперь сила инфекции λ = β·I/N_eff, где N_eff — параметр.
# Это позволяет модели адаптировать масштаб восприимчивой популяции.
# ─────────────────────────────────────────────────────────────

function seird!(du, u, p, t)
    S, E, I, R, D  = u
    β, μ, N_eff    = p

    λ = β * I / N_eff   # force of infection
    du[1] = -λ * S
    du[2] =  λ * S   - σ_fix * E
    du[3] =  σ_fix * E - (γ_fix + μ) * I
    du[4] =  γ_fix * I
    du[5] =  μ * I
end

function run_seird(β, μ, E0, I0, N_eff)
    N_eff = max(N_eff, E0 + I0 + 1.0)
    S0    = N_eff - E0 - I0
    u0    = [S0, E0, I0, 0.0, 0.0]
    prob  = ODEProblem(seird!, u0, (0.0, Float64(T-1)), (β, μ, N_eff))
    solve(prob, AutoVern7(Rodas5P());
          saveat=1.0, abstol=1e-8, reltol=1e-6, maxiters=1_000_000)
end

observables(sol, μ) = σ_fix .* sol[2, :],
                      μ     .* sol[3, :]

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 4: Аналитическое предобусловливание
#
# Шаг 4.1: β из скорости роста начальной фазы
# Шаг 4.2: N_eff из уравнения финального размера эпидемии
# Шаг 4.3: 1D-сканирование N_eff для попадания в пик
# ─────────────────────────────────────────────────────────────

# --- 4.1: β из скорости роста ---
function estimate_β_from_growth(y; frac=0.25)
    peak = argmax(y)
    n    = max(7, round(Int, peak * frac))
    n    = min(n, peak - 1)
    n < 3 && return nothing
    seg  = y[1:n]
    ok   = seg .> 0.5
    sum(ok) < 3 && return nothing
    xs = Float64.(findall(ok))
    ys = log.(seg[ok])
    x̄, ȳ = mean(xs), mean(ys)
    r = sum((xs.-x̄).*(ys.-ȳ)) / (sum((xs.-x̄).^2) + 1e-12)
    r > 0 || return nothing
    return r, (r + σ_fix) * (r + γ_fix + μ_prior) / σ_fix
end

res_β = estimate_β_from_growth(obs_cases)
if res_β !== nothing
    r_est, β_est = res_β
    @printf "\n  r=%.4f/д → β_est=%.5f (R0=%.3f)\n" r_est β_est β_est/γ_fix
else
    β_est = β_prior
    @printf "\n  Рост не определён, β_est=β_prior=%.5f\n" β_est
end

# --- 4.2: N_eff из уравнения финального размера ---
# Уравнение: p = 1 - exp(-R0·p),  R0 = β/γ
# где p — доля переболевших. N_eff = total_cases / p
function final_size_attack_rate(R0; tol=1e-8)
    R0 <= 1.0 && return 0.01   # ниже порога эпидемии
    p = 0.5
    for _ in 1:1000
        p_new = 1.0 - exp(-R0 * p)
        abs(p_new - p) < tol && return p_new
        p = p_new
    end
    return p
end

R0_est   = β_est / γ_fix
p_attack = final_size_attack_rate(R0_est)
N_eff_fs = total_cases / p_attack   # оценка из финального размера
# Ограничиваем разумным диапазоном
N_eff_fs = clamp(N_eff_fs, total_cases * 1.5, N_CITY)

@printf "  R0_est=%.3f → attack_rate=%.1f%% → N_eff_fs=%.0f\n" R0_est 100*p_attack N_eff_fs

# --- 4.3: 1D-сканирование N_eff для попадания в максимум ---
# При заданном β, σ, γ, μ пик σ·E достигается при S = N_eff·(γ_fix+μ_prior)/β
# → пик тем выше, чем меньше N_eff
# Сканируем N_eff от N_eff_fs/5 до N_eff_fs*5, ищем минимум |peak_model - peak_obs|

println("\n  1D-сканирование N_eff для подгонки пика...")

I0_scan  = max(1.0, obs_cases[1] / (γ_fix + μ_prior + 1e-12))
E0_scan  = max(1.0, I0_scan * (γ_fix + μ_prior) / σ_fix)

scan_Ns = exp.(range(log(max(total_cases*1.1, 10.0)),
                     log(N_CITY), length=200))

best_scan = let
    best = (err=Inf, N_eff=N_eff_fs)
    for N_eff_try in scan_Ns
        sol = run_seird(β_est, μ_prior, E0_scan, I0_scan, N_eff_try)
        sol.retcode != ReturnCode.Success && continue
        mc, _ = observables(sol, μ_prior)
        peak_model = isempty(mc) ? 0.0 : maximum(mc)
        err = abs(log(peak_model + 1) - log(peak_cases + 1))
        if err < best.err
            best = (err=err, N_eff=N_eff_try)
        end
    end
    best
end

N_eff_init = best_scan.N_eff
@printf "  N_eff после сканирования: %.0f  (пиковая ошибка log: %.4f)\n" N_eff_init best_scan.err

# Стартовые IC
I0_init = max(1.0, obs_cases[1] / (γ_fix + μ_prior + 1e-12))
E0_init = max(1.0, I0_init * (γ_fix + μ_prior) / σ_fix)

println("\n=== Стартовая точка ===")
@printf "  β=%.5f  μ=%.6f  E₀=%.2f  I₀=%.2f  N_eff=%.0f\n" β_est μ_prior E0_init I0_init N_eff_init

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 5: Функция потерь — log-MSE + мягкий штраф на N_eff
# ─────────────────────────────────────────────────────────────

const log_obs_c  = log1p.(obs_cases)
const log_obs_d  = log1p.(obs_deaths)
const DEATHS_W   = sum(obs_deaths) > 5.0 ? 3.0 : 0.0

function loss(θ::Vector{Float64})
    β, μ, E0, I0, N_eff = exp.(θ)

    any(x -> x < 1e-12, (β, μ, E0, I0))  && return 1e18
    N_eff < E0 + I0 + 1.0                 && return 1e18
    N_eff > N_CITY                         && return 1e18

    sol = run_seird(β, μ, E0, I0, N_eff)
    sol.retcode !== ReturnCode.Success     && return 1e18

    mc, md = observables(sol, μ)
    n = min(length(mc), T)

    lc = mean((log1p.(max.(mc[1:n], 0.0)) .- log_obs_c[1:n]).^2)
    ld = mean((log1p.(max.(md[1:n], 0.0)) .- log_obs_d[1:n]).^2)

    # Мягкий штраф: N_eff не должен уходить ниже суммарных случаев
    penalty = N_eff < total_cases * 1.1 ? 10.0 * (log(total_cases * 1.1) - θ[5])^2 : 0.0

    return lc + DEATHS_W * ld + penalty
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 6: Prior run с аналитическими начальными условиями
# ─────────────────────────────────────────────────────────────

θ0 = log.([β_est, μ_prior, E0_init, I0_init, N_eff_init])
sol_prior = run_seird(β_est, μ_prior, E0_init, I0_init, N_eff_init)
cases_prior, deaths_prior = sol_prior.retcode == ReturnCode.Success ?
    observables(sol_prior, μ_prior) : (zeros(T), zeros(T))

println("\n=== Prior run ===")
@printf "  retcode: %s\n" sol_prior.retcode
@printf "  loss(θ₀) = %.6f\n" loss(θ0)
if sol_prior.retcode == ReturnCode.Success
    @printf "  peak_prior = %.2f  (obs: %.2f)\n" maximum(cases_prior) peak_cases
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 7: Мульти-старт оптимизация
# ─────────────────────────────────────────────────────────────

println("\n=== Мульти-старт ($N_STARTS стартов) ===")
rng = MersenneTwister(SEED)

# Возмущения: β и N_eff — умеренные (у нас хорошие оценки),
# μ, E₀, I₀ — шире
perturb_σ = [0.4, 1.0, 1.2, 1.2, 0.6]  # [β, μ, E₀, I₀, N_eff]

nm_results = map(1:N_STARTS) do _
    θ_start = θ0 .+ perturb_σ .* randn(rng, 5)
    res = optimize(loss, θ_start, NelderMead(),
                   Optim.Options(iterations=5_000, show_trace=false, g_tol=1e-7))
    (loss=Optim.minimum(res), θ=Optim.minimizer(res))
end

sort!(nm_results, by=x->x.loss)
@printf "  Топ-5: %.4f  %.4f  %.4f  %.4f  %.4f\n" [nm_results[i].loss for i in 1:5]...

# LBFGS от топ-3
lbfgs_results = map(nm_results[1:3]) do nm
    r = optimize(loss, nm.θ, LBFGS(),
                 Optim.Options(iterations=10_000, show_trace=false, g_tol=1e-14))
    (loss=Optim.minimum(r), θ=Optim.minimizer(r))
end
sort!(lbfgs_results, by=x->x.loss)
best = lbfgs_results[1]

β_f, μ_f, E0_f, I0_f, N_eff_f = exp.(best.θ)

println("\n=== Подобранные параметры ===")
@printf "  β     : %.5f → %.5f  (R0: %.3f → %.3f)\n" β_est β_f β_est/γ_fix β_f/γ_fix
@printf "  σ     : %.5f  (фиксировано)\n" σ_fix
@printf "  γ     : %.5f  (фиксировано)\n" γ_fix
@printf "  μ     : %.6f → %.6f\n" μ_prior μ_f
@printf "  E₀    : %.2f → %.2f\n" E0_init E0_f
@printf "  I₀    : %.2f → %.2f\n" I0_init I0_f
@printf "  N_eff : %.0f → %.0f  (%.1f%% от N_city)\n" N_eff_init N_eff_f 100*N_eff_f/N_CITY
@printf "  Финальная потеря: %.6f\n" best.loss

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 8: Финальное решение + графики
# ─────────────────────────────────────────────────────────────

sol_fit = run_seird(β_f, μ_f, E0_f, I0_f, N_eff_f)
cases_fit, deaths_fit = observables(sol_fit, μ_f)
n_plt = min(length(cases_fit), T)
dates = df_wave.date[1:n_plt]

@printf "\n  peak_fit=%.2f  peak_obs=%.2f  ratio=%.3f\n" maximum(cases_fit) peak_cases maximum(cases_fit)/peak_cases

# А) Случаи линейная
pA = plot(dates, obs_cases[1:n_plt];
          label="данные", color=:steelblue, lw=2, alpha=0.8,
          xlabel="Дата", ylabel="Случаев / день",
          title="SEIRD волна $WAVE_NUM  [loss=$(round(best.loss,sigdigits=3))  N_eff=$(round(Int,N_eff_f))]",
          legend=:topright, size=(1100,380), grid=true, gridalpha=0.3)
plot!(pA, dates, cases_prior[1:n_plt]; label="prior (аналит.)", color=:orange, lw=1.5, ls=:dash)
plot!(pA, dates, cases_fit[1:n_plt];   label="fit",  color=:crimson, lw=2.5)

# Б) log-шкала
pB = plot(dates, log1p.(obs_cases[1:n_plt]);
          label="данные", color=:steelblue, lw=2, alpha=0.8,
          xlabel="Дата", ylabel="log(случаев+1)",
          title="log-шкала (метрика оптимизации)",
          legend=:topright, size=(1100,380), grid=true, gridalpha=0.3)
plot!(pB, dates, log1p.(max.(cases_prior[1:n_plt],0.0)); label="prior", color=:orange, lw=1.5, ls=:dash)
plot!(pB, dates, log1p.(max.(cases_fit[1:n_plt], 0.0));  label="fit",   color=:crimson, lw=2.5)

# В) Компартменты
S_v = sol_fit[1,1:n_plt]; E_v = sol_fit[2,1:n_plt]
I_v = sol_fit[3,1:n_plt]; R_v = sol_fit[4,1:n_plt]; D_v = sol_fit[5,1:n_plt]
pC = plot(dates, [S_v E_v I_v R_v D_v];
          label=["S" "E" "I" "R" "D"], lw=2,
          xlabel="Дата", ylabel="Человек",
          title="Компартменты  (N_eff=$(round(Int,N_eff_f)) из $(Int(N_CITY)))",
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
    param       = ["β",    "σ",    "γ",    "μ",   "E0",   "I0",   "N_eff", "R0"],
    fixed       = [false,  true,   true,   false, false,  false,  false,   false],
    prior_value = [β_est,  σ_fix,  γ_fix,  μ_prior, E0_init, I0_init, N_eff_init, β_est/γ_fix],
    fit_value   = [β_f,    σ_fix,  γ_fix,  μ_f,   E0_f,   I0_f,   N_eff_f,  β_f/γ_fix],
)
df_par.delta_pct = ifelse.(df_par.fixed, 0.0,
    100 .* (df_par.fit_value .- df_par.prior_value) ./ (df_par.prior_value .+ 1e-12))

@save "./data_out/seird_wave$(WAVE_NUM)_results.jld2" df_traj df_par
CSV.write("./data_out/seird_wave$(WAVE_NUM)_trajectories.csv", df_traj)
CSV.write("./data_out/seird_wave$(WAVE_NUM)_params.csv", df_par)

println("\n=== Итог ===")
println(df_par)
println("Готово ✓")
