# ============================================================
# main6.jl — SEIRD-модель для одной волны COVID-19
#
# СТРАТЕГИЯ:
#   1. Аналитический N_eff из площади под кривой до пика
#   2. Аналитический I₀ из обратной экстраполяции роста
#   3. 2D сеточный поиск (β × N_eff) — найти область минимума
#   4. NelderMead + LBFGS от лучшей точки сетки
# ============================================================

using JLD2, CSV, DataFrames, Dates, Printf, Random
using DifferentialEquations
using Optim
using Plots, Statistics

const N_CITY   = 200_000.0

# ─────────────────────────────────────────────────────────────
# 1. Данные
# ─────────────────────────────────────────────────────────────

df_all  = CSV.read("./data_out/covid_data_enriched_with_strains.csv", DataFrame)

wave_numbers = sort(unique(df_all.wave))
println("Доступные волны: ", wave_numbers)

for WAVE_NUM in wave_numbers
    df_wave = sort(filter(:wave => ==(WAVE_NUM), df_all), :date)
    nrow(df_wave) > 0 || continue

    println("Волна $WAVE_NUM: $(nrow(df_wave)) дней  (с $(minimum(df_wave.date)) по $(maximum(df_wave.date)))")


df_wave = sort(filter(:wave => ==(WAVE_NUM), df_all), :date)
nrow(df_wave) > 0 || error("Волна не найдена")

obs_cases  = Float64.(coalesce.(df_wave.daily_interp_smooth,  0.0))
obs_deaths = Float64.(coalesce.(df_wave.deaths_daily,  0.0))
T          = nrow(df_wave)

peak_cases  = maximum(obs_cases)
peak_idx    = argmax(obs_cases)
total_cases = sum(obs_cases)
cases_to_peak = sum(obs_cases[1:peak_idx])

println("=" ^ 60)
@printf "  Волна %d | %d дней\n" WAVE_NUM T
@printf "  peak=%.2f (день %d)  total=%.0f  to_peak=%.0f\n" peak_cases peak_idx total_cases cases_to_peak
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# 2. Параметры штамма
# ─────────────────────────────────────────────────────────────

σ_fix   = mean(skipmissing(df_wave.sigma_avg_per_day))
γ_fix   = mean(skipmissing(df_wave.gamma_avg_per_day))
μ_prior = mean(skipmissing(df_wave.mu_avg_per_day))
β_prior = mean(skipmissing(df_wave.beta_avg_per_day))

@printf "\n  σ=%.4f  γ=%.4f  μ=%.5f  β_prior=%.4f  R0_prior=%.2f\n" σ_fix γ_fix μ_prior β_prior β_prior/γ_fix

# ─────────────────────────────────────────────────────────────
# 3. ODE
# ─────────────────────────────────────────────────────────────

function seird!(du, u, p, t)
    S, E, I, R, D = u
    β, μ, N_eff   = p
    λ = β * I / N_eff
    du[1] = -λ * S
    du[2] =  λ * S   - σ_fix * E
    du[3] =  σ_fix * E - (γ_fix + μ) * I
    du[4] =  γ_fix * I
    du[5] =  μ * I
end

function run_seird(β, μ, E0, I0, N_eff)
    N_eff = max(N_eff, E0 + I0 + 1.0)
    u0 = [N_eff - E0 - I0, E0, I0, 0.0, 0.0]
    prob = ODEProblem(seird!, u0, (0.0, Float64(T-1)), (β, μ, N_eff))
    solve(prob, AutoVern7(Rodas5P());
          saveat=1.0, abstol=1e-8, reltol=1e-6, maxiters=1_000_000,
          verbose=false)   # глушим предупреждения решателя
end

obs_c(sol, μ) = σ_fix .* sol[2, :]
obs_d(sol, μ) = μ     .* sol[3, :]

# ─────────────────────────────────────────────────────────────
# 4. Аналитическая оценка N_eff и I₀
#
# N_eff из площади до пика:
#   На пике истощено S: ΔS = cases_to_peak
#   По теории SEIR: S_peak = N_eff / R0
#   → N_eff - N_eff/R0 = cases_to_peak
#   → N_eff = cases_to_peak * R0 / (R0 - 1)
#
# I₀ из обратной экстраполяции роста:
#   В начальной фазе I(t) ≈ I_peak * exp(-r*(peak_idx - t))
#   I₀ = I_peak * exp(-r * peak_idx)
# ─────────────────────────────────────────────────────────────

function estimate_growth_rate(y)
    pk = argmax(y)
    n  = max(7, round(Int, pk * 0.3))
    n  = min(n, pk - 1)
    seg = y[1:n]; ok = seg .> 0.5
    sum(ok) < 3 && return 0.05   # fallback
    xs = Float64.(findall(ok)); ys = log.(seg[ok])
    x̄, ȳ = mean(xs), mean(ys)
    r = sum((xs.-x̄).*(ys.-ȳ)) / (sum((xs.-x̄).^2)+1e-12)
    return max(r, 1e-4)
end

r_est  = estimate_growth_rate(obs_cases)
β_est  = (r_est + σ_fix) * (r_est + γ_fix + μ_prior) / σ_fix
R0_est = β_est / γ_fix

# N_eff из площади до пика (аналитически)
N_eff_analytic = R0_est > 1.0 ?
    cases_to_peak * R0_est / (R0_est - 1.0) :
    total_cases * 5.0
N_eff_analytic = clamp(N_eff_analytic, total_cases * 2.0, N_CITY)

# I₀ из обратной экстраполяции
I_peak = peak_cases / (γ_fix + μ_prior + 1e-12)
I0_analytic = max(0.5, I_peak * exp(-r_est * peak_idx))
E0_analytic = max(0.5, I0_analytic * (γ_fix + μ_prior) / σ_fix)

@printf "\n  r=%.4f  β_est=%.4f  R0_est=%.2f\n" r_est β_est R0_est
@printf "  N_eff_analytic=%.0f  I₀=%.2f  E₀=%.2f\n" N_eff_analytic I0_analytic E0_analytic

# Проверка аналитической точки
sol_check = run_seird(β_est, μ_prior, E0_analytic, I0_analytic, N_eff_analytic)
if sol_check.retcode == ReturnCode.Success
    mc = obs_c(sol_check, μ_prior)
    @printf "  peak_analytic=%.2f  peak_obs=%.2f\n" maximum(mc) peak_cases
end

# ─────────────────────────────────────────────────────────────
# 5. Сеточный поиск по (β, N_eff)
#
# Перебираем все комбинации β и N_eff, фиксируя μ=μ_prior,
# I₀ и E₀ — аналитические.
# Это позволяет нарисовать landscape потерь и найти глобальный минимум.
# ─────────────────────────────────────────────────────────────

println("\n  Сеточный поиск (β × N_eff)...")

# β: вокруг β_est с запасом ×3 в каждую сторону
β_grid    = exp.(range(log(β_est/4), log(β_est*4), length=30))
# N_eff: от total_cases*1.5 до N_CITY
N_eff_grid = exp.(range(log(max(total_cases*1.5,10)), log(N_CITY), length=30))

 log_obs_c = log1p.(obs_cases)
 log_obs_d = log1p.(obs_deaths)
 DEATHS_W  = sum(obs_deaths) > 5.0 ? 3.0 : 0.0

function loss_fixed_IC(β, μ, E0, I0, N_eff)
    sol = run_seird(β, μ, E0, I0, N_eff)
    sol.retcode !== ReturnCode.Success && return Inf
    mc = obs_c(sol, μ); md = obs_d(sol, μ)
    n  = min(length(mc), T)
    lc = mean((log1p.(max.(mc[1:n],0.0)) .- log_obs_c[1:n]).^2)
    ld = mean((log1p.(max.(md[1:n],0.0)) .- log_obs_d[1:n]).^2)
    return lc + DEATHS_W * ld
end

# Сетка потерь
grid_losses = fill(Inf, length(β_grid), length(N_eff_grid))
for (i,β) in enumerate(β_grid), (j,N) in enumerate(N_eff_grid)
    # I₀ адаптируем к каждой паре (β, N_eff): обратная экстраполяция
    r_loc = max(1e-4, β - γ_fix - σ_fix/2)   # грубое r
    I0_loc = max(0.5, I_peak * exp(-r_loc * peak_idx))
    E0_loc = max(0.5, I0_loc * (γ_fix + μ_prior) / σ_fix)
    grid_losses[i,j] = loss_fixed_IC(β, μ_prior, E0_loc, I0_loc, N)
end

# Лучшая точка сетки
best_grid_idx  = argmin(grid_losses)
β_grid_best    = β_grid[best_grid_idx[1]]
N_eff_grid_best = N_eff_grid[best_grid_idx[2]]
loss_grid_best = grid_losses[best_grid_idx]

@printf "  Лучшая точка сетки: β=%.5f  N_eff=%.0f  loss=%.5f\n" β_grid_best N_eff_grid_best loss_grid_best

# ─────────────────────────────────────────────────────────────
# 6. Функция потерь для оптимизации
#    θ = [log β, log μ, log E₀, log I₀, log N_eff]
# ─────────────────────────────────────────────────────────────

function loss(θ::Vector{Float64})
    β, μ, E0, I0, N_eff = exp.(θ)
    any(x -> x < 1e-12, (β, μ, E0, I0)) && return 1e18
    N_eff < E0 + I0 + 1.0               && return 1e18
    N_eff > N_CITY                       && return 1e18
    v = loss_fixed_IC(β, μ, E0, I0, N_eff)
    isinf(v) && return 1e18
    return v
end

# ─────────────────────────────────────────────────────────────
# 7. Оптимизация от лучшей точки сетки
# ─────────────────────────────────────────────────────────────

# IC для лучшей точки сетки
r_best = max(1e-4, β_grid_best - γ_fix - σ_fix/2)
I0_grid_best = max(0.5, I_peak * exp(-r_best * peak_idx))
E0_grid_best = max(0.5, I0_grid_best * (γ_fix + μ_prior) / σ_fix)

θ_grid = log.([β_grid_best, μ_prior, E0_grid_best, I0_grid_best, N_eff_grid_best])
@printf "\n  loss(θ_grid) = %.6f\n" loss(θ_grid)

# NelderMead от точки сетки + несколько случайных стартов вокруг
rng = MersenneTwister(42)
candidates = [θ_grid]
for _ in 1:19
    push!(candidates, θ_grid .+ [0.3,0.8,1.0,1.0,0.4] .* randn(rng, 5))
end

nm_results = map(candidates) do θ_s
    res = optimize(loss, θ_s, NelderMead(),
                   Optim.Options(iterations=8_000, show_trace=false, g_tol=1e-8))
    (loss=Optim.minimum(res), θ=Optim.minimizer(res))
end
sort!(nm_results, by=x->x.loss)
@printf "  Топ-3 NelderMead: %.5f  %.5f  %.5f\n" nm_results[1].loss nm_results[2].loss nm_results[3].loss

# LBFGS уточнение
res_lbfgs = optimize(loss, nm_results[1].θ, LBFGS(),
                     Optim.Options(iterations=10_000, show_trace=false, g_tol=1e-14))
θ_fit = Optim.minimizer(res_lbfgs)
β_f, μ_f, E0_f, I0_f, N_eff_f = exp.(θ_fit)

println("\n=== Подобранные параметры ===")
@printf "  β     = %.5f  (R0=%.3f)\n" β_f β_f/γ_fix
@printf "  σ     = %.5f  (фиксировано)\n" σ_fix
@printf "  γ     = %.5f  (фиксировано)\n" γ_fix
@printf "  μ     = %.6f\n" μ_f
@printf "  E₀    = %.2f\n" E0_f
@printf "  I₀    = %.2f\n" I0_f
@printf "  N_eff = %.0f  (%.1f%% от N_city)\n" N_eff_f 100*N_eff_f/N_CITY
@printf "  loss  = %.6f\n" Optim.minimum(res_lbfgs)

# ─────────────────────────────────────────────────────────────
# 8. Графики
# ─────────────────────────────────────────────────────────────

sol_fit = run_seird(β_f, μ_f, E0_f, I0_f, N_eff_f)
cases_fit = obs_c(sol_fit, μ_f)
deaths_fit = obs_d(sol_fit, μ_f)
n_plt = min(length(cases_fit), T)
dates = df_wave.date[1:n_plt]

# Карта потерь сетки (heatmap)
p_heat = heatmap(log10.(N_eff_grid), log10.(β_grid),
                 clamp.(grid_losses, 0, 2);
                 xlabel="log10(N_eff)", ylabel="log10(β)",
                 title="Landscape потерь (β × N_eff)",
                 color=:viridis, size=(500,400))
scatter!(p_heat, [log10(N_eff_grid_best)], [log10(β_grid_best)];
         color=:red, markersize=8, label="grid_best")
scatter!(p_heat, [log10(N_eff_f)], [log10(β_f)];
         color=:yellow, markersize=8, markershape=:star5, label="fit")

# Случаи: линейная шкала
pA = plot(dates, obs_cases[1:n_plt];
          label="данные", color=:steelblue, lw=2, alpha=0.8,
          xlabel="Дата", ylabel="Случаев/день",
          title="SEIRD волна $WAVE_NUM  [loss=$(round(Optim.minimum(res_lbfgs),sigdigits=3))  N_eff=$(round(Int,N_eff_f))]",
          legend=:topright, size=(1100,380), grid=true, gridalpha=0.3)
plot!(pA, dates, cases_fit[1:n_plt]; label="fit", color=:crimson, lw=2.5)

# log-шкала
pB = plot(dates, log1p.(obs_cases[1:n_plt]);
          label="данные", color=:steelblue, lw=2, alpha=0.8,
          xlabel="Дата", ylabel="log(случаев+1)",
          title="log-шкала", legend=:topright, size=(1100,380))
plot!(pB, dates, log1p.(max.(cases_fit[1:n_plt],0.0)); label="fit", color=:crimson, lw=2.5)

# Компартменты
S_v=sol_fit[1,1:n_plt]; E_v=sol_fit[2,1:n_plt]
I_v=sol_fit[3,1:n_plt]; R_v=sol_fit[4,1:n_plt]; D_v=sol_fit[5,1:n_plt]
pC = plot(dates, [S_v E_v I_v R_v D_v];
          label=["S" "E" "I" "R" "D"], lw=2,
          xlabel="Дата", ylabel="Человек",
          title="Компартменты (N_eff=$(round(Int,N_eff_f)))",
          legend=:right, size=(1100,380))

panel = plot(p_heat, pA, pB, pC; layout=(4,1), size=(1100,1600))
display(panel)
savefig(panel, "./figures/covid/seird_wave$(WAVE_NUM)_fit.png")
savefig(panel, "./figures/covid/seird_wave$(WAVE_NUM)_fit.svg")
println("\nГрафики → figures/covid/seird_wave$(WAVE_NUM)_fit.(png|svg)")

# ─────────────────────────────────────────────────────────────
# 9. Сохранение
# ─────────────────────────────────────────────────────────────

df_traj = DataFrame(
    date=df_wave.date[1:n_plt],
    obs_cases=obs_cases[1:n_plt], obs_deaths=obs_deaths[1:n_plt],
    fit_cases=cases_fit[1:n_plt], fit_deaths=deaths_fit[1:n_plt],
    S=S_v, E=E_v, I=I_v, R=R_v, D=D_v,
)
df_par = DataFrame(
    param     = ["β",   "σ",    "γ",    "μ",  "E0",  "I0",  "N_eff", "R0"],
    fixed     = [false, true,   true,   false,false, false, false,   false],
    fit_value = [β_f,   σ_fix,  γ_fix,  μ_f,  E0_f,  I0_f,  N_eff_f, β_f/γ_fix],
)

@save "./data_out/seird_wave$(WAVE_NUM)61_results.jld2" df_traj df_par
CSV.write("./data_out/seird_wave$(WAVE_NUM)61_trajectories.csv", df_traj)
CSV.write("./data_out/seird_wave$(WAVE_NUM)61_params.csv", df_par)

println("\n=== Итог ==="); println(df_par); println("Готово ✓")
end