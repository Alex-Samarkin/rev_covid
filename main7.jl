# ============================================================
# main7.jl — Автоматический прогон SEIRD по всем волнам
#
# Логика:
#   для каждой волны:
#     1. Фильтруем df по номеру волны
#     2. Читаем σ, γ, μ, β прямо из колонок датасета (по дате)
#        — они уже присоединены в main4 через strain_daily_resolved
#     3. Сеточный поиск (β × N_eff) для нахождения стартовой точки
#     4. NelderMead + LBFGS
#     5. Сохраняем параметры и траектории по каждой волне
#
# Свободные параметры: β, μ, E₀, I₀, N_eff
# Фиксированные: σ, γ — из колонок датасета (среднее по волне)
# ============================================================

using JLD2, CSV, DataFrames, Dates, Printf, Random
using DifferentialEquations
using Optim
using Plots, Statistics

const N_CITY = 200_000.0
const SEED   = 42

# ─────────────────────────────────────────────────────────────
# 1. Загрузка данных
# ─────────────────────────────────────────────────────────────

df = CSV.read("./data_out/covid_data_enriched_with_strains.csv", DataFrame)

# Все доступные волны (без missing)
all_waves = sort(unique(skipmissing(df.wave)))
println("Доступные волны: $all_waves")

# ─────────────────────────────────────────────────────────────
# 2. ODE — N_eff внутри вектора параметров
# ─────────────────────────────────────────────────────────────

# σ_w, γ_w — захватываются из замыкания при каждом прогоне волны
function make_seird(σ_w, γ_w)
    function seird!(du, u, p, t)
        S, E, I, R, D = u
        β, μ, N_eff   = p
        λ = β * I / N_eff
        du[1] = -λ * S
        du[2] =  λ * S    - σ_w * E
        du[3] =  σ_w * E  - (γ_w + μ) * I
        du[4] =  γ_w * I
        du[5] =  μ * I
    end
    return seird!
end

function run_seird(seird!, β, μ, E0, I0, N_eff, T)
    N_eff = max(N_eff, E0 + I0 + 1.0)
    u0    = [N_eff - E0 - I0, E0, I0, 0.0, 0.0]
    prob  = ODEProblem(seird!, u0, (0.0, Float64(T-1)), (β, μ, N_eff))
    solve(prob, AutoVern7(Rodas5P());
          saveat=1.0, abstol=1e-8, reltol=1e-6,
          maxiters=1_000_000, verbose=false)
end

# ─────────────────────────────────────────────────────────────
# 3. Вспомогательные функции
# ─────────────────────────────────────────────────────────────

function estimate_growth_rate(y)
    pk = argmax(y)
    n  = max(7, round(Int, pk * 0.3))
    n  = min(n, pk - 1)
    n < 3 && return 0.05
    seg = y[1:n]; ok = seg .> 0.5
    sum(ok) < 3 && return 0.05
    xs = Float64.(findall(ok)); ys = log.(seg[ok])
    x̄, ȳ = mean(xs), mean(ys)
    r = sum((xs.-x̄).*(ys.-ȳ)) / (sum((xs.-x̄).^2)+1e-12)
    return max(r, 1e-4)
end

# ─────────────────────────────────────────────────────────────
# 4. Функция подгонки одной волны
# ─────────────────────────────────────────────────────────────

function fit_wave(wave_num; verbose=true)

    dfw = sort(filter(:wave => ==(wave_num), df), :date)
    nrow(dfw) < 10 && return nothing

    # Changed to smooth data
    obs_c = Float64.(coalesce.(dfw.daily_interp_smooth,  0.0))
    obs_d = Float64.(coalesce.(dfw.deaths_daily,  0.0))
    T     = nrow(dfw)

    peak_cases  = maximum(obs_c)
    peak_idx    = argmax(obs_c)
    total_cases = sum(obs_c)
    cases_to_pk = sum(obs_c[1:peak_idx])

    # --- Параметры штамма из колонок датасета (среднее по волне) ---
    σ_w = mean(skipmissing(dfw.sigma_avg_per_day))
    γ_w = mean(skipmissing(dfw.gamma_avg_per_day))
    μ_p = mean(skipmissing(dfw.mu_avg_per_day))
    β_p = mean(skipmissing(dfw.beta_avg_per_day))

    # Имя штамма — самый частый в волне
    strain = if "strain_name" in names(dfw)
        strains = sort(combine(groupby(dropmissing(dfw, :strain_name), :strain_name),
                               nrow => :n), :n, rev=true).strain_name
        isempty(strains) ? "?" : first(strains)
    else "?"
    end

    verbose && @printf "\n%s\n  Волна %d | штамм: %s | %d дней | peak=%.2f\n" (
        "="^55) wave_num strain T peak_cases
    verbose && @printf "  σ=%.5f  γ=%.5f  μ=%.6f  β_prior=%.5f  R0=%.2f\n" σ_w γ_w μ_p β_p β_p/γ_w

    seird! = make_seird(σ_w, γ_w)

    # --- Аналитические начальные условия ---
    r_est  = estimate_growth_rate(obs_c)
    β_est  = (r_est + σ_w) * (r_est + γ_w + μ_p) / σ_w
    R0_est = β_est / γ_w

    N_eff_an = R0_est > 1.0 ?
        cases_to_pk * R0_est / (R0_est - 1.0) :
        total_cases * 5.0
    N_eff_an = clamp(N_eff_an, total_cases * 2.0, N_CITY)

    I_peak   = peak_cases / (γ_w + μ_p + 1e-12)
    I0_an    = max(0.5, I_peak * exp(-r_est * peak_idx))
    E0_an    = max(0.5, I0_an * (γ_w + μ_p) / σ_w)

    # --- Сеточный поиск β × N_eff (20×20) ---
    β_grid    = exp.(range(log(β_est/4), log(β_est*4), length=20))
    Neff_grid = exp.(range(log(max(total_cases*1.5,5.0)), log(N_CITY), length=20))

    log_obs_c = log1p.(obs_c)
    log_obs_d = log1p.(obs_d)
    DW = sum(obs_d) > 5.0 ? 3.0 : 0.0

    function loss_grid(β, μ, E0, I0, N_eff)
        sol = run_seird(seird!, β, μ, E0, I0, N_eff, T)
        sol.retcode !== ReturnCode.Success && return Inf
        mc = σ_w .* sol[2, :]; md = μ .* sol[3, :]
        n  = min(length(mc), T)
        lc = mean((log1p.(max.(mc[1:n],0.0)) .- log_obs_c[1:n]).^2)
        ld = mean((log1p.(max.(md[1:n],0.0)) .- log_obs_d[1:n]).^2)
        return lc + DW * ld
    end

    best_g = (err=Inf, β=β_est, N=N_eff_an)
    for β in β_grid, N in Neff_grid
        r_loc  = max(1e-4, β - γ_w - σ_w/2)
        I0_loc = max(0.5, I_peak * exp(-r_loc * peak_idx))
        E0_loc = max(0.5, I0_loc * (γ_w + μ_p) / σ_w)
        v = loss_grid(β, μ_p, E0_loc, I0_loc, N)
        isinf(v) || v >= best_g.err || (best_g = (err=v, β=β, N=N))
    end

    verbose && @printf "  Сетка → β=%.5f  N_eff=%.0f  loss=%.5f\n" best_g.β best_g.N best_g.err

    # --- Функция потерь для оптимизации ---
    # θ = [log β, log μ, log E₀, log I₀, log N_eff]
    function loss(θ)
        β, μ, E0, I0, N_eff = exp.(θ)
        any(x -> x < 1e-12, (β, μ, E0, I0)) && return 1e18
        N_eff < E0 + I0 + 1.0               && return 1e18
        N_eff > N_CITY                       && return 1e18
        v = loss_grid(β, μ, E0, I0, N_eff)
        isinf(v) && return 1e18
        return v
    end

    # Стартовая точка из сетки
    r_g   = max(1e-4, best_g.β - γ_w - σ_w/2)
    I0_g  = max(0.5, I_peak * exp(-r_g * peak_idx))
    E0_g  = max(0.5, I0_g * (γ_w + μ_p) / σ_w)
    θ0 = log.([best_g.β, μ_p, E0_g, I0_g, best_g.N])

    # --- NelderMead + случайные старты ---
    rng = MersenneTwister(SEED + wave_num)
    cands = Vector{Vector{Float64}}()
    push!(cands, θ0)
    append!(cands, [θ0 .+ [0.3,0.8,1.0,1.0,0.4] .* randn(rng,5) for _ in 1:14])
    nm = map(cands) do θs
        r = optimize(loss, θs, NelderMead(),
                     Optim.Options(iterations=6_000, show_trace=false, g_tol=1e-8))
        (loss=Optim.minimum(r), θ=Optim.minimizer(r))
    end
    sort!(nm, by=x->x.loss)
    verbose && @printf "  NelderMead → loss=%.6f\n" nm[1].loss

    # --- LBFGS уточнение ---
    res = optimize(loss, nm[1].θ, LBFGS(),
                   Optim.Options(iterations=10_000, show_trace=false, g_tol=1e-14))
    θ_f = Optim.minimizer(res)
    β_f, μ_f, E0_f, I0_f, N_f = exp.(θ_f)
    loss_f = Optim.minimum(res)

    verbose && @printf "  LBFGS      → loss=%.6f\n" loss_f
    verbose && @printf "  β=%.5f (R0=%.3f)  μ=%.6f  N_eff=%.0f  E₀=%.1f  I₀=%.1f\n" β_f β_f/γ_w μ_f N_f E0_f I0_f

    # --- Финальная траектория ---
    sol = run_seird(seird!, β_f, μ_f, E0_f, I0_f, N_f, T)
    cases_fit  = σ_w .* sol[2, :]
    deaths_fit = μ_f .* sol[3, :]
    n_plt = min(length(cases_fit), T)

    return (
        wave        = wave_num,
        strain      = strain,
        dates       = dfw.date[1:n_plt],
        obs_cases   = obs_c[1:n_plt],
        obs_deaths  = obs_d[1:n_plt],
        fit_cases   = cases_fit[1:n_plt],
        fit_deaths  = deaths_fit[1:n_plt],
        S = sol[1,1:n_plt], E = sol[2,1:n_plt],
        I = sol[3,1:n_plt], R = sol[4,1:n_plt], D = sol[5,1:n_plt],
        # параметры
        σ_fix=σ_w, γ_fix=γ_w,
        β_prior=β_p, μ_prior=μ_p,
        β_fit=β_f,  μ_fit=μ_f,
        E0=E0_f, I0=I0_f, N_eff=N_f,
        R0_prior=β_p/γ_w, R0_fit=β_f/γ_w,
        loss=loss_f,
        T=T, peak_obs=peak_cases,
        peak_fit=isempty(cases_fit) ? 0.0 : maximum(cases_fit),
    )
end

# ─────────────────────────────────────────────────────────────
# 5. Прогон по всем волнам
# ─────────────────────────────────────────────────────────────

println("\n" * "="^55)
println("  ПРОГОН SEIRD ПО ВСЕМ ВОЛНАМ")
println("="^55)

results = []
for w in all_waves
    println("Идет анализ волны $(w)")
    r = fit_wave(w)
    r !== nothing && push!(results, r)
end

# ─────────────────────────────────────────────────────────────
# 6. Сводная таблица параметров
# ─────────────────────────────────────────────────────────────

df_summary = DataFrame(
    wave     = [r.wave     for r in results],
    strain   = [r.strain   for r in results],
    T_days   = [r.T        for r in results],
    peak_obs = [r.peak_obs for r in results],
    peak_fit = [round(r.peak_fit, digits=1) for r in results],
    peak_ratio = [round(r.peak_fit / max(r.peak_obs,1e-6), digits=3) for r in results],
    R0_prior = [round(r.R0_prior, digits=3) for r in results],
    R0_fit   = [round(r.R0_fit,   digits=3) for r in results],
    β_prior  = [round(r.β_prior,  digits=5) for r in results],
    β_fit    = [round(r.β_fit,    digits=5) for r in results],
    σ_fix    = [round(r.σ_fix,    digits=5) for r in results],
    γ_fix    = [round(r.γ_fix,    digits=5) for r in results],
    μ_prior  = [round(r.μ_prior,  digits=6) for r in results],
    μ_fit    = [round(r.μ_fit,    digits=6) for r in results],
    N_eff    = [round(Int, r.N_eff) for r in results],
    N_eff_pct = [round(100*r.N_eff/N_CITY, digits=1) for r in results],
    loss     = [round(r.loss, sigdigits=4) for r in results],
)

println("\n=== Сводная таблица параметров ===")
println(df_summary)

# ─────────────────────────────────────────────────────────────
# 7. Графики — панель по всем волнам
# ─────────────────────────────────────────────────────────────

n_waves = length(results)
plots_lin = []   # линейная шкала
plots_log = []   # log-шкала

for r in results
    # Линейная
    p = plot(r.dates, r.obs_cases;
             label="данные", color=:steelblue, lw=1.5, alpha=0.7,
             title="Волна $(r.wave): $(r.strain)\nloss=$(round(r.loss,sigdigits=3))  R0=$(round(r.R0_fit,digits=2))  N_eff=$(round(Int,r.N_eff))",
             titlefontsize=8, xlabel="", ylabel="случаев/день",
             legend=:topright, grid=true, gridalpha=0.3)
    plot!(p, r.dates, r.fit_cases;
          label="fit", color=:crimson, lw=2)
    push!(plots_lin, p)

    # Log
    pl = plot(r.dates, log1p.(r.obs_cases);
              label="данные", color=:steelblue, lw=1.5, alpha=0.7,
              title="Волна $(r.wave) (log)", titlefontsize=8,
              xlabel="", ylabel="log(случаев+1)",
              legend=:topright, grid=true, gridalpha=0.3)
    plot!(pl, r.dates, log1p.(max.(r.fit_cases, 0.0));
          label="fit", color=:crimson, lw=2)
    push!(plots_log, pl)
end

# Панель линейная
panel_lin = plot(plots_lin...; layout=(n_waves,1),
                 size=(1100, 350*n_waves))
savefig(panel_lin, "./figures/covid/seird_all_waves_linear.png")
savefig(panel_lin, "./figures/covid/seird_all_waves_linear.svg")

# Панель log
panel_log = plot(plots_log...; layout=(n_waves,1),
                 size=(1100, 350*n_waves))
savefig(panel_log, "./figures/covid/seird_all_waves_log.png")
savefig(panel_log, "./figures/covid/seird_all_waves_log.svg")

println("\nГрафики → figures/covid/seird_all_waves_(linear|log).(png|svg)")

# --- Сравнительный график R0 prior vs fit ---
p_r0 = plot(df_summary.wave, df_summary.R0_prior;
            label="R0 prior (штамм)", marker=:circle, lw=2, color=:orange,
            xlabel="Волна", ylabel="R0",
            title="R0: данные штамма vs подобранное",
            legend=:topright, size=(800,400))
plot!(p_r0, df_summary.wave, df_summary.R0_fit;
      label="R0 fit", marker=:star5, lw=2, color=:crimson)
savefig(p_r0, "./figures/covid/seird_R0_comparison.png")
savefig(p_r0, "./figures/covid/seird_R0_comparison.svg")

# --- График N_eff по волнам ---
p_neff = bar(df_summary.wave, df_summary.N_eff_pct;
             label="N_eff (% от N_city)", color=:steelblue, alpha=0.8,
             xlabel="Волна", ylabel="% от 200 000",
             title="Эффективная восприимчивая популяция по волнам",
             legend=false, size=(800,400))
savefig(p_neff, "./figures/covid/seird_Neff_by_wave.png")
savefig(p_neff, "./figures/covid/seird_Neff_by_wave.svg")

println("Графики R0 и N_eff → figures/covid/seird_R0_comparison.*, seird_Neff_by_wave.*")

# ─────────────────────────────────────────────────────────────
# 8. Сохранение результатов
# ─────────────────────────────────────────────────────────────

# Сводная таблица параметров
CSV.write("./data_out/seird_all_waves_params.csv", df_summary)

# Траектории по каждой волне — один большой CSV
df_all_traj = vcat([
    DataFrame(
        wave       = r.wave,
        strain     = r.strain,
        date       = r.dates,
        obs_cases  = r.obs_cases,
        obs_deaths = r.obs_deaths,
        fit_cases  = r.fit_cases,
        fit_deaths = r.fit_deaths,
        S=r.S, E=r.E, I=r.I, R=r.R, D=r.D,
    ) for r in results
]...)
CSV.write("./data_out/seird_all_waves_trajectories.csv", df_all_traj)

@save "./data_out/seird_all_waves_results.jld2" results df_summary df_all_traj

println("\n=== Сохранено ===")
println("  ./data_out/seird_all_waves_params.csv")
println("  ./data_out/seird_all_waves_trajectories.csv")
println("  ./data_out/seird_all_waves_results.jld2")
println("Готово ✓")
