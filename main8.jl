# ============================================================
# main8.jl — Дробная SEIRD-модель (производная Капуто)
#             Численное решение: схема Грюнвальда–Летникова (GL)
#
# Дробная система порядка α_k ∈ (0.5, 1.0]:
#   D^{α_k} S = -β·S·I / N_eff
#   D^{α_k} E =  β·S·I / N_eff  - σ·E
#   D^{α_k} I =  σ·E             - (γ+μ)·I
#   D^{α_k} R =  γ·I
#   D^{α_k} D =  μ·I
#
# GL-дискретизация (h = 1 день):
#   u_n = F(u_{n-1}) + Σ_{j=1}^{n} |w_j^{α_k}| · u_{n-j}
#
# Физический смысл α_k < 1:
#   • «память» — текущий прирост зависит от всей истории
#   • субдиффузионный режим: более медленный рост и спад
#   • α_k → 1 вырождается в стандартную SEIRD (Euler)
#
# GL-веса:  |w_1| = α_k,  |w_j| = |w_{j-1}| · (j-1-α_k)/j,  j≥2
#   Сумма Σ_{j=1}^∞ |w_j| = 1 (нормировка из свойств GL)
#
# Свободные параметры:
#   INTEGER:    θ = [log β, log μ, log E₀, log I₀, log N_eff]
#   FRACTIONAL: θ = [log β, log μ, log E₀, log I₀, log N_eff, φ(α_k)]
#     φ(α_k) ∈ ℝ → α_k = 0.5 + 0.5·sigmoid(φ) ∈ (0.5, 1.0)
#
# Зависимости:
#   Pkg.add(["JLD2","CSV","DataFrames","Dates","Printf","Random","Optim","Plots","Statistics"])
# ============================================================

using JLD2, CSV, DataFrames, Dates, Printf, Random
using Optim
using Plots, Statistics

const N_CITY = 200_000.0
const SEED   = 42

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 1: Загрузка данных
# ─────────────────────────────────────────────────────────────

df = CSV.read("./data_out/covid_data_enriched_with_strains.csv", DataFrame)
all_waves = sort(unique(skipmissing(df.wave)))
println("Доступные волны: $all_waves")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 2: Ядро GL — абсолютные веса |w_j^α|
#
# Рекуррентная формула:
#   |w_1| = α
#   |w_j| = |w_{j-1}| · (j-1-α)/j   для j ≥ 2
#
# Для α=1: |w_1|=1, |w_j|=0 для j≥2 → стандартный Euler
# Для α<1: бесконечный «хвост» → память
# ─────────────────────────────────────────────────────────────

function gl_weights_abs(α::Float64, n_max::Int)
    w = zeros(n_max)
    w[1] = α
    for j in 2:n_max
        w[j] = w[j-1] * (j - 1 - α) / j
    end
    return w
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 3: Интегратор дробной SEIRD
#
# Явная GL-схема (h=1):
#   u_n = F(u_{n-1})  +  Σ_{j=1}^{n} |w_j| · u_{n-j}
#
# U[компартмент, время]: U[:,k] = u_{k-1}  (1-based)
# Компартменты: 1=S, 2=E, 3=I, 4=R, 5=D
# ─────────────────────────────────────────────────────────────

function run_fseird(σ_w, γ_w, β, μ, E0, I0, N_eff, α_k, T)
    N_eff  = max(N_eff, E0 + I0 + 1.0)
    U      = zeros(5, T)
    U[:, 1] = [N_eff - E0 - I0, E0, I0, 0.0, 0.0]

    # GL-веса вычисляем один раз на всю длину волны
    wabs = gl_weights_abs(α_k, T)

    for n in 1:T-1
        S, E, I, R, D = U[1,n], U[2,n], U[3,n], U[4,n], U[5,n]
        λ = β * max(I, 0.0) / N_eff

        # Вынуждающий член: правая часть дробного ОДУ при u_{n-1}
        fS = -λ * S
        fE =  λ * S   - σ_w * E
        fI =  σ_w * E - (γ_w + μ) * I
        fR =  γ_w * I
        fD =  μ * I

        # Память: Σ_{j=1}^{n} |w_j| · u_{n-j}
        # u_{n-j} (0-based) = U[:, n+1-j] (1-based)
        mS = mE = mI = mR = mD = 0.0
        for j in 1:n
            k   = n + 1 - j
            mS += wabs[j] * U[1, k]
            mE += wabs[j] * U[2, k]
            mI += wabs[j] * U[3, k]
            mR += wabs[j] * U[4, k]
            mD += wabs[j] * U[5, k]
        end

        # Новое состояние — зажимаем снизу нулём
        U[1, n+1] = max(fS + mS, 0.0)
        U[2, n+1] = max(fE + mE, 0.0)
        U[3, n+1] = max(fI + mI, 0.0)
        U[4, n+1] = max(fR + mR, 0.0)
        U[5, n+1] = max(fD + mD, 0.0)
    end

    return U
end

# Наблюдаемые: потоки E→I (новые случаи) и I→D (смерти)
frac_obs(U, σ_w, μ) = (σ_w .* U[2, :], μ .* U[3, :])

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 4: Параметризация α_k
#
# φ(α_k) ∈ ℝ → α_k ∈ (0.5, 1.0) через сигмоиду
# Стартуем вблизи α_k = 0.95 (близко к стандартной модели)
# ─────────────────────────────────────────────────────────────

alpha_from_phi(x) = 0.5 + 0.5 / (1.0 + exp(-x))
phi_from_alpha(α) = log(max((α - 0.5) / (1.0 - α), 1e-9))

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 5: Оценка скорости роста из начальной фазы
# ─────────────────────────────────────────────────────────────

function estimate_growth_rate(y)
    pk = argmax(y)
    n  = min(max(7, round(Int, pk * 0.3)), pk - 1)
    n < 3 && return 0.05
    seg = y[1:n]; ok = seg .> 0.5
    sum(ok) < 3 && return 0.05
    xs = Float64.(findall(ok)); ys = log.(seg[ok])
    x̄, ȳ = mean(xs), mean(ys)
    r = sum((xs.-x̄).*(ys.-ȳ)) / (sum((xs.-x̄).^2) + 1e-12)
    return max(r, 1e-4)
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 6: Подгонка одной волны
#           Прогон INTEGER (α=1) и FRACTIONAL (α свободен)
# ─────────────────────────────────────────────────────────────

function fit_wave_frac(wave_num; verbose=true)

    dfw = sort(filter(:wave => ==(wave_num), df), :date)
    nrow(dfw) < 10 && return nothing

    obs_c = Float64.(coalesce.(dfw.daily_interp_smooth, 0.0))
    obs_d = Float64.(coalesce.(dfw.deaths_daily, 0.0))
    T     = nrow(dfw)

    peak_cases  = maximum(obs_c)
    peak_idx    = argmax(obs_c)
    total_cases = sum(obs_c)
    cases_to_pk = sum(obs_c[1:peak_idx])

    # Параметры штамма из колонок датасета (среднее по волне)
    σ_w = mean(skipmissing(dfw.sigma_avg_per_day))
    γ_w = mean(skipmissing(dfw.gamma_avg_per_day))
    μ_p = mean(skipmissing(dfw.mu_avg_per_day))
    β_p = mean(skipmissing(dfw.beta_avg_per_day))

    strain = if "strain_name" in names(dfw)
        s = sort(combine(groupby(dropmissing(dfw,:strain_name),:strain_name),
                         nrow=>:n), :n, rev=true).strain_name
        isempty(s) ? "?" : first(s)
    else "?" end

    verbose && @printf "\n%s\n  Волна %d | %s | %d дней | peak=%.2f\n" ("="^55) wave_num strain T peak_cases
    verbose && @printf "  σ=%.5f  γ=%.5f  μ=%.6f  β_prior=%.5f\n" σ_w γ_w μ_p β_p

    # Аналитические начальные условия (те же что в main7)
    r_est  = estimate_growth_rate(obs_c)
    β_est  = (r_est + σ_w) * (r_est + γ_w + μ_p) / σ_w
    R0_est = β_est / γ_w
    N_eff_an = clamp(
        R0_est > 1.0 ? cases_to_pk * R0_est / (R0_est - 1.0) : total_cases * 5.0,
        total_cases * 2.0, N_CITY)
    I_peak = peak_cases / (γ_w + μ_p + 1e-12)
    I0_an  = max(0.5, I_peak * exp(-r_est * peak_idx))
    E0_an  = max(0.5, I0_an * (γ_w + μ_p) / σ_w)

    # ── Сеточный поиск (β × N_eff) с α_k=1 для скорости ──
    β_grid    = exp.(range(log(β_est/4),  log(β_est*4), length=15))
    Neff_grid = exp.(range(log(max(total_cases*1.5, 5.0)), log(N_CITY), length=15))

    log_obs_c = log1p.(obs_c)
    log_obs_d = log1p.(obs_d)
    DW = sum(obs_d) > 5.0 ? 3.0 : 0.0

    # Базовая функция потерь (log-MSE)
    function loss_core(β, μ, E0, I0, N_eff, α_k)
        U = run_fseird(σ_w, γ_w, β, μ, E0, I0, N_eff, α_k, T)
        mc, md = frac_obs(U, σ_w, μ)
        lc = mean((log1p.(max.(mc, 0.0)) .- log_obs_c).^2)
        ld = mean((log1p.(max.(md, 0.0)) .- log_obs_d).^2)
        return lc + DW * ld
    end

    best_g = (err=Inf, β=β_est, N=N_eff_an)
    for β in β_grid, N in Neff_grid
        r_loc  = max(1e-4, β - γ_w - σ_w/2)
        I0_loc = max(0.5, I_peak * exp(-r_loc * peak_idx))
        E0_loc = max(0.5, I0_loc * (γ_w + μ_p) / σ_w)
        v = loss_core(β, μ_p, E0_loc, I0_loc, N, 1.0)
        isinf(v) || v >= best_g.err || (best_g = (err=v, β=β, N=N))
    end

    r_g  = max(1e-4, best_g.β - γ_w - σ_w/2)
    I0_g = max(0.5, I_peak * exp(-r_g * peak_idx))
    E0_g = max(0.5, I0_g * (γ_w + μ_p) / σ_w)
    verbose && @printf "  Сетка → β=%.5f  N_eff=%.0f  loss=%.5f\n" best_g.β best_g.N best_g.err

    θ_grid = log.([best_g.β, μ_p, E0_g, I0_g, best_g.N])

    # ── INTEGER (α_k = 1.0 фиксировано) ──
    function loss_int(θ)
        β, μ, E0, I0, N_eff = exp.(θ)
        any(x -> x < 1e-12, (β, μ, E0, I0)) && return 1e18
        (N_eff < E0+I0+1 || N_eff > N_CITY)  && return 1e18
        v = loss_core(β, μ, E0, I0, N_eff, 1.0)
        isinf(v) ? 1e18 : v
    end

    rng1 = MersenneTwister(SEED + wave_num)
    cands_int = vcat(
        [θ_grid],
        [θ_grid .+ [0.3,0.8,1.0,1.0,0.4] .* randn(rng1, 5) for _ in 1:9],
    )
    nm_int = sort(map(cands_int) do θs
        r = optimize(loss_int, θs, NelderMead(),
                     Optim.Options(iterations=4_000, show_trace=false, g_tol=1e-7))
        (loss=Optim.minimum(r), θ=Optim.minimizer(r))
    end, by=x->x.loss)
    res_int = optimize(loss_int, nm_int[1].θ, LBFGS(),
                       Optim.Options(iterations=5_000, show_trace=false, g_tol=1e-12))
    β_i, μ_i, E0_i, I0_i, N_i = exp.(Optim.minimizer(res_int))
    loss_i = Optim.minimum(res_int)
    verbose && @printf "  INTEGER   → loss=%.6f  R0=%.3f  N_eff=%.0f\n" loss_i β_i/γ_w N_i

    # ── FRACTIONAL (α_k ∈ (0.5, 1.0) свободен) ──
    # θ = [log β, log μ, log E₀, log I₀, log N_eff, φ(α_k)]
    φ0 = phi_from_alpha(0.95)   # стартуем вблизи стандартной модели
    θ0_frac = [θ_grid; φ0]

    function loss_frac(θ)
        β, μ, E0, I0, N_eff = exp.(θ[1:5])
        α_k = alpha_from_phi(θ[6])
        any(x -> x < 1e-12, (β, μ, E0, I0)) && return 1e18
        (N_eff < E0+I0+1 || N_eff > N_CITY)  && return 1e18
        v = loss_core(β, μ, E0, I0, N_eff, α_k)
        isinf(v) ? 1e18 : v
    end

    rng2 = MersenneTwister(SEED + wave_num + 1000)
    cands_frac = vcat(
        [θ0_frac],
        [θ0_frac .+ [0.3,0.8,1.0,1.0,0.4,0.5] .* randn(rng2, 6) for _ in 1:14],
    )
    nm_frac = sort(map(cands_frac) do θs
        r = optimize(loss_frac, θs, NelderMead(),
                     Optim.Options(iterations=5_000, show_trace=false, g_tol=1e-7))
        (loss=Optim.minimum(r), θ=Optim.minimizer(r))
    end, by=x->x.loss)
    res_frac = optimize(loss_frac, nm_frac[1].θ, LBFGS(),
                        Optim.Options(iterations=8_000, show_trace=false, g_tol=1e-14))
    θ_f   = Optim.minimizer(res_frac)
    β_f, μ_f, E0_f, I0_f, N_f = exp.(θ_f[1:5])
    α_k_f = alpha_from_phi(θ_f[6])
    loss_f = Optim.minimum(res_frac)
    verbose && @printf "  FRACTIONAL→ loss=%.6f  R0=%.3f  N_eff=%.0f  α_k=%.4f\n" loss_f β_f/γ_w N_f α_k_f
    verbose && @printf "  Улучшение потери: %.2f%%\n" 100*(loss_i - loss_f)/max(loss_i, 1e-12)

    # Финальные траектории
    U_i = run_fseird(σ_w, γ_w, β_i, μ_i, E0_i, I0_i, N_i, 1.0,   T)
    U_f = run_fseird(σ_w, γ_w, β_f, μ_f, E0_f, I0_f, N_f, α_k_f, T)
    ci, di = frac_obs(U_i, σ_w, μ_i)
    cf, df_ = frac_obs(U_f, σ_w, μ_f)

    return (
        wave=wave_num, strain=strain, T=T,
        dates=dfw.date,
        obs_cases=obs_c, obs_deaths=obs_d,
        # INTEGER
        cases_int=ci, deaths_int=di,
        S_i=U_i[1,:], E_i=U_i[2,:], I_i=U_i[3,:], R_i=U_i[4,:], D_i=U_i[5,:],
        β_int=β_i, μ_int=μ_i, N_int=N_i,
        R0_int=β_i/γ_w, loss_int=loss_i,
        # FRACTIONAL
        cases_frac=cf, deaths_frac=df_,
        S_f=U_f[1,:], E_f=U_f[2,:], I_f=U_f[3,:], R_f=U_f[4,:], D_f=U_f[5,:],
        β_frac=β_f, μ_frac=μ_f, N_frac=N_f, α_k=α_k_f,
        R0_frac=β_f/γ_w, loss_frac=loss_f,
        # прочее
        σ_fix=σ_w, γ_fix=γ_w, β_prior=β_p, μ_prior=μ_p,
        R0_prior=β_p/γ_w,
        peak_obs=peak_cases,
        peak_int=maximum(ci), peak_frac=maximum(cf),
        loss_improve_pct=100*(loss_i - loss_f)/max(loss_i, 1e-12),
    )
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 7: Прогон по всем волнам
# ─────────────────────────────────────────────────────────────

println("\n" * "="^55)
println("  ДРОБНАЯ SEIRD: ПРОГОН ПО ВСЕМ ВОЛНАМ")
println("="^55)

results = []
for w in all_waves
    println("\nАнализ волны $w...")
    r = Base.invokelatest(fit_wave_frac, w)
    r !== nothing && push!(results, r)
end

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 8: Сводная таблица параметров
# ─────────────────────────────────────────────────────────────

df_summary = DataFrame(
    wave          = [r.wave          for r in results],
    strain        = [r.strain        for r in results],
    T_days        = [r.T             for r in results],
    peak_obs      = [round(r.peak_obs,  digits=2) for r in results],
    peak_int      = [round(r.peak_int,  digits=2) for r in results],
    peak_frac     = [round(r.peak_frac, digits=2) for r in results],
    R0_prior      = [round(r.R0_prior,  digits=3) for r in results],
    R0_int        = [round(r.R0_int,    digits=3) for r in results],
    R0_frac       = [round(r.R0_frac,   digits=3) for r in results],
    α_k           = [round(r.α_k,       digits=4) for r in results],
    N_eff_int     = [round(Int, r.N_int)           for r in results],
    N_eff_frac    = [round(Int, r.N_frac)          for r in results],
    N_int_pct     = [round(100*r.N_int /N_CITY, digits=1) for r in results],
    N_frac_pct    = [round(100*r.N_frac/N_CITY, digits=1) for r in results],
    loss_int      = [round(r.loss_int,  sigdigits=4) for r in results],
    loss_frac     = [round(r.loss_frac, sigdigits=4) for r in results],
    improve_pct   = [round(r.loss_improve_pct, digits=2) for r in results],
)

println("\n=== Сводная таблица ===")
println(df_summary)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 9: Графики
# ─────────────────────────────────────────────────────────────

mkpath("./figures/covid")
n_waves = length(results)

# ── 9А. Панель волн: наблюдения + integer + fractional ──────
wave_panels = map(results) do r
    p = plot(r.dates, r.obs_cases;
             label="данные", color=:steelblue, lw=2, alpha=0.8,
             title="Волна $(r.wave): $(r.strain)\n" *
                   "α_k=$(round(r.α_k,digits=3))  " *
                   "R0_frac=$(round(r.R0_frac,digits=2))  " *
                   "Δloss=$(round(r.loss_improve_pct,digits=1))%",
             titlefontsize=8, xlabel="", ylabel="случаев/день",
             legend=:topright, grid=true, gridalpha=0.3, size=(1100,380))
    plot!(p, r.dates, r.cases_int;
          label="integer (α=1)", color=:orange, lw=1.5, ls=:dash)
    plot!(p, r.dates, r.cases_frac;
          label="fractional", color=:crimson, lw=2.5)
    p
end

panel_waves = plot(wave_panels...; layout=(n_waves,1), size=(1100, 380*n_waves))
savefig(panel_waves, "./figures/covid/fseird_waves_fit.png")
savefig(panel_waves, "./figures/covid/fseird_waves_fit.svg")
println("→ fseird_waves_fit")

# ── 9Б. Log-шкала ───────────────────────────────────────────
wave_panels_log = map(results) do r
    p = plot(r.dates, log1p.(r.obs_cases);
             label="данные", color=:steelblue, lw=2, alpha=0.8,
             title="Волна $(r.wave) (log-шкала)", titlefontsize=8,
             xlabel="", ylabel="log(случаев+1)",
             legend=:topright, grid=true, gridalpha=0.3)
    plot!(p, r.dates, log1p.(max.(r.cases_int,  0.0));
          label="integer", color=:orange, lw=1.5, ls=:dash)
    plot!(p, r.dates, log1p.(max.(r.cases_frac, 0.0));
          label="fractional", color=:crimson, lw=2.5)
    p
end
panel_log = plot(wave_panels_log...; layout=(n_waves,1), size=(1100, 320*n_waves))
savefig(panel_log, "./figures/covid/fseird_waves_log.png")
savefig(panel_log, "./figures/covid/fseird_waves_log.svg")
println("→ fseird_waves_log")

# ── 9В. Компартменты: integer vs fractional ─────────────────
for r in results
    pc = plot(r.dates, [r.S_f r.E_f r.I_f r.R_f r.D_f];
              label=["S" "E" "I" "R" "D"],
              lw=2, xlabel="Дата", ylabel="Человек",
              title="Компартменты (дробная модель) — волна $(r.wave)  α_k=$(round(r.α_k,digits=3))",
              legend=:right, size=(1100,380), grid=true, gridalpha=0.3)
    plot!(pc, r.dates, r.I_i; label="I (integer)", color=:black, lw=1.5, ls=:dot)
    savefig(pc, "./figures/covid/fseird_compartments_wave$(r.wave).png")
end
println("→ fseird_compartments_wave*")

# ── 9Г. α_k по волнам ───────────────────────────────────────
p_alpha = bar(df_summary.wave, df_summary.α_k;
              color=:mediumorchid, alpha=0.85,
              xlabel="Волна", ylabel="α_k",
              title="Порядок дробной производной по волнам\n(α_k=1 ↔ стандартная SEIRD)",
              legend=false, ylim=(0,1.05), size=(800,420))
hline!(p_alpha, [1.0]; color=:black, ls=:dash, lw=1, label="α=1")
annotate!(p_alpha, df_summary.wave, df_summary.α_k .+ 0.03,
          [text("$(v)", 9, :center) for v in df_summary.α_k])
savefig(p_alpha, "./figures/covid/fseird_alpha_by_wave.png")
savefig(p_alpha, "./figures/covid/fseird_alpha_by_wave.svg")
println("→ fseird_alpha_by_wave")

# ── 9Д. Сравнение R0: prior / integer / fractional ──────────
p_r0 = plot(df_summary.wave, df_summary.R0_prior;
            label="R0 prior", marker=:circle, lw=2, color=:steelblue,
            xlabel="Волна", ylabel="R0",
            title="R0: prior (штамм) vs Integer vs Fractional",
            legend=:topright, size=(800,420))
plot!(p_r0, df_summary.wave, df_summary.R0_int;
      label="R0 integer",    marker=:square,  lw=2, color=:orange,   ls=:dash)
plot!(p_r0, df_summary.wave, df_summary.R0_frac;
      label="R0 fractional", marker=:star5,   lw=2, color=:crimson)
savefig(p_r0, "./figures/covid/fseird_R0_comparison.png")
savefig(p_r0, "./figures/covid/fseird_R0_comparison.svg")
println("→ fseird_R0_comparison")

# ── 9Е. N_eff: integer vs fractional ────────────────────────
x_pos = collect(1:n_waves)
width = 0.35
p_neff = plot(; xlabel="Волна",
                ylabel="% от N_city",
                title="N_eff по волнам: integer vs fractional",
                legend=:topright, size=(800,420))
bar!(p_neff, x_pos .- width/2, df_summary.N_int_pct;
     width=width, label="Integer", color=:orange, alpha=0.8)
bar!(p_neff, x_pos .+ width/2, df_summary.N_frac_pct;
     width=width, label="Fractional", color=:mediumorchid, alpha=0.8)
plot!(p_neff, xticks=(x_pos, string.(df_summary.wave)))
savefig(p_neff, "./figures/covid/fseird_Neff_comparison.png")
savefig(p_neff, "./figures/covid/fseird_Neff_comparison.svg")
println("→ fseird_Neff_comparison")

# ── 9Ж. Улучшение потери (loss_int → loss_frac) ─────────────
p_loss = plot(df_summary.wave, df_summary.loss_int;
              label="loss integer", marker=:circle, lw=2, color=:orange,
              xlabel="Волна", ylabel="log-MSE",
              title="Потеря: integer vs fractional",
              legend=:topright, size=(800,420))
plot!(p_loss, df_summary.wave, df_summary.loss_frac;
      label="loss fractional", marker=:star5, lw=2, color=:crimson)
p_impr = bar(df_summary.wave, df_summary.improve_pct;
             color=:seagreen, alpha=0.8, legend=false,
             xlabel="Волна", ylabel="Δloss, %",
             title="Относительное улучшение потери (fractional vs integer)")
panel_loss = plot(p_loss, p_impr; layout=(2,1), size=(800,700))
savefig(panel_loss, "./figures/covid/fseird_loss_improvement.png")
savefig(panel_loss, "./figures/covid/fseird_loss_improvement.svg")
println("→ fseird_loss_improvement")

# ── 9З. Память: ядра GL для разных α_k ──────────────────────
# Показываем насколько далеко "смотрит назад" модель
lags = 1:60
p_mem = plot(title="Ядро памяти GL: веса |w_j^{α}| по лагам",
             xlabel="Лаг j (дней)", ylabel="|w_j|",
             legend=:topright, size=(900,450))
alphas_demo = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
colors_demo = [:purple, :blue, :teal, :green, :orange, :red]
for (αd, c) in zip(alphas_demo, colors_demo)
    w = gl_weights_abs(αd, maximum(lags))
    plot!(p_mem, lags, w[lags];
          label="α=$(αd)", color=c, lw=2, alpha=0.85)
end
savefig(p_mem, "./figures/covid/fseird_memory_kernels.png")
savefig(p_mem, "./figures/covid/fseird_memory_kernels.svg")
println("→ fseird_memory_kernels")

# ── 9И. Накопленная память: Σ |w_j| vs лаг ──────────────────
p_cumem = plot(title="Накопленная память: Σ_{j=1}^{L} |w_j^α|",
               xlabel="Горизонт L (дней)", ylabel="Σ |w_j|",
               legend=:bottomright, size=(900,420))
for (αd, c) in zip(alphas_demo, colors_demo)
    w = gl_weights_abs(αd, maximum(lags))
    plot!(p_cumem, lags, cumsum(w[lags]); label="α=$(αd)", color=c, lw=2)
end
hline!(p_cumem, [1.0]; color=:black, ls=:dot, lw=1, label="предел=1")
savefig(p_cumem, "./figures/covid/fseird_cumulative_memory.png")
savefig(p_cumem, "./figures/covid/fseird_cumulative_memory.svg")
println("→ fseird_cumulative_memory")

# ── 9К. Фазовый портрет I(S): integer vs fractional ─────────
for r in results
    pph = plot(r.S_i, r.I_i;
               label="integer (α=1)", color=:orange, lw=1.5, ls=:dash,
               xlabel="S (восприимчивые)", ylabel="I (инфекционные)",
               title="Фазовый портрет S-I — волна $(r.wave)  α_k=$(round(r.α_k,digits=3))",
               legend=:topright, size=(700,500))
    plot!(pph, r.S_f, r.I_f; label="fractional", color=:crimson, lw=2)
    scatter!(pph, [r.S_i[1]], [r.I_i[1]]; color=:black, markersize=8, label="старт")
    savefig(pph, "./figures/covid/fseird_phase_wave$(r.wave).png")
end
println("→ fseird_phase_wave*")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ 10: Сохранение результатов
# ─────────────────────────────────────────────────────────────

# Сводная таблица параметров
CSV.write("./data_out/fseird_params.csv", df_summary)

# Все траектории: integer + fractional в одном файле
df_traj = vcat([
    DataFrame(
        wave        = r.wave,
        strain      = r.strain,
        date        = r.dates,
        obs_cases   = r.obs_cases,
        obs_deaths  = r.obs_deaths,
        # integer
        fit_int     = r.cases_int,
        S_int=r.S_i, E_int=r.E_i, I_int=r.I_i, R_int=r.R_i, D_int=r.D_i,
        # fractional
        fit_frac    = r.cases_frac,
        alpha_k     = fill(r.α_k, r.T),  # постоянная по волне
        S_frac=r.S_f, E_frac=r.E_f, I_frac=r.I_f, R_frac=r.R_f, D_frac=r.D_f,
    ) for r in results
]...)
CSV.write("./data_out/fseird_trajectories.csv", df_traj)

# JLD2 с полными объектами
@save "./data_out/fseird_results.jld2" results df_summary df_traj

println("\n=== Сохранено ===")
println("  ./data_out/fseird_params.csv")
println("  ./data_out/fseird_trajectories.csv")
println("  ./data_out/fseird_results.jld2")
println("\n=== Итоговая таблица ===")
println(df_summary[:, [:wave,:strain,:α_k,:R0_prior,:R0_int,:R0_frac,:loss_int,:loss_frac,:improve_pct]])
println("\nГотово ✓")
