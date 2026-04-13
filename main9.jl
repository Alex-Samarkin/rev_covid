# ============================================================
# main9.jl — Формальное сравнение моделей
#             Integer SEIRD (main7) vs Fractional SEIRD (main8)
#
# Методы сравнения:
#
#   1. Метрики качества подгонки (per wave + aggregate)
#      RMSE, MAE, MAPE, R² — на линейной шкале
#      log-RMSE             — на log-шкале (метрика оптимизации)
#
#   2. Информационные критерии (AIC, BIC)
#      Предположение: остатки log1p(fit) - log1p(obs) ~ N(0, σ²)
#      AIC = n·log(MSE) + 2k,   k_int=5, k_frac=6
#      BIC = n·log(MSE) + k·log(n)
#      ΔAIC > 2 → существенная поддержка дробной модели
#      ΔAIC > 10 → сильная поддержка
#
#   3. Likelihood Ratio Test (LRT)
#      Модели вложены: integer = fractional при α_k=1
#      Λ = n·(log MSE_int - log MSE_frac) ~ χ²(1) при H0: α_k=1
#      p = 1 - cdf(χ²(1), Λ)
#
#   4. Анализ остатков
#      Остатки: r_t = log1p(obs_t) - log1p(fit_t)
#      ACF(r): автокорреляция — хороший fit → r белый шум
#      SS_res / SS_tot = 1 - R²
#
#   5. Диагностические графики
#      Scatter obs vs fit, residuals vs time, ACF, Q-Q
# ============================================================

using JLD2, CSV, DataFrames, Dates, Printf, Random
using Plots
using StatsPlots
using Statistics, LinearAlgebra

# χ²-распределение (CDF) — только стандартная библиотека
# χ²(1): P(X ≤ x) = erf(√(x/2))
# Используем нижнюю неполную гамма-функцию через аппроксимацию
function chi2_cdf_1dof(x)
    x <= 0 && return 0.0
    # erf через серию Тейлора — достаточно для χ²(1)
    z = sqrt(x / 2)
    # erf(z): используем формулу Абрамовица-Стегуна 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 +
           t * (-0.284496736 +
           t * (1.421413741 +
           t * (-1.453152027 +
           t *  1.061405429))))
    return 1.0 - poly * exp(-z^2)
end
pval_chi2_1(Λ) = max(0.0, 1.0 - chi2_cdf_1dof(Λ))

# ─────────────────────────────────────────────────────────────
# 1. Загрузка результатов обеих моделей
# ─────────────────────────────────────────────────────────────

# main7: стандартная SEIRD (DifferentialEquations.jl, RK4/5)
df7 = CSV.read("./data_out/seird_all_waves_trajectories.csv", DataFrame)
pm7 = CSV.read("./data_out/seird_all_waves_params.csv",       DataFrame)

# main8: дробная SEIRD (GL-схема, integer + fractional внутри)
df8 = CSV.read("./data_out/fseird_trajectories.csv", DataFrame)
pm8 = CSV.read("./data_out/fseird_params.csv",       DataFrame)

all_waves = sort(unique(df7.wave))
println("Волны для сравнения: $all_waves")

# ─────────────────────────────────────────────────────────────
# 2. Вычисление метрик качества
#
# Для каждой волны и каждой модели:
#   • RMSE  = √(mean((obs-fit)²))
#   • MAE   = mean(|obs-fit|)
#   • MAPE  = mean(|obs-fit|/max(obs,1)) * 100%
#   • R²    = 1 - SS_res/SS_tot  (на линейной шкале)
#   • logMSE = mean((log1p(fit)-log1p(obs))²)  — метрика оптимизации
#   • AIC   = n·log(logMSE) + 2k
#   • BIC   = n·log(logMSE) + k·log(n)
#   • LRT_Λ = n·(log(logMSE_int) - log(logMSE_frac))  [только int vs frac]
# ─────────────────────────────────────────────────────────────

# Число свободных параметров каждой модели
const K_INT  = 5   # β, μ, E₀, I₀, N_eff
const K_FRAC = 6   # + α_k

function wave_metrics(obs, fit, k)
    n    = length(obs)
    res  = obs .- fit
    ss_r = sum(res.^2)
    ss_t = sum((obs .- mean(obs)).^2)
    rmse = sqrt(ss_r / n)
    mae  = mean(abs.(res))
    mape = mean(abs.(res) ./ max.(obs, 1.0)) * 100
    r2   = 1.0 - ss_r / max(ss_t, 1e-12)

    log_res = log1p.(max.(fit, 0.0)) .- log1p.(obs)
    lmse    = mean(log_res.^2)

    aic = n * log(lmse + 1e-30) + 2 * k
    bic = n * log(lmse + 1e-30) + k * log(n)

    return (n=n, rmse=rmse, mae=mae, mape=mape, r2=r2, lmse=lmse, aic=aic, bic=bic)
end

rows = []
for w in all_waves
    obs7 = df7[df7.wave .== w, :obs_cases]
    fit7 = df7[df7.wave .== w, :fit_cases]

    # main8 содержит три варианта: integer (fit_int) и fractional (fit_frac)
    obs8    = df8[df8.wave .== w, :obs_cases]
    fit8_i  = df8[df8.wave .== w, :fit_int]
    fit8_f  = df8[df8.wave .== w, :fit_frac]

    # Берём минимальную общую длину (на случай небольших расхождений)
    n = min(length(obs7), length(obs8))
    obs = obs7[1:n]; f7 = fit7[1:n]; f8i = fit8_i[1:n]; f8f = fit8_f[1:n]

    m7   = wave_metrics(obs, f7,   K_INT)
    m8i  = wave_metrics(obs, f8i,  K_INT)
    m8f  = wave_metrics(obs, f8f,  K_FRAC)

    # LRT: integer GL (main8) vs fractional GL (main8)
    # Только их можно напрямую сравнивать через LRT (одна функция потерь)
    Λ   = n * (log(m8i.lmse + 1e-30) - log(m8f.lmse + 1e-30))
    Λ   = max(Λ, 0.0)   # при численном шуме может быть чуть < 0
    pv  = pval_chi2_1(Λ)

    # ΔAIC, ΔBIC: main7 vs main8-frac (разные решатели, неформально)
    ΔAIC_7vs8f = m7.aic   - m8f.aic
    ΔBIC_7vs8f = m7.bic   - m8f.bic

    # α_k этой волны
    α_k = pm8[pm8.wave .== w, :α_k][1]

    push!(rows, (
        wave = w,
        n    = n,
        α_k  = α_k,
        # main7
        rmse_7   = m7.rmse,  mae_7  = m7.mae,  mape_7  = m7.mape,
        r2_7     = m7.r2,    lmse_7 = m7.lmse, aic_7   = m7.aic,   bic_7  = m7.bic,
        # main8 integer (GL Euler)
        rmse_8i  = m8i.rmse, mae_8i = m8i.mae, mape_8i = m8i.mape,
        r2_8i    = m8i.r2,   lmse_8i= m8i.lmse,aic_8i  = m8i.aic,  bic_8i = m8i.bic,
        # main8 fractional
        rmse_8f  = m8f.rmse, mae_8f = m8f.mae, mape_8f = m8f.mape,
        r2_8f    = m8f.r2,   lmse_8f= m8f.lmse,aic_8f  = m8f.aic,  bic_8f = m8f.bic,
        # сравнение
        LRT_Lambda = Λ,
        LRT_pval   = pv,
        ΔAIC_int_vs_frac = m8i.aic - m8f.aic,  # >0 → frac лучше
        ΔBIC_int_vs_frac = m8i.bic - m8f.bic,
        ΔAIC_7_vs_8f     = ΔAIC_7vs8f,          # >0 → frac лучше main7
        ΔBIC_7_vs_8f     = ΔBIC_7vs8f,
    ))
end

df_cmp = DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 3. Итоговая таблица сравнения
# ─────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("  СРАВНЕНИЕ МОДЕЛЕЙ: INTEGER SEIRD vs FRACTIONAL SEIRD")
println("="^70)

println("\n── Метрики качества (RMSE на линейной шкале) ──")
for r in eachrow(df_cmp)
    @printf "  Волна %d (n=%d, α_k=%.3f):\n" r.wave r.n r.α_k
    @printf "    RMSE: main7=%.3f  int_GL=%.3f  frac_GL=%.3f\n" r.rmse_7 r.rmse_8i r.rmse_8f
    @printf "    R²  : main7=%.4f  int_GL=%.4f  frac_GL=%.4f\n" r.r2_7   r.r2_8i   r.r2_8f
    @printf "    MAPE: main7=%.1f%%  int_GL=%.1f%%  frac_GL=%.1f%%\n" r.mape_7 r.mape_8i r.mape_8f
end

println("\n── Информационные критерии (ΔAIC, ΔBIC: >0 → frac лучше) ──")
for r in eachrow(df_cmp)
    sig = r.LRT_pval < 0.001 ? "***" : r.LRT_pval < 0.01 ? "**" :
          r.LRT_pval < 0.05  ? "*"   : "ns"
    @printf "  Волна %d: ΔAIC(int→frac)=%+.2f  ΔBIC=%+.2f  LRT Λ=%.3f  p=%.4f %s\n" r.wave r.ΔAIC_int_vs_frac r.ΔBIC_int_vs_frac r.LRT_Lambda r.LRT_pval sig
end

println("\n── Сравнение main7 (ODE) vs main8-frac (GL) ──")
for r in eachrow(df_cmp)
    @printf "  Волна %d: ΔAIC=%+.2f  ΔBIC=%+.2f\n" r.wave r.ΔAIC_7_vs_8f r.ΔBIC_7_vs_8f
end

# Агрегированные итоги
println("\n── Агрегат по всем волнам ──")
@printf "  mean R²  : main7=%.4f  int_GL=%.4f  frac_GL=%.4f\n" mean(df_cmp.r2_7) mean(df_cmp.r2_8i) mean(df_cmp.r2_8f)
@printf "  mean RMSE: main7=%.3f  int_GL=%.3f  frac_GL=%.3f\n" mean(df_cmp.rmse_7) mean(df_cmp.rmse_8i) mean(df_cmp.rmse_8f)
@printf "  mean MAPE: main7=%.1f%%  int_GL=%.1f%%  frac_GL=%.1f%%\n" mean(df_cmp.mape_7) mean(df_cmp.mape_8i) mean(df_cmp.mape_8f)
n_sig = sum(df_cmp.LRT_pval .< 0.05)
@printf "  LRT p<0.05: %d из %d волн → дробная модель статистически значима в %d случаях\n" n_sig length(all_waves) n_sig

# ─────────────────────────────────────────────────────────────
# 4. Анализ остатков
# ─────────────────────────────────────────────────────────────

# ACF первых 14 лагов остатков log-шкалы
function acf(x, max_lag)
    x̄ = mean(x); n = length(x)
    var0 = sum((x .- x̄).^2) / n
    [sum((x[1:n-l] .- x̄) .* (x[1+l:n] .- x̄)) / (n * var0) for l in 1:max_lag]
end

df_acf_rows = []
for w in all_waves
    obs = df7[df7.wave .== w, :obs_cases]
    f7  = df7[df7.wave .== w, :fit_cases]
    f8f = df8[df8.wave .== w, :fit_frac]
    n   = min(length(obs), length(f8f))

    r7  = log1p.(obs[1:n]) .- log1p.(max.(f7[1:n],  0.0))
    r8f = log1p.(obs[1:n]) .- log1p.(max.(f8f[1:n], 0.0))

    a7  = acf(r7,  14)
    a8f = acf(r8f, 14)

    push!(df_acf_rows, (wave=w,
        acf1_7=a7[1],  acf7_7=a7[7],
        acf1_8f=a8f[1], acf7_8f=a8f[7],
        mean_abs_acf_7=mean(abs.(a7)), mean_abs_acf_8f=mean(abs.(a8f)),
    ))
end
df_acf = DataFrame(df_acf_rows)

println("\n── Автокорреляция остатков (меньше → лучше) ──")
println("  (ACF[1]: корреляция с лагом 1 день; ACF[7]: с лагом 7 дней)")
for r in eachrow(df_acf)
    @printf "  Волна %d: main7 ACF[1]=%.3f ACF[7]=%.3f | frac ACF[1]=%.3f ACF[7]=%.3f\n" r.wave r.acf1_7 r.acf7_7 r.acf1_8f r.acf7_8f
end

# ─────────────────────────────────────────────────────────────
# 5. Графики
# ─────────────────────────────────────────────────────────────

mkpath("./figures/covid")

# ── 5А. Scatter obs vs fit (все волны, все модели) ──────────
scatter_panels = map(all_waves) do w
    obs = df7[df7.wave .== w, :obs_cases]
    f7  = df7[df7.wave .== w, :fit_cases]
    f8f = df8[df8.wave .== w, :fit_frac]
    n   = min(length(obs), length(f8f))
    obs = obs[1:n]; f7 = f7[1:n]; f8f = f8f[1:n]
    lim = maximum(obs) * 1.1

    p = scatter(obs, f7; label="main7", color=:orange, alpha=0.5,
                markersize=3,
                xlabel="obs", ylabel="fit",
                title="Волна $w: obs vs fit", titlefontsize=8,
                xlim=(0,lim), ylim=(0,lim), size=(400,400), aspect_ratio=1)
    scatter!(p, obs, f8f; label="frac", color=:crimson, alpha=0.5, markersize=3)
    plot!(p, [0,lim], [0,lim]; color=:black, lw=1, ls=:dot, label="y=x")
    p
end
panel_scatter = plot(scatter_panels...; layout=(1, length(all_waves)),
                     size=(450*length(all_waves), 430))
savefig(panel_scatter, "./figures/covid/cmp_scatter_obs_vs_fit.png")
savefig(panel_scatter, "./figures/covid/cmp_scatter_obs_vs_fit.svg")
println("\n→ cmp_scatter_obs_vs_fit")

# ── 5Б. Остатки во времени ──────────────────────────────────
resid_panels = map(all_waves) do w
    dates = df7[df7.wave .== w, :date]
    obs   = df7[df7.wave .== w, :obs_cases]
    f7    = df7[df7.wave .== w, :fit_cases]
    f8i   = df8[df8.wave .== w, :fit_int]
    f8f   = df8[df8.wave .== w, :fit_frac]
    n     = min(length(obs), length(f8f))
    dates = dates[1:n]; obs=obs[1:n]; f7=f7[1:n]; f8i=f8i[1:n]; f8f=f8f[1:n]

    r7  = log1p.(obs) .- log1p.(max.(f7,  0.0))
    r8i = log1p.(obs) .- log1p.(max.(f8i, 0.0))
    r8f = log1p.(obs) .- log1p.(max.(f8f, 0.0))

    p = plot(dates, r7;  label="main7",   color=:orange, lw=1.5, alpha=0.8,
             title="Остатки (log), волна $w", titlefontsize=8,
             xlabel="", ylabel="log(obs)-log(fit)", legend=:topright)
    plot!(p, dates, r8i; label="int GL", color=:steelblue, lw=1.5, ls=:dash)
    plot!(p, dates, r8f; label="frac GL", color=:crimson, lw=2)
    hline!(p, [0.0]; color=:black, lw=1, ls=:dot, label="")
    p
end
panel_resid = plot(resid_panels...; layout=(length(all_waves),1),
                   size=(1100, 300*length(all_waves)))
savefig(panel_resid, "./figures/covid/cmp_residuals.png")
savefig(panel_resid, "./figures/covid/cmp_residuals.svg")
println("→ cmp_residuals")

# ── 5В. ACF остатков по волнам ──────────────────────────────
acf_panels = map(all_waves) do w
    obs = df7[df7.wave .== w, :obs_cases]
    f7  = df7[df7.wave .== w, :fit_cases]
    f8f = df8[df8.wave .== w, :fit_frac]
    n   = min(length(obs), length(f8f))
    r7  = log1p.(obs[1:n]) .- log1p.(max.(f7[1:n],  0.0))
    r8f = log1p.(obs[1:n]) .- log1p.(max.(f8f[1:n], 0.0))
    lags_acf = 1:min(21, n÷2)
    a7  = acf(r7,  maximum(lags_acf))
    a8f = acf(r8f, maximum(lags_acf))
    ci  = 1.96 / sqrt(n)   # 95% доверительный интервал для белого шума

    p = bar(lags_acf, a7; alpha=0.5, color=:orange, label="main7",
            title="ACF остатков, волна $w", titlefontsize=8,
            xlabel="Лаг (дни)", ylabel="ACF",
            ylim=(-1,1), legend=:topright)
    bar!(p, lags_acf, a8f; alpha=0.5, color=:crimson, label="frac GL")
    hline!(p, [ ci, -ci]; color=:black, lw=1, ls=:dash, label="±1.96/√n")
    p
end
panel_acf = plot(acf_panels...; layout=(length(all_waves),1),
                 size=(900, 300*length(all_waves)))
savefig(panel_acf, "./figures/covid/cmp_acf_residuals.png")
savefig(panel_acf, "./figures/covid/cmp_acf_residuals.svg")
println("→ cmp_acf_residuals")

# ── 5Г. R² по волнам: столбики трёх моделей ─────────────────
p_r2 = StatsPlots.groupedbar(
    hcat(df_cmp.r2_7, df_cmp.r2_8i, df_cmp.r2_8f);
    bar_position = :dodge,
    label=["main7 (ODE)" "int GL" "frac GL"],
    color=[:orange :steelblue :crimson], alpha=0.85,
    xlabel="Волна", ylabel="R²",
    title="Коэффициент детерминации R² по волнам",
    xticks=(1:length(all_waves), string.(df_cmp.wave)),
    ylim=(0, 1.05), size=(800, 420))
savefig(p_r2, "./figures/covid/cmp_R2_by_wave.png")
savefig(p_r2, "./figures/covid/cmp_R2_by_wave.svg")
println("→ cmp_R2_by_wave")

# ── 5Д. ΔAIC: int GL vs frac GL (LRT region) ────────────────
p_aic = bar(df_cmp.wave, df_cmp.ΔAIC_int_vs_frac;
            color = ifelse.(df_cmp.LRT_pval .< 0.05, :crimson, :gray60),
            alpha=0.85, legend=false,
            xlabel="Волна", ylabel="ΔAIC (int→frac)",
            title="ΔAIC: integer GL vs fractional GL\n(красный → p<0.05, дробная модель значимо лучше)",
            size=(800,420))
hline!(p_aic, [2.0, 10.0]; color=[:orange :red], lw=1.5, ls=:dash,
       label=["ΔAIC=2" "ΔAIC=10"])
savefig(p_aic, "./figures/covid/cmp_ΔAIC.png")
savefig(p_aic, "./figures/covid/cmp_ΔAIC.svg")
println("→ cmp_ΔAIC")

# ── 5Е. Q-Q остатков (main7 vs frac) — первая волна ─────────
qq_panels = map(all_waves) do w
    obs = df7[df7.wave .== w, :obs_cases]
    f7  = df7[df7.wave .== w, :fit_cases]
    f8f = df8[df8.wave .== w, :fit_frac]
    n   = min(length(obs), length(f8f))
    r7  = sort(log1p.(obs[1:n]) .- log1p.(max.(f7[1:n],  0.0)))
    r8f = sort(log1p.(obs[1:n]) .- log1p.(max.(f8f[1:n], 0.0)))

    # Теоретические квантили N(0,1) масштабируем на σ
    q = [quantile(randn(100_000), (i-0.5)/n) for i in 1:n]

    p = scatter(q, r7; label="main7", color=:orange, alpha=0.5, markersize=3,
                title="Q-Q остатков, волна $w", titlefontsize=8,
                xlabel="теор. квантили", ylabel="эмпир. квантили")
    scatter!(p, q, r8f; label="frac GL", color=:crimson, alpha=0.5, markersize=3)
    plot!(p, [minimum(q), maximum(q)], [minimum(q), maximum(q)];
          color=:black, lw=1, ls=:dot, label="y=x")
    p
end
panel_qq = plot(qq_panels...; layout=(1, length(all_waves)),
                size=(430*length(all_waves), 420))
savefig(panel_qq, "./figures/covid/cmp_qq_residuals.png")
savefig(panel_qq, "./figures/covid/cmp_qq_residuals.svg")
println("→ cmp_qq_residuals")

# ── 5Ж. Сводный радар-диаграмм: нет в Plots, делаем heatmap метрик
metric_names = ["R²", "1-MAPE/100", "1-logMSE*10"]
n_w = length(all_waves)
heat_data = zeros(3, n_w)
for (j, r) in enumerate(eachrow(df_cmp))
    heat_data[1, j] = r.r2_8f - r.r2_7          # R²: frac vs main7
    heat_data[2, j] = (r.mape_7 - r.mape_8f) / 100   # MAPE: main7 - frac
    heat_data[3, j] = (r.lmse_7 - r.lmse_8f) * 10    # logMSE: main7 - frac
end
p_heat = heatmap(string.(df_cmp.wave), metric_names, heat_data;
                 color=:RdYlGn, clim=(-0.1, 0.1),
                 title="Преимущество frac GL над main7\n(зелёный → frac лучше)",
                 xlabel="Волна", size=(600, 300))
savefig(p_heat, "./figures/covid/cmp_advantage_heatmap.png")
savefig(p_heat, "./figures/covid/cmp_advantage_heatmap.svg")
println("→ cmp_advantage_heatmap")

# ─────────────────────────────────────────────────────────────
# 6. Детальное сравнение по каждой волне
#
# Для каждой волны генерируется отдельный файл:
#   cmp_wave_N_detail.png  (6-панельный дашборд)
#
# Панели:
#   A. Подгонка (линейная): obs + main7 + int_GL + frac_GL
#   B. Подгонка (log):      те же + полосы ±RMSE
#   C. Остатки во времени:  r_t = log1p(obs) - log1p(fit)
#   D. ACF остатков:        3 модели + граница белого шума
#   E. Q-Q остатков:        main7 vs frac_GL
#   F. Текстовая сводка метрик (через annotate на пустой оси)
# ─────────────────────────────────────────────────────────────

println("\n── Детальные панели по волнам ──")

for w in all_waves
    # ── Данные ──────────────────────────────────────────────
    row    = df_cmp[df_cmp.wave .== w, :][1, :]
    pm_row = pm8[pm8.wave .== w, :][1, :]

    dates  = df7[df7.wave .== w, :date]
    obs    = df7[df7.wave .== w, :obs_cases]
    f7     = df7[df7.wave .== w, :fit_cases]
    f8i    = df8[df8.wave .== w, :fit_int]
    f8f    = df8[df8.wave .== w, :fit_frac]
    n      = min(length(obs), length(f8f))
    dates  = dates[1:n]; obs=obs[1:n]
    f7=f7[1:n]; f8i=f8i[1:n]; f8f=f8f[1:n]

    # Остатки (log-шкала)
    r7   = log1p.(obs) .- log1p.(max.(f7,  0.0))
    r8i  = log1p.(obs) .- log1p.(max.(f8i, 0.0))
    r8f  = log1p.(obs) .- log1p.(max.(f8f, 0.0))

    ci_band = 1.96 / sqrt(n)
    sig_str = row.LRT_pval < 0.001 ? "***" : row.LRT_pval < 0.01 ? "**" :
              row.LRT_pval < 0.05  ? "*"   : "ns"
    strain_str = pm_row.strain

    # ── Панель A: подгонка линейная ──────────────────────────
    ylim_top = maximum(obs) * 1.2
    pA = plot(dates, obs;
              label="наблюдения", color=:steelblue, lw=2, alpha=0.85,
              ylabel="случаев / день", title="A) Подгонка (линейная)",
              titlefontsize=9, legend=:topright,
              ylim=(0, ylim_top), grid=true, gridalpha=0.3)
    plot!(pA, dates, f7;   label="main7 (ODE)", color=:orange,    lw=1.8, ls=:dash)
    plot!(pA, dates, f8i;  label="int GL (α=1)", color=:steelblue, lw=1.5, ls=:dot)
    plot!(pA, dates, f8f;  label="frac GL",      color=:crimson,   lw=2.5)

    # ── Панель B: log-шкала + полосы ±RMSE ──────────────────
    # Полоса доверия ±RMSE вокруг frac-подгонки
    log_f8f = log1p.(max.(f8f, 0.0))
    pB = plot(dates, log1p.(obs);
              label="наблюдения", color=:steelblue, lw=2, alpha=0.85,
              ylabel="log(случаев + 1)", title="B) Подгонка (log-шкала)",
              titlefontsize=9, legend=:topright, grid=true, gridalpha=0.3)
    plot!(pB, dates, log1p.(max.(f7,  0.0)); label="main7",  color=:orange, lw=1.5, ls=:dash)
    plot!(pB, dates, log1p.(max.(f8f, 0.0)); label="frac GL",color=:crimson, lw=2.5)
    # Полоса ±RMSE(log) вокруг frac
    rmse_log_f = sqrt(row.lmse_8f)
    plot!(pB, dates, log_f8f .- rmse_log_f; fillrange=log_f8f .+ rmse_log_f,
          fillalpha=0.12, color=:crimson, lw=0, label="±RMSE_log (frac)")

    # ── Панель C: остатки во времени ─────────────────────────
    pC = plot(dates, r7;
              label="main7",   color=:orange,    lw=1.5, alpha=0.8,
              ylabel="log(obs) − log(fit)", title="C) Остатки (log-шкала)",
              titlefontsize=9, legend=:topright, grid=true, gridalpha=0.3)
    plot!(pC, dates, r8i; label="int GL",  color=:steelblue, lw=1.5, ls=:dot)
    plot!(pC, dates, r8f; label="frac GL", color=:crimson,   lw=2)
    hline!(pC, [0.0];     color=:black, lw=1, ls=:dash, label="")
    # Полоса белого шума ±1.96/√n
    hline!(pC, [ci_band, -ci_band]; color=:gray50, lw=1, ls=:dot, label="±1.96/√n")

    # ── Панель D: ACF остатков ───────────────────────────────
    max_lag  = min(21, n÷2)
    lags_plt = 1:max_lag
    a7_w  = acf(r7,  max_lag)
    a8i_w = acf(r8i, max_lag)
    a8f_w = acf(r8f, max_lag)

    pD = bar(lags_plt .- 0.25, a7_w;
             bar_width=0.22, alpha=0.75, color=:orange, label="main7",
             ylabel="ACF", title="D) Автокорреляция остатков",
             titlefontsize=9, ylim=(-1,1), legend=:topright)
    bar!(pD, lags_plt,        a8i_w;
         bar_width=0.22, alpha=0.75, color=:steelblue, label="int GL")
    bar!(pD, lags_plt .+ 0.25, a8f_w;
         bar_width=0.22, alpha=0.75, color=:crimson, label="frac GL")
    hline!(pD, [ci_band, -ci_band]; color=:black, lw=1.2, ls=:dash, label="±1.96/√n")
    xlabel!(pD, "Лаг (дни)")

    # ── Панель E: Q-Q ────────────────────────────────────────
    q_theory = [quantile(randn(200_000), (i-0.5)/n) for i in 1:n]
    pE = scatter(q_theory, sort(r7);
                 label="main7",  color=:orange,  alpha=0.55, markersize=3,
                 xlabel="теор. квантили N(0,σ)", ylabel="эмпир. квантили",
                 title="E) Q-Q остатков", titlefontsize=9, legend=:topleft)
    scatter!(pE, q_theory, sort(r8f);
             label="frac GL", color=:crimson, alpha=0.55, markersize=3)
    lim_qq = max(maximum(abs.(r7)), maximum(abs.(r8f)), maximum(abs.(q_theory))) * 1.05
    plot!(pE, [-lim_qq, lim_qq], [-lim_qq, lim_qq];
          color=:black, lw=1, ls=:dot, label="y=x")

    # ── Панель F: текстовая сводка ───────────────────────────
    # Рисуем невидимую ось с annotate
    pF = plot(; axis=false, grid=false, legend=false,
              title="F) Метрики волны $w", titlefontsize=9,
              xlim=(0,1), ylim=(0,1))

    txt = """
Волна:  $w   Штамм: $(strain_str)
n = $n дней   α_k = $(round(row.α_k, digits=4))

         main7      int_GL    frac_GL
R²    $(lpad(round(row.r2_7,   digits=3),7))  $(lpad(round(row.r2_8i,  digits=3),7))  $(lpad(round(row.r2_8f,  digits=3),7))
RMSE  $(lpad(round(row.rmse_7, digits=2),7))  $(lpad(round(row.rmse_8i,digits=2),7))  $(lpad(round(row.rmse_8f,digits=2),7))
MAPE  $(lpad(round(row.mape_7, digits=1),6))%  $(lpad(round(row.mape_8i,digits=1),6))%  $(lpad(round(row.mape_8f,digits=1),6))%
logMSE $(lpad(round(row.lmse_7,digits=4),7))  $(lpad(round(row.lmse_8i,digits=4),7))  $(lpad(round(row.lmse_8f,digits=4),7))
AIC   $(lpad(round(row.aic_7,  digits=1),7))  $(lpad(round(row.aic_8i, digits=1),7))  $(lpad(round(row.aic_8f, digits=1),7))

LRT:  Λ=$(round(row.LRT_Lambda,digits=3))   p=$(round(row.LRT_pval,sigdigits=3)) $(sig_str)
ΔAIC (int→frac) = $(round(row.ΔAIC_int_vs_frac, digits=2))
ΔAIC (main7→frac) = $(round(row.ΔAIC_7_vs_8f,   digits=2))

ACF[1]: m7=$(round(a7_w[1],digits=3))  frac=$(round(a8f_w[1],digits=3))
    """
    annotate!(pF, 0.05, 0.95, text(txt, :left, 10))

    # ── Сборка 2×3 панели ───────────────────────────────────
    wave_panel = plot(pA, pB, pC, pD, pE, pF;
                      layout=(2, 3),
                      size=(1920, 1080),
                      plot_title="Волна $w: $(strain_str)   α_k=$(round(row.α_k,digits=3))",
                      plot_titlefontsize=11)

    fname = "./figures/covid/cmp_wave_$(w)_detail"
    savefig(wave_panel, fname * ".png")
    savefig(wave_panel, fname * ".svg")
    println("  → cmp_wave_$(w)_detail.png")
end

# ─────────────────────────────────────────────────────────────
# 6. Сохранение
# ─────────────────────────────────────────────────────────────

# Полная таблица сравнения
CSV.write("./data_out/model_comparison.csv", df_cmp)

# Краткая публикационная таблица
df_pub = select(df_cmp,
    :wave, :n, :α_k,
    :r2_7   => :R2_main7,
    :r2_8f  => :R2_frac,
    :rmse_7 => :RMSE_main7,
    :rmse_8f => :RMSE_frac,
    :mape_7  => :MAPE_main7,
    :mape_8f => :MAPE_frac,
    :ΔAIC_int_vs_frac => :ΔAIC,
    :ΔBIC_int_vs_frac => :ΔBIC,
    :LRT_Lambda => :LRT_stat,
    :LRT_pval   => :LRT_pval,
)
# Метки значимости
df_pub.sig = map(p -> p < 0.001 ? "***" : p < 0.01 ? "**" :
                      p < 0.05  ? "*"   : "ns", df_pub.LRT_pval)
CSV.write("./data_out/model_comparison_pub.csv", df_pub)

println("\n=== Сохранено ===")
println("  ./data_out/model_comparison.csv")
println("  ./data_out/model_comparison_pub.csv")

println("\n=== Публикационная таблица ===")
println(df_pub)
println("\nГотово ✓")