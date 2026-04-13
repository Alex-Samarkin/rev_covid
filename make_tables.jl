# ============================================================
# make_tables.jl — генерация Markdown-таблиц для статьи
#
# Запуск:  julia make_tables.jl  (из корня проекта)
# Вывод:   ./data_out/tables_output.md  +  stdout
#
# Точные схемы столбцов CSV:
#
#   seird_all_waves_params.csv (main7):
#     wave, strain, T_days, peak_obs, peak_fit, peak_ratio,
#     R0_prior, R0_fit, β_prior, β_fit, σ_fix, γ_fix,
#     μ_prior, μ_fit, N_eff, N_eff_pct, loss
#
#   fseird_params.csv (main8):
#     wave, strain, T_days, peak_obs, peak_int, peak_frac,
#     R0_prior, R0_int, R0_frac, α_k,
#     N_eff_int, N_eff_frac, N_int_pct, N_frac_pct,
#     loss_int, loss_frac, improve_pct
#
#   model_comparison.csv (main9):
#     wave, n, α_k,
#     rmse_7, mae_7, mape_7, r2_7, lmse_7, aic_7, bic_7,
#     rmse_8i, mae_8i, mape_8i, r2_8i, lmse_8i, aic_8i, bic_8i,
#     rmse_8f, mae_8f, mape_8f, r2_8f, lmse_8f, aic_8f, bic_8f,
#     LRT_Lambda, LRT_pval,
#     ΔAIC_int_vs_frac, ΔBIC_int_vs_frac, ΔAIC_7_vs_8f, ΔBIC_7_vs_8f
#
#   seird_all_waves_trajectories.csv (main7):
#     wave, strain, date, obs_cases, obs_deaths,
#     fit_cases, fit_deaths, S, E, I, R, D
# ============================================================

using CSV, DataFrames, Dates, Printf, Statistics

fmt(x::AbstractFloat, d) = @sprintf("%.*f", d, x)
fmt(x::Integer, _)       = string(x)
fmt(x::Missing, _)       = "—"
fmt(x, _)               = string(x)

md_row(cells) = "| " * join(string.(cells), " | ") * " |"

function md_table(headers, rows)
    sep = "| " * join(fill(":---:", length(headers)), " | ") * " |"
    lines = [md_row(headers), sep]
    append!(lines, md_row.(rows))
    join(lines, "\n")
end

# ── Загрузка ──────────────────────────────────────────────────

traj = CSV.read("./data_out/seird_all_waves_trajectories.csv", DataFrame)
pm7  = CSV.read("./data_out/seird_all_waves_params.csv",       DataFrame)
pm8  = CSV.read("./data_out/fseird_params.csv",                DataFrame)
cmp  = CSV.read("./data_out/model_comparison.csv",             DataFrame)

all_waves = sort(unique(pm7.wave))
println("Волны: $(all_waves)  (всего: $(length(all_waves)))")

# ── ТАБЛИЦА 1 ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ТАБЛИЦА 1 — Характеристика идентифицированных волн")
println("="^70)

t1h = ["Волна", "Дата начала", "Дата окончания", "Штамм",
       "T, сут", "Пик, случ./д", "Σ случаев", "Σ смертей"]

t1r = map(all_waves) do w
    sub    = sort(filter(:wave => ==(w), traj), :date)
    strain = pm8[pm8.wave .== w, :strain][1]
    [w,
     Dates.format(minimum(sub.date), "dd.mm.yyyy"),
     Dates.format(maximum(sub.date), "dd.mm.yyyy"),
     strain, nrow(sub),
     fmt(maximum(sub.obs_cases), 1),
     string(round(Int, sum(sub.obs_cases))),
     string(round(Int, sum(sub.obs_deaths)))]
end
println(md_table(t1h, t1r))

# ── ТАБЛИЦА 2 ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ТАБЛИЦА 2 — Априорные параметры штаммов")
println("="^70)

t2h = ["Волна", "Штамм", "σ, сут⁻¹", "γ, сут⁻¹",
       "T_incub, сут", "T_infect, сут", "β_prior", "μ_prior", "R₀_prior"]

t2r = map(eachrow(pm7)) do r
    [r.wave, r.strain,
     fmt(r.σ_fix, 5), fmt(r.γ_fix, 5),
     fmt(round(1.0/r.σ_fix, digits=1), 1),
     fmt(round(1.0/r.γ_fix, digits=1), 1),
     fmt(r.β_prior, 5), fmt(r.μ_prior, 6), fmt(r.R0_prior, 2)]
end
println(md_table(t2h, t2r))

# ── ТАБЛИЦА 3 ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ТАБЛИЦА 3 — Подобранные параметры целочисленной SEIRD (main7)")
println("="^70)

t3h = ["Волна", "β_fit", "Δβ, %", "R₀_fit", "R₀_prior",
       "μ_fit", "μ_prior", "N_eff", "N_eff, %", "loss"]

t3r = map(eachrow(pm7)) do r
    Δβ = 100.0 * (r.β_fit - r.β_prior) / max(r.β_prior, 1e-12)
    [r.wave, fmt(r.β_fit,5), fmt(Δβ,1),
     fmt(r.R0_fit,2), fmt(r.R0_prior,2),
     fmt(r.μ_fit,6), fmt(r.μ_prior,6),
     fmt(r.N_eff,0), fmt(r.N_eff_pct,1), fmt(r.loss,5)]
end
println(md_table(t3h, t3r))

# ── ТАБЛИЦА 4 ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ТАБЛИЦА 4 — Метрики качества целочисленной SEIRD (main7)")
println("="^70)

t4h = ["Волна", "T, сут", "RMSE", "MAE", "MAPE, %", "R²", "log-MSE"]

t4r = map(eachrow(cmp)) do r
    T_days = pm7[pm7.wave .== r.wave, :T_days][1]
    [r.wave, T_days,
     fmt(r.rmse_7,2), fmt(r.mae_7,2), fmt(r.mape_7,1),
     fmt(r.r2_7,4), fmt(r.lmse_7,5)]
end
println(md_table(t4h, t4r))

# ── ТАБЛИЦА 5 ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ТАБЛИЦА 5 — Параметры дробной SEIRD (main8) и α_k по волнам")
println("="^70)

t5h = ["Волна", "Штамм", "α_k",
       "R₀_int", "R₀_frac", "ΔR₀, %",
       "N_eff_int", "N_eff_frac", "N_int, %", "N_frac, %",
       "loss_int", "loss_frac", "Δloss, %"]

t5r = map(eachrow(pm8)) do r
    ΔR0 = 100.0 * (r.R0_frac - r.R0_int) / max(r.R0_int, 1e-12)
    [r.wave, r.strain, fmt(r.α_k,4),
     fmt(r.R0_int,2), fmt(r.R0_frac,2), fmt(ΔR0,1),
     fmt(r.N_eff_int,0), fmt(r.N_eff_frac,0),
     fmt(r.N_int_pct,1), fmt(r.N_frac_pct,1),
     fmt(r.loss_int,5), fmt(r.loss_frac,5), fmt(r.improve_pct,1)]
end
println(md_table(t5h, t5r))

# ── ТАБЛИЦА 6 ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ТАБЛИЦА 6 — Формальное сравнение целочисленной и дробной SEIRD")
println("="^70)

t6h = ["Волна", "n", "α_k", "R²_int", "R²_frac",
       "RMSE_int", "RMSE_frac", "ΔAIC", "ΔBIC", "Λ", "p", "Знч."]

sig(p) = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns"

t6r = map(eachrow(cmp)) do r
    [r.wave, r.n, fmt(r.α_k,4),
     fmt(r.r2_8i,4), fmt(r.r2_8f,4),
     fmt(r.rmse_8i,2), fmt(r.rmse_8f,2),
     fmt(r.ΔAIC_int_vs_frac,1), fmt(r.ΔBIC_int_vs_frac,1),
     fmt(r.LRT_Lambda,2), fmt(r.LRT_pval,4), sig(r.LRT_pval)]
end
println(md_table(t6h, t6r))

# ── Числа для текста ──────────────────────────────────────────

println("\n" * "="^70)
println("ЧИСЛА ДЛЯ ТЕКСТА")
println("="^70)

total_c = sum(traj.obs_cases);  total_d = sum(traj.obs_deaths)
T_vals  = pm7.T_days
@printf "Σ случаев: %.0f | Σ смертей: %.0f\n" total_c total_d
@printf "T_days: min=%d  max=%d  median=%.0f\n" minimum(T_vals) maximum(T_vals) median(T_vals)

early = sum(sum(filter(:wave => ==(w), traj).obs_cases) for w in 1:3)
@printf "Доля случаев волн 1–3: %.1f%%\n" 100*early/total_c

mx = cmp[argmax(cmp.ΔAIC_int_vs_frac), :]
@printf "Макс. ΔAIC: %.1f (волна %d) | Макс. ΔBIC: %.1f\n" mx.ΔAIC_int_vs_frac mx.wave maximum(cmp.ΔBIC_int_vs_frac)
println("Значимые p<0.05:       $(cmp[cmp.LRT_pval .< 0.05,  :wave])")
println("Значимые p<0.005 (Bonf):$(cmp[cmp.LRT_pval .< 0.005, :wave])")

late = filter(r -> r.wave >= 7, pm8)
@printf "Среднее Δloss%% волны 7–10: %.1f%%\n" mean(late.improve_pct)

wv = Float64.(pm8.wave); av = Float64.(pm8.α_k)
b̂  = cov(wv,av)/var(wv);  â = mean(av) - b̂*mean(wv)
R2 = 1 - sum((av.-(â.+b̂.*wv)).^2)/sum((av.-mean(av)).^2)
@printf "Тренд α_k: â=%.4f  b̂=%.4f  R²=%.3f\n" â b̂ R2
@printf "Corr(α_k, N_frac_pct): %.3f | Corr(α_k, N_int_pct): %.3f\n" cor(pm8.α_k, pm8.N_frac_pct) cor(pm8.α_k, pm8.N_int_pct)

println("\nКумулятивные случаи до волны k:")
wt = [sum(filter(:wave => ==(w), traj).obs_cases) for w in all_waves]
for (w, c) in zip(all_waves, [0; cumsum(wt)[1:end-1]])
    @printf "  до волны %2d: %7.0f чел. (%.1f%%)\n" w c 100*c/200_000
end

# ── Запись ────────────────────────────────────────────────────

open("./data_out/tables_output.md", "w") do io
    pairs = [
        ("Таблица 1 — Характеристика идентифицированных волн заболеваемости COVID-19 в Пскове",              t1h, t1r),
        ("Таблица 2 — Априорные параметры штаммов: фиксируемые (σ, γ) и стартовые (β, μ)",                  t2h, t2r),
        ("Таблица 3 — Подобранные параметры целочисленной SEIRD-модели (main7) по волнам",                   t3h, t3r),
        ("Таблица 4 — Метрики качества подгонки целочисленной SEIRD (main7) по волнам",                      t4h, t4r),
        ("Таблица 5 — Результаты калибровки дробной SEIRD (main8): α_k и параметры по волнам",               t5h, t5r),
        ("Таблица 6 — Формальное сравнение целочисленной и дробной SEIRD-моделей (LRT, AIC, BIC)",           t6h, t6r),
    ]
    for (title, hdr, rows) in pairs
        write(io, title * "\n\n" * md_table(hdr, rows) * "\n\n")
    end
end

println("\n→ Записано: ./data_out/tables_output.md")
println("Готово ✓")