# make_figures.jl
# Генерация всех публикационных рисунков по данным из data_out/
# Требует: CSV.jl, DataFrames.jl, Plots.jl, StatsBase.jl, Statistics.jl
using Revise

using CSV, DataFrames, Plots, StatsBase, Statistics
# pgfplotsx()  # или gr() для интерактивной работы; pgfplotsx() для публикационного PDF
gr()  # для быстрого рендеринга в процессе разработки; заменить на pgfplotsx() для финальных PDF

# ─── ЗАГРУЗКА ДАННЫХ ──────────────────────────────────────────────────────────
cd(@__DIR__)  # гарантируем, что работаем из папки скрипта
df_waves     = CSV.read("./data_out/seird_all_waves_trajectories.csv", DataFrame)
df_frac_traj = CSV.read("./data_out/fseird_trajectories.csv", DataFrame)
df_main7     = CSV.read("./data_out/seird_all_waves_params.csv", DataFrame)
df_frac      = CSV.read("./data_out/fseird_params.csv", DataFrame)
df_cmp       = CSV.read("./data_out/model_comparison_pub.csv", DataFrame)
df_daily     = CSV.read("./data_out/covid_data_with_waves_and_derivatives.csv", DataFrame)
df_strain    = CSV.read("./csv/covid19_seird_params.csv", DataFrame)

using Dates

parse_date(s::Date) = s
parse_date(s::AbstractString) = Date(s, dateformat"yyyy-mm-dd")

WAVE_COLORS = [:steelblue, :royalblue, :darkorange, :firebrick, :crimson,
               :purple, :olive, :teal, :darkgreen, :sienna]
STRAIN_LABELS = ["Wuhan", "Wuhan", "Alpha", "Delta", "Delta",
                 "Omicron\nBA.2", "Omicron\nBA.4/5", "Omicron\nXBB", 
                 "Omicron\nJN.1", "Omicron\nKP.3"]

# ─── РИС. 1: ПАНОРАМА РЯДА ───────────────────────────────────────────────────
function fig1_panorama()
    df_daily.date_parsed = parse_date.(df_daily.date)
    
    p = plot(size=(1400, 500), margin=5Plots.mm, dpi=300,
             xlabel="Дата", ylabel="Случаев / день",
             title="Заболеваемость COVID-19 в г. Пскове, 2020–2024")
    
    # цветные полосы волн
    wave_df = groupby(df_daily, :wave)
    for k in 1:10
        wk = wave_df[(wave=k,)]
        vspan!(p, [minimum(wk.date_parsed), maximum(wk.date_parsed)],
               alpha=0.12, color=WAVE_COLORS[k], label="")
    end
    
    # сырой ряд
    plot!(p, df_daily.date_parsed, df_daily.daily_interp,
          color=:gray60, lw=0.7, alpha=0.6, label="daily_interp")
    # сглаженный
    plot!(p, df_daily.date_parsed, df_daily.daily_interp_smooth,
          color=:black, lw=2.0, label="Сглаж. 18 дней")
    
    # границы волн — вертикальные линии
    for k in 1:10
        wk = df_daily[df_daily.wave .== k, :]
        vline!(p, [minimum(parse_date.(wk.date))], 
               color=WAVE_COLORS[k], lw=1.5, ls=:dash, label="")
        # аннотация штамма
        annotate!(p, minimum(parse_date.(wk.date)), 
                  maximum(df_daily.daily_interp_smooth)*0.9,
                  text("$(k)", 8, :left, WAVE_COLORS[k]))
    end
    
    savefig(p, "fig1_panorama.pdf")
    savefig(p, "fig1_panorama.png")
    @info "Рисунок 1 сохранён"
    return p
end

# ─── РИС. 2: ПРОФИЛИ ВОЛН ────────────────────────────────────────────────────
function fig2_wave_profiles()
    plots_lin = []
    plots_log = []
    
    for k in 1:10
        wk = df_frac_traj[df_frac_traj.wave .== k, :]
        days = 0:(nrow(wk)-1)
        pk_day = argmax(wk.obs_cases) - 1
        
        plin = plot(days, wk.obs_cases, 
                    color=WAVE_COLORS[k], lw=1.5, label="",
                    title="Волна $k\n$(STRAIN_LABELS[k])",
                    titlefontsize=8, xtickfontsize=7, ytickfontsize=7)
        vline!(plin, [pk_day], color=:black, ls=:dot, lw=1, label="")
        push!(plots_lin, plin)
        
        plog = plot(days, max.(wk.obs_cases, 0.1), 
                    color=WAVE_COLORS[k], lw=1.5, label="",
                    yaxis=:log10, ylims=(0.1, nothing),
                    xtickfontsize=7, ytickfontsize=7)
        push!(plots_log, plog)
    end
    
    p = plot(vcat(plots_lin, plots_log)..., 
             layout=(2, 10), size=(2000, 500), dpi=200)
    
    # Альтернативный layout 4×5:
    # p = plot(vcat(plots_lin, plots_log)..., layout=(4,5), size=(1600,800), dpi=200)
    
    savefig(p, "fig2_wave_profiles.pdf")
    @info "Рисунок 2 сохранён"
    return p
end

# ─── РИС. 3: R₀ prior vs fit ─────────────────────────────────────────────────
function fig3_R0_comparison()
    waves = 1:10
    r0_prior = df_main7.R0_prior
    r0_fit   = df_main7.R0_fit
    
    p = plot(size=(800, 400), dpi=300, margin=5Plots.mm,
             xlabel="Номер волны", ylabel="R₀",
             title="Априорные и подобранные значения R₀ (целочисленная модель)",
             xticks=(1:10, ["$k\n$(STRAIN_LABELS[k])" for k in 1:10]),
             xrotation=20, legend=:topright,
             yaxis=:log10)
    
    # prior
    plot!(p, waves, r0_prior, marker=:circle, ms=8, lw=2, 
          color=:steelblue, label="R₀ prior (литература)")
    # fit
    plot!(p, waves, r0_fit, marker=:square, ms=8, lw=2,
          color=:darkorange, label="R₀ fit (целочисл.)")
    
    # вертикальные отрезки ΔR₀
    for k in waves
        plot!(p, [k, k], [r0_fit[k], r0_prior[k]], 
              color=:gray50, lw=1.5, ls=:dot, label="")
    end
    
    hline!(p, [1.0], color=:red, ls=:dash, lw=1, label="R₀ = 1")
    
    savefig(p, "fig3_R0_comparison.pdf")
    @info "Рисунок 3 сохранён"
    return p
end

# ─── РИС. 4: α_k по волнам ───────────────────────────────────────────────────
function fig4_alpha_dynamics()
    alpha_vals = df_cmp.α_k
    sig        = df_cmp.sig
    waves      = 1:10
    
    # определяем значимость
    is_sig = sig .!= "ns"
    
    p = plot(size=(800, 400), dpi=300, margin=5Plots.mm,
             xlabel="Номер волны", ylabel="α_k",
             title="Порядок дробной производной α_k по волнам",
             ylims=(0.6, 1.05),
             xticks=(1:10, ["$k\n$(STRAIN_LABELS[k])" for k in 1:10]),
             xrotation=20, legend=:bottomleft)
    
    # серый фон для поздних волн 7–10
    vspan!(p, [6.5, 10.5], alpha=0.08, color=:lightcoral, label="Поздний Омикрон")
    
    # горизонтальная линия α=1
    hline!(p, [1.0], color=:crimson, ls=:dash, lw=1.5, label="α = 1 (целочисл.)")
    
    # линия тренда
    k_vals = collect(1:10)
    b = cov(k_vals, alpha_vals) / var(k_vals)
    a = mean(alpha_vals) - b * mean(k_vals)
    x_fit = 1:0.1:10
    plot!(p, x_fit, a .+ b .* x_fit, color=:gray30, lw=1, ls=:dot, label="Линейный тренд")
    
    # незначимые точки
    scatter!(p, waves[.!is_sig], alpha_vals[.!is_sig], 
             marker=:circle, ms=10, color=:white, 
             markerstrokecolor=:gray50, markerstrokewidth=2,
             label="ns")
    # значимые точки
    scatter!(p, waves[is_sig], alpha_vals[is_sig],
             marker=:circle, ms=10, color=:crimson,
             markerstrokecolor=:black, markerstrokewidth=1,
             label="LRT p<0.05 (Bonferroni)")
    
    # звёздочки
    for k in waves
        if is_sig[k]
            stars = sig[k] == "***" ? "***" : (sig[k] == "**" ? "**" : "*")
            annotate!(p, k, alpha_vals[k]+0.02, text(stars, 9, :center, :crimson))
        end
    end
    
    savefig(p, "fig4_alpha_dynamics.pdf")
    @info "Рисунок 4 сохранён"
    return p
end

# ─── РИС. 5: Траектории I(t) для волн 2 и 10 ────────────────────────────────
function fig5_trajectories_contrast()
    for (k, col_title) in [(2, "Волна 2 (α_k = 1.00)"), (10, "Волна 10 (α_k = 0.91)")]
        wk = df_frac_traj[df_frac_traj.wave .== k, :]
        days = 0:(nrow(wk)-1)
        # ... plot obs, fit_int, fit_frac
    end
    
    w2  = df_frac_traj[df_frac_traj.wave .== 2, :]
    w10 = df_frac_traj[df_frac_traj.wave .== 10, :]
    
    d2  = 0:(nrow(w2)-1)
    d10 = 0:(nrow(w10)-1)
    
    p1 = plot(d2, w2.obs_cases, marker=:circle, ms=3, color=:gray40, label="Наблюдения",
              title="Волна 2 (α_k = 1.00)", xlabel="Дни", ylabel="Случаев/день",
              dpi=300)
    plot!(p1, d2, w2.fit_int,  lw=2, ls=:dash, color=:darkorange, label="SEIRD (GL, α=1)")
    plot!(p1, d2, w2.fit_frac, lw=2, color=:crimson, label="F-SEIRD (GL, α_k)")
    
    p2 = plot(d10, w10.obs_cases, marker=:circle, ms=3, color=:gray40, label="Наблюдения",
              title="Волна 10 (α_k = 0.906)", xlabel="Дни", ylabel="Случаев/день",
              dpi=300)
    plot!(p2, d10, w10.fit_int,  lw=2, ls=:dash, color=:darkorange, label="SEIRD (GL, α=1)")
    plot!(p2, d10, w10.fit_frac, lw=2, color=:crimson, label="F-SEIRD (GL, α_k)")
    
    # отметить пик
    pk = argmax(w10.obs_cases) - 1
    vline!(p2, [pk], color=:black, ls=:dot, lw=1, label="Пик")
    
    p = plot(p1, p2, layout=(1,2), size=(1100, 400), dpi=300, margin=5Plots.mm)
    savefig(p, "fig5_trajectories_contrast.pdf")
    @info "Рисунок 5 сохранён"
    return p
end

# ─── РИС. 6: Ядра памяти GL ──────────────────────────────────────────────────
function fig6_gl_memory_kernels()
    alpha_vals = df_cmp.α_k
    lags = 1:60
    
    function gl_weights(α, n_lags)
        w = zeros(n_lags)
        w[1] = 1.0
        for j in 2:n_lags
            w[j] = w[j-1] * (j - 1 - α) / j
        end
        return abs.(w)  # абсолютные значения
    end
    
    p = plot(size=(800, 450), dpi=300, margin=5Plots.mm,
             xlabel="Лаг, дней", ylabel="|w_j(α)|",
             title="Ядра памяти GL для значений α_k (10 волн)",
             yaxis=:log10, ylims=(1e-6, 1.5), legend=:topright)
    
    for k in 1:10
        α = alpha_vals[k]
        w = gl_weights(α, 60)
        lw_k = k in [3,7,8,9,10] ? 2.5 : 1.2
        plot!(p, lags, w, lw=lw_k, color=WAVE_COLORS[k],
              label="Волна $k (α=$(round(α,digits=3)))")
    end
    
    vline!(p, [14], color=:black, ls=:dash, lw=1.5, label="14 дней")
    vline!(p, [30], color=:gray50, ls=:dot, lw=1.5, label="30 дней")
    
    savefig(p, "fig6_gl_kernels.pdf")
    @info "Рисунок 6 сохранён"
    return p
end

# ─── РИС. 7: ΔAIC / ΔBIC по волнам ──────────────────────────────────────────
function fig7_aic_bic()
    waves = 1:10
    Δaic  = df_cmp.ΔAIC
    Δbic  = df_cmp.ΔBIC
    sig   = df_cmp.sig
    
    # ограничим отображение для читаемости: вырежем max до 300
    display_aic = clamp.(Δaic, -20, 300)
    display_bic = clamp.(Δbic, -20, 300)
    
    bar_w = 0.35
    p = bar(waves .- bar_w/2, display_aic, bar_width=bar_w,
            color=[v >= 0 ? :gray60 : :white for v in Δaic],
            label="ΔAIC", size=(900, 450), dpi=300, margin=5Plots.mm,
            xlabel="Волна", ylabel="ΔAIC / ΔBIC",
            title="Информационные критерии: дробная vs. целочисленная SEIRD",
            xticks=(1:10, string.(1:10)), legend=:topleft)
    bar!(p, waves .+ bar_w/2, display_bic, bar_width=bar_w,
         color=[:white for _ in waves], linecolor=:black,
         label="ΔBIC")
    
    hline!(p, [0.0], color=:black, lw=1, label="")
    hline!(p, [10.0], color=:steelblue, ls=:dash, lw=1.5, label="ΔAIC = 10")
    
    # аннотация для волн с ΔAIC > 280 (обрезанных)
    for k in waves
        if Δaic[k] > 300
            annotate!(p, k - bar_w/2, 305, text("$(round(Δaic[k], digits=0))", 8, :center))
        end
        if sig[k] != "ns"
            stars = sig[k]
            annotate!(p, k, max(display_aic[k], display_bic[k]) + 8, 
                      text(stars, 10, :center, :red))
        end
    end
    
    savefig(p, "fig7_aic_bic.pdf")
    @info "Рисунок 7 сохранён"
    return p
end

# ─── РИС. ДОППОЛ. E: N_eff по волнам ─────────────────────────────────────────
function figE_neff_dynamics()
    waves = 1:10
    neff_int  = df_frac.N_int_pct
    neff_frac = df_frac.N_frac_pct
    
    bar_w = 0.35
    p = bar(waves .- bar_w/2, neff_int, bar_width=bar_w,
            color=:steelblue, alpha=0.8, label="N_eff (целочисл.)",
            size=(800, 400), dpi=300, margin=5Plots.mm,
            xlabel="Волна", ylabel="N_eff / N_city, %",
            title="Эффективная восприимчивая популяция по волнам",
            xticks=(1:10, ["$k\n$(STRAIN_LABELS[k])" for k in 1:10]),
            xrotation=20)
    bar!(p, waves .+ bar_w/2, neff_frac, bar_width=bar_w,
         color=:crimson, alpha=0.8, label="N_eff (дробная)")
    
    # аннотация иммунного уклона на волне 4 (BA.1)
    annotate!(p, 4, neff_int[4] + 5, 
              text("← Иммунный\n   уклон BA.1", 7, :center, :purple))
    
    savefig(p, "figE_neff.pdf")
    @info "Рисунок E сохранён"
    return p
end

# ─── РИС. ДОПОЛН. B: Подгонка всех волн ─────────────────────────────────────
function figB_all_fits()
    subplots = []
    for k in 1:10
        wk = df_frac_traj[df_frac_traj.wave .== k, :]
        days = 0:(nrow(wk)-1)
        
        α_k = df_cmp.α_k[k]
        ΔAIC_k = round(df_cmp.ΔAIC[k], digits=1)
        sig_k = df_cmp.sig[k]
        
        title_str = "W$(k) | α=$(round(α_k, digits=3)) | ΔAIC=$(ΔAIC_k) $(sig_k != "ns" ? sig_k : "")"
        
        pk = plot(days, wk.obs_cases, marker=:circle, ms=2, 
                  color=:gray40, label="",
                  title=title_str, titlefontsize=7,
                  xtickfontsize=6, ytickfontsize=6,
                  dpi=150)
        plot!(pk, days, wk.fit_int,  lw=1.5, ls=:dash, color=:darkorange, label="")
        plot!(pk, days, wk.fit_frac, lw=1.5, color=:crimson, label="")
        
        # цветная рамка для значимых волн
        if sig_k != "ns"
            plot!(pk, framestyle=:box, foreground_color_border=:crimson)
        end
        
        push!(subplots, pk)
    end
    
    p = plot(subplots..., layout=(2, 5), size=(1800, 700), dpi=150, 
             margin=2Plots.mm)
    savefig(p, "figB_all_fits.pdf")
    @info "Рисунок B (все волны) сохранён"
    return p
end

# ─── ГЛАВНЫЙ ЗАПУСК ──────────────────────────────────────────────────────────
mkpath("figures_out")
cd("figures_out")

fig1_panorama()
fig2_wave_profiles()
fig3_R0_comparison()
fig4_alpha_dynamics()
fig5_trajectories_contrast()
fig6_gl_memory_kernels()
fig7_aic_bic()
figE_neff_dynamics()
figB_all_fits()

@info "Все рисунки сохранены в ./figures_out/"
