# fig9_alpha_context.jl
# Рисунок 9: α_k vs иммунологический контекст (3 субграфика)
# Источники: fseird_params.csv, model_comparison_pub.csv,
#            seird_all_waves_trajectories.csv

using Revise
using CSV, DataFrames, Plots, Statistics, StatsBase, Printf
gr()   # для PDF-публикации; замените на gr() для интерактивного просмотра

# ── ЗАГРУЗКА ─────────────────────────────────────────────────────────────────
cd(@__DIR__)  # гарантируем, что путь к данным будет от папки скрипта
df_frac = CSV.read("./data_out/fseird_params.csv", DataFrame)
df_cmp  = CSV.read("./data_out/model_comparison_pub.csv", DataFrame)
df_traj = CSV.read("./data_out/seird_all_waves_trajectories.csv", DataFrame)

# ── ВЫЧИСЛЕНИЕ ПЕРЕМЕННЫХ ─────────────────────────────────────────────────────

α      = df_cmp.α_k                        # вектор 10 значений
sig    = df_cmp.sig                         # "ns" / "*" / "**" / "***"
is_sig = sig .!= "ns"

# (А) Кумулятивные случаи в волнах j < k (нижняя оценка кумулятивной заражённости)
cases_by_wave = [
    sum(df_traj[df_traj.wave .== k, :obs_cases])
    for k in 1:10
]
cumul_prior = [sum(cases_by_wave[1:k-1]) for k in 1:10]   # 0 для волны 1

# (Б) N_eff_frac / N_city (%)
neff_pct = df_frac.N_frac_pct   # уже в %

# (В) Индекс асимметрии: длина_спада / длина_подъёма
function asymmetry_index(wave_k)
    series = df_traj[df_traj.wave .== wave_k, :obs_cases]
    pk     = argmax(series)           # 1-based
    rise   = pk                       # дней от начала до пика включительно
    fall   = length(series) - pk + 1  # дней от пика до конца включительно
    return fall / max(rise, 1)
end
asym = [asymmetry_index(k) for k in 1:10]

# ── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ───────────────────────────────────────────────────

# Линейная регрессия y = a + b*x, возвращает (a, b, r_spearman, p_approx)
function linreg_and_spearman(x, y)
    n  = length(x)
    b  = cov(x, y) / var(x)
    a  = mean(y) - b * mean(x)
    rs = corspearman(x, y)
    # приближённое p-значение для ρ Спирмена (t-распределение, df=n-2)
    t_stat = rs * sqrt((n - 2) / (1 - rs^2 + 1e-15))
    p_val  = 2 * (1 - cdf(TDist(n - 2), abs(t_stat)))
    return a, b, rs, p_val
end

using Distributions   # для TDist и cdf

# CI-полоса для линейной регрессии
function ci_band(x_fit, x_obs, y_obs, conf=0.95)
    n   = length(x_obs)
    b   = cov(x_obs, y_obs) / var(x_obs)
    a   = mean(y_obs) - b * mean(x_obs)
    ŷ   = a .+ b .* x_fit
    res = y_obs .- (a .+ b .* x_obs)
    s2  = sum(res.^2) / (n - 2)
    SE  = sqrt.(s2 .* (1/n .+ (x_fit .- mean(x_obs)).^2 / sum((x_obs .- mean(x_obs)).^2)))
    t   = quantile(TDist(n - 2), 1 - (1 - conf)/2)
    return ŷ, ŷ .- t.*SE, ŷ .+ t.*SE
end

# ── ЦВЕТА И СТИЛЬ ─────────────────────────────────────────────────────────────
COLOR_SIG = :crimson
COLOR_NS  = :steelblue
MARKER_SIG = :circle
MARKER_NS  = :circle

wave_labels = string.(1:10)

marker_colors = [is_sig[k] ? COLOR_SIG : COLOR_NS for k in 1:10]
marker_fills  = marker_colors   # закрашенные = значимые, контур = нс
ms = 11

# ── СУБГРАФИК А: α_k vs кумулятивные случаи ─────────────────────────────────
x_A = cumul_prior ./ 1000   # в тысячах случаев для читаемости оси
x_A_fit = range(minimum(x_A)*0.9, maximum(x_A)*1.05, length=100)
ŷ_A, lo_A, hi_A = ci_band(collect(x_A_fit), x_A, α)
a_A, b_A, rs_A, p_A = linreg_and_spearman(x_A, α)

p_stars_A = p_A < 0.001 ? "p < 0.001" : (p_A < 0.01 ? "p < 0.01" : @sprintf("p = %.3f", p_A))
lbl_A = @sprintf("ρ_S = %.2f, %s", rs_A, p_stars_A)

pA = plot(x_A_fit, ŷ_A,
          ribbon = (ŷ_A .- lo_A, hi_A .- ŷ_A),
          fillalpha = 0.15, fillcolor = :gray50,
          lw = 1.5, color = :gray40, ls = :solid, label = "Линейная регрессия",
          xlabel = "Кумулятивные случаи\nволн j < k, тыс.",
          ylabel = "α_k",
          title  = "(А)",
          xlims  = (minimum(x_A_fit), maximum(x_A_fit)),
          ylims  = (0.68, 1.06),
          legend = :topright,
          dpi = 300, size = (380, 380),
          framestyle = :box,
          tickfontsize = 9, labelfontsize = 10, titlefontsize = 11)

# горизонтальная линия α = 1
hline!(pA, [1.0], color = :black, lw = 1, ls = :dash, label = "α = 1")

# точки: нс — пустые, значимые — закрашенные
scatter!(pA, x_A[.!is_sig], α[.!is_sig],
         marker = :circle, ms = ms,
         color = :white, markerstrokecolor = COLOR_NS, markerstrokewidth = 2,
         label = "ns")
scatter!(pA, x_A[is_sig], α[is_sig],
         marker = :circle, ms = ms,
         color = COLOR_SIG, markerstrokecolor = :black, markerstrokewidth = 1,
         label = "p < 0.05 (Бонферрони)")

# подписи номеров волн
for k in 1:10
    dx = k in [6, 9] ? 0.5 : -0.5
    dy = 0.012
    annotate!(pA, x_A[k] + dx, α[k] + dy, text(string(k), 8, :center, :gray30))
end

# аннотация коэффициента корреляции
annotate!(pA, minimum(x_A_fit) + 0.05*(maximum(x_A_fit)-minimum(x_A_fit)),
          0.715, text(lbl_A, 8, :left, :gray20))

# ── СУБГРАФИК Б: α_k vs N_eff_frac % ─────────────────────────────────────────
x_B = neff_pct
x_B_fit = range(minimum(x_B)*0.9, maximum(x_B)*1.05, length=100)
ŷ_B, lo_B, hi_B = ci_band(collect(x_B_fit), x_B, α)
a_B, b_B, rs_B, p_B = linreg_and_spearman(x_B, α)

p_stars_B = p_B < 0.001 ? "p < 0.001" : (p_B < 0.01 ? "p < 0.01" : @sprintf("p = %.3f", p_B))
lbl_B = @sprintf("ρ_S = %.2f, %s", rs_B, p_stars_B)

pB = plot(x_B_fit, ŷ_B,
          ribbon = (ŷ_B .- lo_B, hi_B .- ŷ_B),
          fillalpha = 0.15, fillcolor = :gray50,
          lw = 1.5, color = :gray40, label = "",
          xlabel = "N_eff_frac / N_city, %",
          ylabel = "",
          title  = "(Б)",
          xlims  = (minimum(x_B_fit), maximum(x_B_fit)),
          ylims  = (0.68, 1.06),
          legend = :none,
          dpi = 300, size = (380, 380),
          framestyle = :box,
          tickfontsize = 9, labelfontsize = 10, titlefontsize = 11)

hline!(pB, [1.0], color = :black, lw = 1, ls = :dash, label = "")

scatter!(pB, x_B[.!is_sig], α[.!is_sig],
         marker = :circle, ms = ms,
         color = :white, markerstrokecolor = COLOR_NS, markerstrokewidth = 2,
         label = "")
scatter!(pB, x_B[is_sig], α[is_sig],
         marker = :circle, ms = ms,
         color = COLOR_SIG, markerstrokecolor = :black, markerstrokewidth = 1,
         label = "")

for k in 1:10
    dx = k == 2 ? 2.0 : (k == 5 ? 2.5 : -2.0)
    dy = 0.012
    annotate!(pB, x_B[k] + dx, α[k] + dy, text(string(k), 8, :center, :gray30))
end

annotate!(pB, minimum(x_B_fit) + 0.05*(maximum(x_B_fit)-minimum(x_B_fit)),
          0.715, text(lbl_B, 8, :left, :gray20))

# ── СУБГРАФИК В: α_k vs индекс асимметрии ────────────────────────────────────
x_C = asym
# волна 10 — аномальный выброс (asym ≈ 4.97), отображаем усечённую ось
x_C_plot = min.(x_C, 3.5)   # для визуализации; точка подписывается отдельно
x_C_fit = range(0.0, 3.6, length=100)
ŷ_C, lo_C, hi_C = ci_band(collect(x_C_fit), x_C, α)
a_C, b_C, rs_C, p_C = linreg_and_spearman(x_C, α)

p_stars_C = p_C < 0.001 ? "p < 0.001" : (p_C < 0.01 ? "p < 0.01" : @sprintf("p = %.3f", p_C))
lbl_C = @sprintf("ρ_S = %.2f, %s", rs_C, p_stars_C)

pC = plot(x_C_fit, ŷ_C,
          ribbon = (ŷ_C .- lo_C, hi_C .- ŷ_C),
          fillalpha = 0.15, fillcolor = :gray50,
          lw = 1.5, color = :gray40, label = "",
          xlabel = "Индекс асимметрии\n(длина спада / длина подъёма)",
          ylabel = "",
          title  = "(В)",
          xlims  = (-0.1, 3.7),
          ylims  = (0.68, 1.06),
          legend = :none,
          dpi = 300, size = (380, 380),
          framestyle = :box,
          tickfontsize = 9, labelfontsize = 10, titlefontsize = 11)

hline!(pC, [1.0], color = :black, lw = 1, ls = :dash, label = "")

scatter!(pC, x_C_plot[.!is_sig], α[.!is_sig],
         marker = :circle, ms = ms,
         color = :white, markerstrokecolor = COLOR_NS, markerstrokewidth = 2,
         label = "")
scatter!(pC, x_C_plot[is_sig], α[is_sig],
         marker = :circle, ms = ms,
         color = COLOR_SIG, markerstrokecolor = :black, markerstrokewidth = 1,
         label = "")

for k in 1:10
    # волна 10: x_C реальное ~4.97, но отображаем при 3.5 + стрелка
    xpos = k == 10 ? 3.5 : x_C_plot[k]
    dx = 0.08; dy = 0.012
    annotate!(pC, xpos + dx, α[k] + dy, text(string(k), 8, :center, :gray30))
end

# пометить выброс волны 10
annotate!(pC, 3.5, α[10] - 0.025,
          text("(asym=4.97→)", 7, :right, :gray50))

annotate!(pC, 0.05, 0.715, text(lbl_C, 8, :left, :gray20))

# ── СБОРКА И СОХРАНЕНИЕ ──────────────────────────────────────────────────────
fig9 = plot(pA, pB, pC,
            layout = (1, 3),
            size   = (1180, 420),
            margin = 8Plots.mm,
            plot_title = "Рисунок 9. Взаимосвязь α_k с характеристиками иммунологического контекста волн",
            plot_titlefontsize = 11)

savefig(fig9, "fig9_alpha_context.pdf")
savefig(fig9, "fig9_alpha_context.png")

@info "fig9_alpha_context.pdf / .png сохранены"
