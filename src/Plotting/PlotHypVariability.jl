"""
    plot_hyperparameter_rmse_variability(rmse_per_fold::Vector{Float64},
                                         ins_rmse::Float64,
                                         static_rmse::Float64)

Visualise the spread of RMSE values across the 10 hyperparameter folds, together
with the baseline (INS only) and static‑mean‑correction RMSE.

# Arguments
- `rmse_per_fold`: Per‑fold RMSE (length = number of folds).
- `ins_rmse`: RMSE of the uncorrected aligned INS trajectory (baseline).
- `static_rmse`: RMSE after applying a static (mean) correction.

# Returns
- A `Figure` object (Makie) with two subplots.
"""
function plot_hyperparameter_rmse_variability(
    rmse_per_fold::Vector{Float64},
    ins_rmse::Float64,
    static_rmse::Float64
)
    n_folds = length(rmse_per_fold)
    fold_ids = 1:n_folds

    fig = Figure(size=(1200, 500))
    # Left subplot: bar chart
    ax1 = Axis(fig[1, 1]; title="RMSE per fold",
        xlabel="Hyperparameter fold", ylabel="RMSE (m)")
    bars = barplot!(ax1, fold_ids, rmse_per_fold; color=:steelblue, alpha=0.75,
        strokewidth=0.6, strokecolor=:white)
    # Baselines
    hlines!(ax1, [ins_rmse]; color=:black, linestyle=:dash, linewidth=1.2,
        label="INS only")
    hlines!(ax1, [static_rmse]; color=:tomato, linestyle=:dot, linewidth=1.2,
        label="Static correction")
    hlines!(ax1, [mean(rmse_per_fold)]; color=:steelblue, linewidth=1.0, alpha=0.6,
        label="GP mean (RMSE = $(round(mean(rmse_per_fold), digits=4)) m)")

    # Annotate each bar with its RMSE value
    for (i, val) in enumerate(rmse_per_fold)
        text!(ax1, i, val + 0.0005; text="$(round(val, digits=3))",
            align=(:center, :bottom), fontsize=8, color=:steelblue)
    end
    axislegend(ax1; position=:lt, fontsize=8)

    # Right subplot: boxplot + jittered scatter
    ax2 = Axis(fig[1, 2]; title="Distribution  μ = $(round(mean(rmse_per_fold), digits=4)) m  σ = $(round(std(rmse_per_fold), digits=4)) m",
        ylabel="RMSE (m)")
    # Simple boxplot: manual using quantiles
    q = quantile(rmse_per_fold, [0.25, 0.5, 0.75])
    iqr_box = q[3] - q[1]
    whisker_low = max(minimum(rmse_per_fold), q[1] - 1.5 * iqr_box)
    whisker_high = min(maximum(rmse_per_fold), q[3] + 1.5 * iqr_box)
    # Box
    lines!(ax2, [0.7, 1.3], [q[1], q[1]]; color=:black, linewidth=1)   # bottom of box
    lines!(ax2, [0.7, 1.3], [q[3], q[3]]; color=:black, linewidth=1)   # top of box
    lines!(ax2, [0.85, 1.15], [q[1], q[1]]; color=:steelblue, linewidth=4, alpha=0.5)  # fill
    lines!(ax2, [0.85, 1.15], [q[3], q[3]]; color=:steelblue, linewidth=4, alpha=0.5)
    # Median line
    lines!(ax2, [0.7, 1.3], [q[2], q[2]]; color=:white, linewidth=2)
    # Whiskers
    lines!(ax2, [1.0, 1.0], [whisker_low, q[1]]; color=:black, linestyle=:dash)
    lines!(ax2, [1.0, 1.0], [q[3], whisker_high]; color=:black, linestyle=:dash)
    lines!(ax2, [0.8, 1.2], [whisker_low, whisker_low]; color=:black)
    lines!(ax2, [0.8, 1.2], [whisker_high, whisker_high]; color=:black)

    # Jittered scatter
    jitter = 2 * (rand(Float64, n_folds) .- 0.5) * 0.08
    scatter!(ax2, 1.0 .+ jitter, rmse_per_fold; color=:steelblue,
        markersize=8, alpha=0.85, label="Folds")

    # Baselines
    hlines!(ax2, [ins_rmse]; color=:black, linestyle=:dash, linewidth=1.2,
        label="INS only")
    hlines!(ax2, [static_rmse]; color=:tomato, linestyle=:dot, linewidth=1.2,
        label="Static correction")

    axislegend(ax2; position=:lt, fontsize=8)
    hidexdecorations!(ax2)   # remove x ticks because only one group
    return fig
end