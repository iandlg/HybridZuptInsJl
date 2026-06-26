function plot_beta_evolution(beta_history::Matrix{Float64})
    m, N = size(beta_history)

    log_abs_beta = log10.(abs.(beta_history) .+ eps())
    vmax = maximum(log_abs_beta)
    vmin = minimum(log_abs_beta)

    fig = Figure(size=(1200, 600))
    ax = Axis(fig[1, 1];
        title="Beta vector evolution (log₁₀ |β|)",
        xlabel="Time step",
        ylabel="Beta index")

    hm = heatmap!(ax, 1:N, 1:m, log_abs_beta';
        colormap=:viridis,
        colorrange=(vmin, vmax))

    Colorbar(fig[1, 2], hm;
        label="log₁₀ |β|",
        width=15,
        ticksize=5,
        ticks=(range(vmin, vmax; length=6),
            ["10^$(round(v; digits=1))" for v in range(vmin, vmax; length=6)]))

    return fig
end

function plot_corrector_boxplots(
    df::DataFrame,
    metric::Symbol;
    save_path::Union{String,Nothing}=nothing,
)
    if !(metric in (:rmse, :rmse_rate))
        throw(ArgumentError("metric must be :rmse or :rmse_rate"))
    end

    ratios = sort(unique(df.train_ratio))
    correctors = sort(unique(df.corrector))
    n_correctors = length(correctors)

    colors = Makie.wong_colors()
    corr_to_color = Dict(correctors[i] => colors[mod1(i, length(colors))] for i in 1:n_correctors)

    min_gap = length(ratios) > 1 ? minimum(diff(ratios)) : 1.0
    total_width = 0.8 * min_gap
    box_width = total_width / n_correctors
    offsets = ((1:n_correctors) .- (n_correctors + 1) / 2) * box_width

    fig = Figure(size=(900, 600))
    title_str = metric == :rmse ? "RMSE" : "RMSE rate"
    ax = Axis(fig[1, 1],
        xlabel="Train ratio",
        ylabel=title_str,
        title="$(title_str) per train ratio",
        xticks=(ratios, string.(ratios)),
        xticklabelsize=16
    )

    labeled = Set{String}()

    for (j, corr) in enumerate(correctors)
        for (i, ratio) in enumerate(ratios)
            mask = (df.train_ratio .== ratio) .& (df.corrector .== corr)
            sub = df[mask, :]
            vals = sub[:, metric]
            clean_mask = .!isnan.(vals)
            clean_vals = Float64.(vals[clean_mask])
            clean_ids = sub[clean_mask, :trial_id]  # assumes column is named :trial_id

            if length(clean_vals) >= 2
                x_pos = ratio + offsets[j]
                x_positions = fill(x_pos, length(clean_vals))

                boxplot!(ax, x_positions, clean_vals,
                    width=box_width,
                    color=corr_to_color[corr],
                    label=corr in labeled ? nothing : corr)
                push!(labeled, corr)

                # Identify and annotate outliers using Tukey fences
                q1, q3 = quantile(clean_vals, [0.25, 0.75])
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr

                for (v, tid) in zip(clean_vals, clean_ids)
                    if v < lower_fence || v > upper_fence
                        text!(ax, x_pos, v,
                            text=string(tid),
                            fontsize=12,
                            align=(:center, :bottom),
                            offset=(6, 4),  # nudge label just above the point
                        )
                    end
                end
            end
        end
    end

    if !isempty(labeled)
        Legend(fig[2, :], ax; orientation=:horizontal, tellwidth=false)
    end

    if !isnothing(save_path)
        save(save_path, fig)
    end

    return fig
end