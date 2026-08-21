function plot_hp_sensitivity(
    df::DataFrame, grid::ParamGrid, log_range::Tuple{Float64,Float64};
    save_path::Union{String,Nothing}=nothing)

    n_rows, n_cols = size(grid.specs)
    fig = Figure(size=(400 * n_cols, 400 * n_rows))

    # Get baseline RMSE
    baseline_row = df[df.parameter .== "baseline", :]
    baseline_rmse = isempty(baseline_row) ? 1.0 : first(baseline_row.rmse)

    col_colors = Makie.wong_colors()
    for row in 1:n_rows
        for col in 1:n_cols
            spec = grid.specs[row, col]
            if isnothing(spec)
                # Skip empty cell – leave blank
                continue
            end
            # Find data for this parameter
            sub = df[df.parameter .== spec.name, :]
            if isempty(sub)
                @warn "Missing data for $(spec.name)"
                continue
            end
            ax = Axis(fig[row, col];
                xlabel="log₁₀(relative change)",
                ylabel=row == n_rows ? "RMSE ratio" : "",
                title=spec.name,
                xgridstyle=:dash,
                ygridstyle=:dash)
            base_val = first(sub.base_value)
            rel_change = log10.(sub.tested_value ./ base_val)
            lines!(ax, rel_change, sub.rmse_ratio;
                color=col_colors[col], linewidth=2)
            scatter!(ax, rel_change, sub.rmse_ratio;
                color=col_colors[col], markersize=6)
            hlines!(ax, 1.0, color=:gray, linestyle=:dash, linewidth=1)
            xlims!(ax, log_range...)
        end
    end

    # Add global y-label
    Label(fig[1:n_rows, 0], "RMSE ratio", rotation=π / 2, fontsize=14)

    if !isnothing(save_path)
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    display(fig)
    return fig
end

"""
    plot_channel_boxplots(df; show_trials, save_path)

Box-plot the full raw-value distribution per channel (one box = all samples across all
trials), then overlay each trial's mean ± std as a scatter + error bar.
"""
function plot_channel_boxplots(
    df::DataFrame;
    show_trials::Bool=true,
    save_path::Union{String,Nothing}=nothing,
)
    input_df = subset(df, :io_type => x -> x .== :input)
    output_df = subset(df, :io_type => x -> x .== :output)

    fig = Figure(size=(1000, 500))
    ax_in = Axis(fig[1, 1], title="Input channels",
        xlabel="Channel", ylabel="Value")
    ax_out = Axis(fig[1, 2], title="Output channels",
        xlabel="Channel", ylabel="Value")

    # ------------------------------------------------------------------ #
    function draw_panel!(ax, panel_df, box_color)

        channels = sort(unique(panel_df.channel))
        ax.xticks = channels

        # ── 1. Box plot over ALL raw samples ──────────────────────────── #
        boxplot!(ax, panel_df.channel, panel_df.value;
            color=(box_color, 0.45),
            mediancolor=:white,
            whiskerwidth=0.5,
            show_outliers=true,
            outliercolor=(box_color, 0.3),
        )

        # ── 2. Per-trial mean ± std overlaid ──────────────────────────── #
        if show_trials
            trial_ids = sort(unique(panel_df.trial_id))

            # One distinct color per trial drawn from the :tableau_10 palette.
            palette = Makie.wong_colors()
            trial_color = Dict(t => palette[mod1(i, length(palette))]
                               for (i, t) in enumerate(trial_ids))

            # Global mean per channel (across all trials) — used to gauge how
            # "central" each trial point is and scale its jitter accordingly.
            ch_global_mean = Dict(
                ch => mean(panel_df.value[panel_df.channel .== ch])
                for ch in channels
            )
            all_trial_means = [
                mean(panel_df.value[panel_df.channel .== ch.&&panel_df.trial_id .== t])
                for ch in channels for t in trial_ids
            ]
            mean_min = minimum(all_trial_means)
            mean_max = maximum(all_trial_means)
            range_mean = max(mean_max - mean_min, eps(Float64))

            decay = 5.0   # jitter sharpness (3–10): higher → tighter near mean
            max_jitter = 0.4   # half-width at maximum spread

            for trial in trial_ids
                tdf = subset(panel_df, :trial_id => x -> x .== trial)
                color = trial_color[trial]

                xs = Float64[]
                mus = Float64[]
                sigmas = Float64[]

                for ch in channels
                    vals = tdf.value[tdf.channel .== ch]
                    isempty(vals) && continue

                    mu = mean(vals)

                    # Trials close to the channel mean get MORE jitter so they
                    # don't pile up; distant outliers need less spread.
                    d = clamp(abs(ch_global_mean[ch] - mu) / range_mean, 0.0, 1.0)
                    jitter_factor = exp(-decay * d)
                    jitter = (rand() - 0.5) * max_jitter * jitter_factor

                    push!(xs, ch + jitter)
                    push!(mus, mu)
                    push!(sigmas, std(vals))
                end

                errorbars!(ax, xs, mus, sigmas;
                    color=(color, 0.7),
                    whiskerwidth=3,
                    linewidth=1,
                )
                scatter!(ax, xs, mus;
                    markersize=6,
                    color=color,
                    strokewidth=0,
                )
                # Trial number label just above each mean dot
                for (x, mu) in zip(xs, mus)
                    text!(ax, x, mu;
                        text=string(trial),
                        offset=(0, 6),
                        align=(:right, :bottom),
                        fontsize=10,
                        color=color,
                    )
                end
            end
        end
    end
    # ------------------------------------------------------------------ #

    draw_panel!(ax_in, input_df, :steelblue)
    draw_panel!(ax_out, output_df, :coral)

    linkyaxes!(ax_in, ax_out)

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_max_relative_change(
        df::DataFrame;
        exclude_baseline::Bool=true,
        save_path::Union{String,Nothing}=nothing,
    )

Bar chart of the maximum (most positive) `relative_change` achieved per parameter,
as produced by `vary_hsgp_parameters`. Bars are sorted ascending left-to-right
(smallest max relative change on the left, largest on the right) and colored by
`type`.

# Arguments
- `df`: Output of `vary_hsgp_parameters` (needs `parameter`, `type`, `relative_change`).
- `exclude_baseline`: If `true` (default), drops the `parameter == "baseline"` row
  before computing per-parameter maxima.
- `save_path`: Optional path to save the figure.

# Returns
- A `Figure` object.
"""
function plot_max_relative_change(
    df::DataFrame;
    exclude_baseline::Bool=true,
    save_path::Union{String,Nothing}=nothing,
    show_text_val::Bool=false,
    color_offset::Int=3
)
    plot_df = exclude_baseline ? df[df.parameter .!= "baseline", :] : df
    isempty(plot_df) && error("No rows to plot after filtering baseline")

    # Max relative_change per parameter, keeping the associated type
    gdf = combine(
        groupby(plot_df, :parameter),
        :relative_change => maximum => :max_relative_change,
        :type => first => :type,
    )

    # Sort ascending: smallest on the left, largest on the right
    sort!(gdf, :max_relative_change; rev=true)

    n = nrow(gdf)
    xs = 1:n

    types = unique(gdf.type)
    colors = Makie.wong_colors()
    type_color = Dict(t => colors[mod1(i+color_offset, length(colors))] for (i, t) in enumerate(types))
    bar_colors = [type_color[t] for t in gdf.type]

    fig = Figure(size=(max(800, 40 * n), 500))
    ax = Axis(fig[1, 1];
        title="Maximum relative RMSE change per parameter",
        ylabel="Max relative change",
        xticks=(xs, gdf.parameter),
        xticklabelrotation=π / 3,
        xgridvisible=false)

    barplot!(ax, xs, gdf.max_relative_change; color=bar_colors)
    hlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)

    if show_text_val
        for (x, val) in zip(xs, gdf.max_relative_change)
            text!(ax, x, val;
                text="$(round(val, digits=3))",
                align=(:center, val >= 0 ? :bottom : :top),
                fontsize=8)
        end
    end
    legend_elems = [PolyElement(color=type_color[t]) for t in types]
    Legend(fig[1, 2], legend_elems, string.(types); tellheight=false)

    isnothing(save_path) || save(save_path, fig)
    return fig
end