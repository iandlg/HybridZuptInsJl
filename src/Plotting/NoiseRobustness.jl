"""
Shared grouped-boxplot machinery for the noise-robustness figures: `group_col` on
the x axis, `series_col` side-by-side within each group, both ordered by their
`*_order` companion columns so the figure follows the order the sweep was declared
in rather than alphabetical order. Every box spans the trials — and, if the sweep
drew more than one noise realisation per cell (more than one seed), those too.
"""
function _grouped_boxplot!(
    ax::Axis,
    sub::DataFrame,
    value_col::Symbol;
    group_col::Symbol=:noise_spec_tag,
    group_order_col::Symbol=:noise_spec_order,
    series_col::Symbol=:estimator,
    series_order_col::Symbol=:estimator_order,
    show_outliers::Bool=true,
    show_points::Bool=false,
)
    group_order_map = Dict{Any,Int}()
    series_order_map = Dict{Any,Int}()
    for row in eachrow(sub)
        group_order_map[row[group_col]] = row[group_order_col]
        series_order_map[row[series_col]] = row[series_order_col]
    end
    groups = sort(unique(sub[:, group_col]), by=g -> group_order_map[g])
    series = sort(unique(sub[:, series_col]), by=e -> series_order_map[e])
    n_groups, n_series = length(groups), length(series)

    # Colour by name, not by position within THIS figure: the paired figure omits
    # the baseline, and indexing by position would shift every remaining estimator
    # onto the colour its neighbour had in the unpaired one. The shared table keys
    # on the estimator name, so the paired figure simply has no "ZUPT only" box.
    series_color = series_color_map(series)

    group_width = 0.8
    bar_width = n_series > 0 ? group_width / n_series : group_width
    offsets = ((1:n_series) .- (n_series + 1) / 2) * bar_width

    # Fixed RNG: the point jitter is cosmetic, and a figure that moves between
    # rebuilds is a nuisance when it sits in a document.
    jitter_rng = Random.Xoshiro(0)

    labeled = Set{Any}()
    for (g, grp) in enumerate(groups)
        gdf = sub[sub[:, group_col] .== grp, :]

        for (j, ser) in enumerate(series)
            edf = gdf[gdf[:, series_col] .== ser, :]
            isempty(edf) && continue

            vals = Float64.(edf[:, value_col])
            clean_vals = vals[.!isnan.(vals)]
            length(clean_vals) < 1 && continue

            x_pos = g + offsets[j]

            boxplot!(ax, fill(x_pos, length(clean_vals)), clean_vals;
                width=bar_width * 0.9,
                color=series_color[ser],
                label=ser in labeled ? nothing : ser,
                show_outliers=show_outliers)
            if show_points
                jitter = (rand(jitter_rng, length(clean_vals)) .- 0.5) .* (bar_width * 0.35)
                scatter!(ax, x_pos .+ jitter, clean_vals;
                    color=(:black, 0.45), markersize=4)
            end
            push!(labeled, ser)
        end
    end

    ax.xticks = (1:n_groups, string.(groups))
    return labeled
end

"""
    plot_noise_sweep_boxplots(df::DataFrame, dataset_name::AbstractString;
                              metric::Symbol=:rmse,
                              save_path::Union{String,Nothing}=nothing)

For a single dataset (from the `run_online_correction_sweep` output), plot box plots of
per-trial results grouped by noise specification (`noise_spec_tag`), with the different
correction methods (`estimator`) shown side-by-side within each noise-spec group. Each
box spans the trials for that (noise spec, estimator) combination, times the number of
noise draws the sweep took per cell.

Groups are ordered by `noise_spec_order` and estimators within each group are ordered by
`estimator_order`, both following the display order already encoded in the dataframe by
`run_online_correction_sweep`.

# Arguments
- `df`: DataFrame produced by `run_online_correction_sweep`.
- `dataset_name`: Which `dataset_name` to filter to and plot.
- `metric`: `:rmse`, `:rmse_rate` or `:rmse_yaw`.
- `show_points`: overlay the individual trials on each box. Off by default; worth
  turning on when a box summarises ~10 points and the quartiles could be mistaken
  for an error bar.
- `save_path`: Optional path to save the figure.

# Returns
- A `Figure` object.

!!! note "This figure is unpaired"
    Each estimator's box is built independently, so overlapping boxes do **not** mean
    the estimators are indistinguishable — they may differ consistently within every
    trial while their boxes overlap, because walk-to-walk difficulty is the larger
    source of spread. `plot_noise_paired_relative_change` is the paired counterpart.
"""
function plot_noise_sweep_boxplots(
    df::DataFrame,
    dataset_name::AbstractString;
    metric::Symbol=:rmse,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
)
    check_metric(metric)

    sub = df[df.dataset_name .== dataset_name, :]
    isempty(sub) && error("No rows found for dataset_name = $dataset_name")

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Noise specification",
        ylabel=metric_label(metric),
        title="$(metric_title(metric)) per noise spec — $dataset_name",
        xticklabelsize=14,
        xticklabelrotation=π / 6,
    )

    labeled = _grouped_boxplot!(ax, sub, metric;
        show_outliers=show_outliers, show_points=show_points)

    if !isempty(labeled)
        Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false)
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_noise_paired_relative_change(paired::DataFrame, dataset_name::AbstractString;
                                      save_path::Union{String,Nothing}=nothing)

The paired counterpart of `plot_noise_sweep_boxplots`: same noise specs on the x axis,
but each box holds the per-trial **relative change against the reference estimator**
(from `paired_estimator_contrast`), in percent, with a reference line at zero. The
reference estimator has no box — it is the zero line.

Because each point is a difference taken within one trial, walk-to-walk difficulty
cancels, so a box sitting clear of zero is a consistent effect — a claim the unpaired
figure cannot support.

# Arguments
- `paired`: output of `paired_estimator_contrast`.
- `dataset_name`: which `dataset_name` to filter to.
- `value_col`: `:rel_change_pct` (default) or `:delta` (the metric's own units).
- `metric`: only used to name the quantity in the axis label.
- `show_points`: overlay the individual trials on each box.
- `show_subtitle`: draw the subtitle under the title (default `true`). Turn it off
  when the document's own caption carries the same information.
"""
function plot_noise_paired_relative_change(
    paired::DataFrame,
    dataset_name::AbstractString;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse_rate,
    reference_label::AbstractString="baseline",
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
    show_subtitle::Bool=true,
)
    check_metric(metric)
    value_col in (:delta, :rel_change_pct) || throw(ArgumentError(
        "value_col must be :delta or :rel_change_pct, got :$value_col"))

    sub = paired[paired.dataset_name .== dataset_name, :]
    isempty(sub) && error("No rows found for dataset_name = $dataset_name")

    as_pct = value_col === :rel_change_pct
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Noise specification",
        # Kept short on purpose: spelling the formula out here overflows the axis
        # once `reference_label` is a real estimator name. The subtitle carries it.
        ylabel=as_pct ? "relative change in $(metric_quantity(metric)) [%]" :
               "change in $(metric_label(metric))",
        title="Paired per-trial change vs \"$reference_label\" — $dataset_name",
        subtitle=!show_subtitle ? "" :
                 as_pct ? "(estimator − $reference_label) / |$reference_label|, per trial" :
                 "estimator − $reference_label, per trial",
        subtitlesize=10,
        xticklabelsize=14,
        xticklabelrotation=π / 6,
    )
    as_pct && (ax.ytickformat = vs -> [string(round(v; digits=1), "%") for v in vs])

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1)
    labeled = _grouped_boxplot!(ax, sub, value_col;
        show_outliers=show_outliers, show_points=show_points)

    if !isempty(labeled)
        Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false)
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end

function plot_multi_track_training_quality(
    df::DataFrame;
    metric::Symbol=:rmse_rate,
    save_path::Union{String,Nothing}=nothing
)
    # ----- Input validation -----
    check_metric(metric)
    metric_name = metric_label(metric)

    # ----- Determine test order (preserve input order) -----
    test_order_map = Dict{Int,Int}()
    for row in eachrow(df)
        test_order_map[row.test_id] = row.test_order
    end
    test_ids = sort(unique(df.test_id), by=tid -> test_order_map[tid])
    n_tests = length(test_ids)
    n_cols = min(3, n_tests)
    n_rows = ceil(Int, n_tests / n_cols)

    # ----- Separate baseline and trained rows -----
    base_df = df[df.train_set .== "Base", :]          # baseline (no training)
    trained_df = df[df.train_set .!= "Base", :]       # all trained steps

    # ----- Extract estimators and their order -----
    est_order_map = Dict{String,Int}()
    for row in eachrow(trained_df)
        est_order_map[row.estimator] = row.estimator_order
    end
    estimators = sort(unique(trained_df.estimator), by=e -> est_order_map[e])
    n_est = length(estimators)

    # ----- Colour palette -----
    # The baseline is drawn from its own `train_set == "Base"` rows and is absent
    # from `estimators`, so it is named explicitly rather than reserving slot 1
    # and shifting everything else along by one.
    baseline_color = method_color("ZUPT only")
    est_colors = series_color_map(estimators)

    # ----- Training step groups: order 0 = baseline, then 1,2,… -----
    group_labels = Dict{Int,String}()
    for row in eachrow(trained_df)
        group_labels[row.train_set_order] = row.train_set
    end
    group_labels[0] = "Base"
    all_orders = sort(collect(keys(group_labels)))

    # ----- Create figure -----
    fig = Figure(size=(450 * n_cols, 450 * n_rows))
    axs = Axis[]
    first_col_axs = Axis[]

    for (idx, test_id) in enumerate(test_ids)
        row_i = (idx - 1) ÷ n_cols + 1
        col_i = (idx - 1) % n_cols + 1

        test_sub = df[df.test_id .== test_id, :]
        test_name = first(test_sub.test_name)

        ax = Axis(fig[row_i, col_i];
            title="Tested on $test_name",
            ylabel=metric_name,
            xgridvisible=false)
        push!(axs, ax)
        col_i == 1 && push!(first_col_axs, ax)

        # Position each group at integer x: baseline at 1, step 1 at 2, etc.
        group_positions = Dict{Int,Float64}()
        for (i, order) in enumerate(all_orders)
            group_positions[order] = Float64(i + 1)
        end

        # ----- Plot baseline (only for group 0) -----
        base_rows = base_df[base_df.test_id .== test_id, :]
        if !isempty(base_rows)
            base_val = first(base_rows[:, metric])
            base_pos = group_positions[0]
            barplot!(ax, [base_pos], [base_val]; color=baseline_color, width=0.6)
            if !isnan(base_val)
                text!(ax, base_pos, base_val; text=string(round(base_val, digits=2)),
                    align=(:center, :bottom), fontsize=9)
            end
        end

        # ----- Plot each training step -----
        for order in all_orders
            order == 0 && continue
            group_center = group_positions[order]
            gdf = trained_df[(trained_df.test_id .== test_id) .& (trained_df.train_set_order .== order), :]
            isempty(gdf) && continue

            # Compute bar positions within the group
            bar_width = n_est > 0 ? 0.8 / n_est : 0.8
            offsets = ((1:n_est) .- (n_est + 1) / 2) .* bar_width

            for (j, est) in enumerate(estimators)
                edf = gdf[gdf.estimator .== est, :]
                isempty(edf) && continue
                val = first(edf[:, metric])
                xpos = group_center + offsets[j]
                barplot!(ax, [xpos], [val]; color=est_colors[est], width=bar_width * 0.9)
                if !isnan(val)
                    text!(ax, xpos, val; text=string(round(val, digits=2)),
                        align=(:center, :bottom), fontsize=8)
                end
            end
        end

        # ----- Set x‑axis ticks -----
        xticks_pos = [group_positions[order] for order in all_orders]
        xticks_lab = [group_labels[order] for order in all_orders]
        ax.xticks = (xticks_pos, xticks_lab)
        ax.xticklabelrotation = π / 6
    end

    # ----- Share one y scale across panels -----
    # Each panel is the same metric on a different test track, so autoscaling them
    # independently makes bars of different magnitude draw at the same height and
    # invites reading a bad track as a good one. Only the leftmost column keeps its
    # ticks/label; the rest are redundant once the scale is common.
    if length(axs) > 1
        linkyaxes!(axs...)
        for ax in axs
            ax in first_col_axs || hideydecorations!(ax; grid=false)
        end
    end

    # ----- Legend -----
    legend_elems = [PolyElement(color=baseline_color)]
    legend_labels = ["Baseline"]
    for est in estimators
        push!(legend_elems, PolyElement(color=est_colors[est]))
        push!(legend_labels, split(est, '.')[end])   # short name
    end
    Legend(fig[n_rows+1, 1:n_cols], legend_elems, legend_labels;
        orientation=:horizontal, tellwidth=false)

    # ----- Save or return -----
    isnothing(save_path) || save(save_path, fig)
    return fig
end