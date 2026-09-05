"""
    boxplot_dataset_comparison(
        df::DataFrame;
        metric::Symbol=:rmse,
        train_ratio::Union{Real,Nothing}=nothing,
        save_path::Union{String,Nothing}=nothing,
    )

Boxplot of `rmse` or `rmse_rate` grouped by dataset (`dataset_name`), with one boxplot per
correction method (`estimator`) inside each dataset group. Each box aggregates over all
`trial_id`s for that `(dataset_name, estimator)` pair.

If the DataFrame contains multiple `train_ratio` values and `train_ratio` is not
specified, one subplot is produced per `train_ratio` (faceted, similar to
`plot_train_data_quality`). Pass `train_ratio` explicitly to restrict to a single ratio
and get a single axis.

# Arguments
- `df`: Output of `run_online_correction_sweep` (needs `dataset_name`, `estimator`,
  `estimator_order`, `train_ratio`, and the chosen `metric` column).
- `metric`: `:rmse` or `:rmse_rate`.
- `train_ratio`: If given, filters to that single train ratio before plotting.
- `save_path`: Optional path to save the figure.

# Returns
- A `Figure` object.
"""
function boxplot_dataset_comparison(
    df::DataFrame;
    metric::Symbol=:rmse,
    train_ratio::Union{Real,Nothing}=nothing,
    save_path::Union{String,Nothing}=nothing,
)
    check_metric(metric)
    metric_name = metric_label(metric)

    plot_df = isnothing(train_ratio) ? df : df[df.train_ratio .== train_ratio, :]
    isempty(plot_df) && error("No rows to plot" * (isnothing(train_ratio) ? "" : " for train_ratio=$train_ratio"))

    ratios = isnothing(train_ratio) ? sort(unique(plot_df.train_ratio)) : [train_ratio]
    n_ratios = length(ratios)
    n_cols = min(2, n_ratios)
    n_rows = ceil(Int, n_ratios / n_cols)

    # Dataset order: first-appearance order in the DataFrame
    dataset_names = unique(plot_df.dataset_name)
    n_groups = length(dataset_names)

    # Estimator order: from estimator_order column
    est_order_map = Dict{String,Int}()
    for row in eachrow(plot_df)
        est_order_map[row.estimator] = row.estimator_order
    end
    estimators = sort(unique(plot_df.estimator), by=e -> est_order_map[e])
    n_est = length(estimators)

    est_color = series_color_map(estimators)

    group_width = 0.8
    bar_width = n_est > 0 ? group_width / n_est : group_width

    fig = Figure(size=(500 * n_cols, 450 * n_rows))

    for (idx, ratio) in enumerate(ratios)
        row_i = (idx - 1) ÷ n_cols + 1
        col_i = (idx - 1) % n_cols + 1

        sub = plot_df[plot_df.train_ratio .== ratio, :]

        ax = Axis(fig[row_i, col_i];
            title=isnothing(train_ratio) && n_ratios > 1 ? "train_ratio = $ratio" : metric_name,
            ylabel=metric_name,
            xgridvisible=false)

        xticks_pos = Float64[]
        xticks_lab = String[]

        for (g, dname) in enumerate(dataset_names)
            group_center = Float64(g)
            gdf = sub[sub.dataset_name .== dname, :]
            offsets = ((1:n_est) .- (n_est + 1) / 2) .* bar_width

            for (j, est) in enumerate(estimators)
                vals = gdf[gdf.estimator .== est, metric]
                isempty(vals) && continue
                xpos = fill(group_center + offsets[j], length(vals))

                boxplot!(ax, xpos, vals;
                    width=bar_width * 0.9,
                    color=est_color[est],
                    show_outliers=true,
                    label=(idx == 1 ? est : nothing))
            end

            push!(xticks_pos, group_center)
            push!(xticks_lab, dname)
        end

        ax.xticks = (xticks_pos, xticks_lab)
        ax.xticklabelrotation = π / 6
    end

    legend_elems = [PolyElement(color=est_color[est]) for est in estimators]
    Legend(fig[n_rows+1, 1:n_cols], legend_elems, estimators;
        orientation=:horizontal, tellwidth=false)

    isnothing(save_path) || save(save_path, fig)
    return fig
end