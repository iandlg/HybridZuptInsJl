"""
    plot_noise_sweep_boxplots(df::DataFrame, dataset_name::AbstractString;
                              metric::Symbol=:rmse,
                              save_path::Union{String,Nothing}=nothing)

For a single dataset (from the `run_online_correction_sweep` output), plot box plots of
per-trial results grouped by noise specification (`noise_spec_tag`), with the different
correction methods (`estimator`) shown side-by-side within each noise-spec group.

Groups are ordered by `noise_spec_order` and estimators within each group are ordered by
`estimator_order`, both following the display order already encoded in the dataframe by
`run_online_correction_sweep`.

# Arguments
- `df`: DataFrame produced by `run_online_correction_sweep`.
- `dataset_name`: Which `dataset_name` to filter to and plot.
- `metric`: `:rmse` or `:rmse_rate`.
- `save_path`: Optional path to save the figure.

# Returns
- A `Figure` object.
"""
function plot_noise_sweep_boxplots(
    df::DataFrame,
    dataset_name::AbstractString;
    metric::Symbol=:rmse,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
)
    metric in (:rmse, :rmse_rate) || throw(ArgumentError("metric must be :rmse or :rmse_rate"))

    sub = df[df.dataset_name .== dataset_name, :]
    isempty(sub) && error("No rows found for dataset_name = $dataset_name")

    # ---- Noise-spec groups (ordered by noise_spec_order) ----
    noise_order_map = Dict{String,Int}()
    for row in eachrow(sub)
        noise_order_map[row.noise_spec_tag] = row.noise_spec_order
    end
    noise_specs = sort(unique(sub.noise_spec_tag), by=tag -> noise_order_map[tag])
    n_groups = length(noise_specs)

    # ---- Estimators (ordered by estimator_order) ----
    est_order_map = Dict{String,Int}()
    for row in eachrow(sub)
        est_order_map[row.estimator] = row.estimator_order
    end
    estimators = sort(unique(sub.estimator), by=e -> est_order_map[e])
    n_est = length(estimators)

    colors = Makie.wong_colors()
    est_color = Dict(estimators[i] => colors[mod1(i, length(colors))] for i in 1:n_est)

    group_width = 0.8
    bar_width = n_est > 0 ? group_width / n_est : group_width
    offsets = ((1:n_est) .- (n_est + 1) / 2) * bar_width

    title_str = metric == :rmse ? "RMSE" : "RMSE rate"
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Noise specification",
        ylabel=title_str,
        title="$(title_str) per noise spec — $dataset_name",
        xticks=(1:n_groups, noise_specs),
        xticklabelsize=14,
        xticklabelrotation=π / 6,
    )

    labeled = Set{String}()

    for (g, noise_tag) in enumerate(noise_specs)
        gdf = sub[sub.noise_spec_tag .== noise_tag, :]

        for (j, est) in enumerate(estimators)
            edf = gdf[gdf.estimator .== est, :]
            isempty(edf) && continue

            vals = Float64.(edf[:, metric])
            clean_mask = .!isnan.(vals)
            clean_vals = vals[clean_mask]
            length(clean_vals) < 1 && continue

            x_pos = g + offsets[j]
            x_positions = fill(x_pos, length(clean_vals))

            boxplot!(ax, x_positions, clean_vals;
                width=bar_width * 0.9,
                color=est_color[est],
                label=est in labeled ? nothing : est,
                show_outliers=show_outliers)
            push!(labeled, est)
        end
    end

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
    metric in (:rmse, :rmse_rate) || throw(ArgumentError("metric must be :rmse or :rmse_rate"))
    metric_name = Dict(:rmse => "RMSE", :rmse_rate => "RMSE rate")[metric]

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
    colors = Makie.wong_colors()
    baseline_color = colors[1]
    est_colors = Dict(estimators[i] => colors[mod1(i+1, length(colors))] for i in 1:n_est)

    # ----- Training step groups: order 0 = baseline, then 1,2,… -----
    group_labels = Dict{Int,String}()
    for row in eachrow(trained_df)
        group_labels[row.train_set_order] = row.train_set
    end
    group_labels[0] = "Base"
    all_orders = sort(collect(keys(group_labels)))

    # ----- Create figure -----
    fig = Figure(size=(450 * n_cols, 450 * n_rows))

    for (idx, test_id) in enumerate(test_ids)
        row_i = (idx - 1) ÷ n_cols + 1
        col_i = (idx - 1) % n_cols + 1

        test_sub = df[df.test_id .== test_id, :]
        test_name = first(test_sub.test_name)

        ax = Axis(fig[row_i, col_i];
            title="Tested on $test_name",
            ylabel=metric_name,
            xgridvisible=false)

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
                text!(ax, base_pos, base_val; text=string(round(base_val, digits=4)),
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
                    text!(ax, xpos, val; text=string(round(val, digits=4)),
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