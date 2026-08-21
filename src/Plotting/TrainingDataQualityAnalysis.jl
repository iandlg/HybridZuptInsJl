"""
    plot_train_data_quality(df::DataFrame; metric=:rmse_rate, save_path=nothing)

Given the output of `training_data_quality_analysis`, produce a grid of grouped bar
plots. Subplots appear in `test_order` order, training-track groups in `train_order`
order, and estimators within each group in `estimator_order` order.
"""
function plot_train_data_quality(
    df::DataFrame;
    metric::Symbol=:rmse_rate,
    save_path::Union{String,Nothing}=nothing
)
    check_metric(metric)

    short_name(s::AbstractString) = split(s, '.')[end]
    metric_name = metric_label(metric)
    baseline_name = isempty(df[isnothing.(df.train_id), :]) ? "Baseline" : first(df[isnothing.(df.train_id), :estimator])
    # Preserve user-supplied dictionary order via the *_order columns
    test_order_map = Dict{Int,Int}()
    for row in eachrow(df)
        test_order_map[row.test_id] = row.test_order
    end
    test_ids = sort(unique(df.test_id), by=tid -> test_order_map[tid])
    n_tests = length(test_ids)
    n_cols = min(3, n_tests)
    n_rows = ceil(Int, n_tests / n_cols)

    trained_df = df[.!isnothing.(df.train_id), :]

    est_order_map = Dict{String,Int}()
    for row in eachrow(trained_df)
        est_order_map[row.estimator] = row.estimator_order
    end
    estimators = sort(unique(trained_df.estimator), by=e -> est_order_map[e])
    n_est = length(estimators)

    colors = Makie.wong_colors()
    est_color = Dict(estimators[i] => colors[mod1(i + 1, length(colors))] for i in 1:n_est)
    baseline_color = colors[1]

    fig = Figure(size=(450 * n_cols, 450 * n_rows))

    for (idx, test_id) in enumerate(test_ids)
        row_i = (idx - 1) ÷ n_cols + 1
        col_i = (idx - 1) % n_cols + 1

        sub = df[df.test_id .== test_id, :]
        test_name = first(sub.test_name)

        baseline_sub = sub[isnothing.(sub.train_id), :]
        baseline_val = isempty(baseline_sub) ? NaN : first(baseline_sub[:, metric])


        trained_sub = sub[.!isnothing.(sub.train_id), :]

        train_order_map = Dict{String,Int}()
        for row in eachrow(trained_sub)
            train_order_map[row.train_name] = row.train_order
        end
        train_names = sort(unique(trained_sub.train_name), by=tname -> train_order_map[tname])
        n_groups_train = length(train_names)

        ax = Axis(fig[row_i, col_i];
            title="Tested on $test_name",
            ylabel=metric_name,
            xgridvisible=false)

        group_width = 0.8
        bar_width = n_est > 0 ? group_width / n_est : group_width

        xticks_pos = Float64[1.0]
        xticks_lab = String[baseline_name]

        if !isnan(baseline_val)
            barplot!(ax, [1.0], [baseline_val]; color=baseline_color, width=0.6)
            text!(ax, 1.0, baseline_val; text="$(round(baseline_val, digits=4))",
                align=(:center, :bottom), fontsize=9)
        end

        for (g, tname) in enumerate(train_names)
            group_center = 1.0 + g
            gdf = trained_sub[trained_sub.train_name .== tname, :]
            offsets = ((1:n_est) .- (n_est + 1) / 2) .* bar_width

            for (j, est) in enumerate(estimators)
                edf = gdf[gdf.estimator .== est, :]
                isempty(edf) && continue
                val = first(edf[:, metric])
                xpos = group_center + offsets[j]

                barplot!(ax, [xpos], [val]; color=est_color[est], width=bar_width * 0.9)
                text!(ax, xpos, val; text="$(round(val, digits=4))",
                    align=(:center, :bottom), fontsize=8)
            end

            push!(xticks_pos, group_center)
            push!(xticks_lab, tname)
        end

        ax.xticks = (xticks_pos, xticks_lab)
        ax.xticklabelrotation = π / 6
    end

    legend_elems = [PolyElement(color=baseline_color)]
    legend_labels = [baseline_name]
    for est in estimators
        push!(legend_elems, PolyElement(color=est_color[est]))
        push!(legend_labels, short_name(est))
    end
    Legend(fig[n_rows+1, 1:n_cols], legend_elems, legend_labels;
        orientation=:horizontal, tellwidth=false)

    isnothing(save_path) || save(save_path, fig)
    return fig
end