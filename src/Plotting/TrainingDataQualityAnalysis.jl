"""
    plot_train_data_quality(df::DataFrame; metric=:rmse_rate, save_path=nothing)

Given the output of `train_test_variability_analysis`, produce a grid of bar
plots (max 3 per row, one per test trial) showing RMSE(-rate) sorted smallest
→ largest, bars labeled with the training-track name.
"""
function plot_train_data_quality(
    df::DataFrame;
    metric::Symbol=:rmse_rate,
    save_path::Union{String,Nothing}=nothing
)
    metric in (:rmse, :rmse_rate) || throw(ArgumentError("metric must be :rmse or :rmse_rate"))

    test_ids = sort(unique(df.test_id))
    n_tests = length(test_ids)
    n_cols = min(3, n_tests)
    n_rows = ceil(Int, n_tests / n_cols)

    fig = Figure(size=(420 * n_cols, 420 * n_rows))

    for (idx, test_id) in enumerate(test_ids)
        row = (idx - 1) ÷ n_cols + 1
        col = (idx - 1) % n_cols + 1

        sub = df[df.test_id .== test_id, :]
        p = sortperm(sub[:, metric])
        sub_sorted = sub[p, :]
        test_name = first(sub_sorted.test_name)

        ax = Axis(fig[row, col];
            title="Tested on $test_name",
            ylabel=string(metric),
            xticks=(1:nrow(sub_sorted), sub_sorted.train_name),
            xticklabelrotation=π / 4,
            xgridvisible=false)

        barplot!(ax, 1:nrow(sub_sorted), sub_sorted[:, metric]; color=:steelblue, alpha=0.8)

        for (i, val) in enumerate(sub_sorted[:, metric])
            text!(ax, i, val; text="$(round(val, digits=4))",
                align=(:center, :bottom), fontsize=9)
        end
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end