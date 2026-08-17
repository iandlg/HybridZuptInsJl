"""
    plot_noise_sweep_boxplots(df::DataFrame, dataset_name::AbstractString;
                              metric::Symbol=:rmse,
                              save_path::Union{String,Nothing}=nothing)

For a single dataset (from the `run_online_correction_sweep` output), plot box plots of
per-trial results grouped by noise level (`pos_std`, assumed uniform across x/y/z — i.e.
"noise will be uniform in space"), with the different correction methods (`estimator`)
shown side-by-side within each noise-level group.

Groups are ordered by `pos_std_order` and estimators within each group are ordered by
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

    # ---- Noise-level groups (ordered by pos_std_order) ----
    noise_level_label(v) = isnothing(v) ? "0" :
                           v isa Real ? string(round(v; digits=4)) :
                           "[" * join(round.(v; digits=4), ", ") * "]"

    noise_order_map = Dict{Any,Int}()
    noise_label_map = Dict{Any,String}()
    for row in eachrow(sub)
        noise_order_map[row.pos_std] = row.pos_std_order
        noise_label_map[row.pos_std] = noise_level_label(row.pos_std)
    end
    noise_levels = sort(unique(sub.pos_std), by=v -> noise_order_map[v])
    n_groups = length(noise_levels)

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
        xlabel="Noise level (pos_std)",
        ylabel=title_str,
        title="$(title_str) per noise level — $dataset_name",
        xticks=(1:n_groups, [noise_label_map[nl] for nl in noise_levels]),
        xticklabelsize=14,
    )

    labeled = Set{String}()

    for (g, noise_level) in enumerate(noise_levels)
        gdf = sub[isequal.(sub.pos_std, Ref(noise_level)), :]

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