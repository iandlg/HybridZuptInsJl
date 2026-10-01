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
    series_colors::Union{Nothing,AbstractDict}=nothing,
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

    # Colour by method NAME via `method_color`, not by position within THIS figure
    # and not by `series_order_col` either: the paired figure omits the baseline and
    # the multi-track figure never declares it as a series at all, so any positional
    # scheme shifts every remaining estimator onto its neighbour's colour. Wong 1/2/3
    # belong to ZUPT only / Static / HSGP by convention -- one table, in
    # `_METHOD_COLOR_INDICES` (Plotting/Regression.jl).
    # `series_colors` overrides the table for series it names, for figures whose
    # series are variants of a method ("HSGP (split)") rather than methods: those miss
    # the table and would all land on one fallback grey. Everything else is unchanged.
    series_color = Dict(ser => isnothing(series_colors) ? method_color(ser) :
                               get(series_colors, ser, method_color(ser)) for ser in series)

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
    plot_noise_paired_relative_change(paired, dataset_name; value_col=:rel_change_pct,
        metric=:rmse_rate, show_points=false, save_path=nothing)

As [`plot_train_ratio_paired_relative_change`](@ref), one group per noise spec.
"""
function plot_noise_paired_relative_change(
    paired::DataFrame,
    dataset_name::AbstractString;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse_rate,
    _ylims::Union{Tuple{Float64,Float64}}=nothing,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
    figsize::Tuple{Int,Int}=(900, 600)
)
    check_metric(metric)
    value_col in (:delta, :rel_change_pct) || throw(ArgumentError(
        "value_col must be :delta or :rel_change_pct, got :$value_col"))

    sub = paired[paired.dataset_name .== dataset_name, :]
    isempty(sub) && error("No rows found for dataset_name = $dataset_name")

    as_pct = value_col === :rel_change_pct
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1],
        ylabel=as_pct ? rich("relative change in ", metric_symbol(metric), " [%]") :
               rich("change in ", metric_label(metric)),
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
    if !isnothing(_ylims)
        ylims!(ax, _ylims[1], _ylims[2])
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_multi_track_training_quality(df; metric=:rmse_rate, save_path=nothing)

One panel per test track of a `multi_track_training_analysis` frame: `metric` against the
number of accumulated training tracks, estimators side by side, untrained baseline
dashed. Boxes span the order seeds; at the last group all seeds share one training set.
"""
function plot_multi_track_training_quality(
    df::DataFrame;
    metric::Symbol=:rmse_rate,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=true,
)
    check_metric(metric)

    # ----- Determine test order (preserve input order) -----
    test_order_map = Dict{Int,Int}()
    for row in eachrow(df)
        test_order_map[row.test_id] = row.test_order
    end
    test_ids = sort(unique(df.test_id), by=tid -> test_order_map[tid])
    n_tests = length(test_ids)
    n_cols = min(3, n_tests)
    n_rows = ceil(Int, n_tests / n_cols)

    base_df = df[df.train_set .== "Base", :]          # baseline (no training)
    trained_df = df[df.train_set .!= "Base", :]       # all trained steps

    fig = Figure(size=(450 * n_cols, 450 * n_rows))
    axs = Axis[]
    first_col_axs = Axis[]
    legend_ax = nothing

    for (idx, test_id) in enumerate(test_ids)
        row_i = (idx - 1) ÷ n_cols + 1
        col_i = (idx - 1) % n_cols + 1

        sub = trained_df[trained_df.test_id .== test_id, :]
        isempty(sub) && continue

        ax = Axis(fig[row_i, col_i];
            title="Tested on $(first(sub.test_name))",
            xlabel="Training tracks accumulated",
            ylabel=metric_label(metric),
            xgridvisible=false)
        push!(axs, ax)
        col_i == 1 && push!(first_col_axs, ax)

        # Baseline first, so it leads the legend: it is the reference every box is read
        # against, not an afterthought. Named and coloured from the dataframe's own
        # baseline rows rather than a literal here, so it follows
        # `base_estimator_name` and keeps the method palette.
        base_rows = base_df[base_df.test_id .== test_id, :]
        if !isempty(base_rows)
            hlines!(ax, [first(base_rows[:, metric])];
                color=method_color(first(base_rows.estimator)),
                linestyle=:dash, linewidth=3, label=first(base_rows.estimator))
        end

        # Group by how many tracks have been accumulated, which is `train_set_order` and is
        # also its own display order.
        _grouped_boxplot!(ax, sub, metric;
            group_col=:train_set_order, group_order_col=:train_set_order,
            show_outliers=show_outliers, show_points=show_points)

        legend_ax = ax
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
    isnothing(legend_ax) || Legend(fig[n_rows+1, 1:n_cols], legend_ax;
        orientation=:horizontal, tellwidth=false)

    # ----- Save or return -----
    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_multi_track_training_noise_panels(df, test_id; metric=:rmse, save_path=nothing)

[`plot_multi_track_training_quality`](@ref) faceted by noise spec for one test track,
on a linked linear y axis. `df` is several `multi_track_training_analysis` frames
stacked, with the caller adding `noise_spec_tag`/`noise_spec_order`.
"""
function plot_multi_track_training_noise_panels(
    df::DataFrame,
    test_id::Integer;
    metric::Symbol=:rmse,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=true,
)
    check_metric(metric)
    _require_cols(df, [:noise_spec_tag, :noise_spec_order], "plot_multi_track_training_noise_panels")

    sub_all = df[df.test_id .== test_id, :]
    if isempty(sub_all)
        available = join(["$(r.test_id) ($(r.test_name))" for r in eachrow(unique(df, :test_id))], ", ")
        throw(ArgumentError("plot_multi_track_training_noise_panels: no rows with \
                             test_id = $test_id. Available: $available"))
    end
    test_name = first(sub_all.test_name)

    # ----- Determine panel order (the order the caller declared the specs in) -----
    spec_order_map = Dict{Any,Int}()
    for row in eachrow(sub_all)
        spec_order_map[row.noise_spec_tag] = row.noise_spec_order
    end
    specs = sort(unique(sub_all.noise_spec_tag), by=s -> spec_order_map[s])
    n_panels = length(specs)

    base_df = sub_all[sub_all.train_set .== "Base", :]          # baseline (no training)
    trained_df = sub_all[sub_all.train_set .!= "Base", :]       # all trained steps

    fig = Figure(size=(450 * n_panels, 450))
    axs = Axis[]
    legend_ax = nothing

    for (idx, spec) in enumerate(specs)
        sub = trained_df[trained_df.noise_spec_tag .== spec, :]
        isempty(sub) && continue

        ax = Axis(fig[1, idx];
            title=string(spec),
            xlabel="Training tracks accumulated",
            ylabel=metric_label(metric),
            xgridvisible=false)
        push!(axs, ax)

        # Baseline first, so it leads the legend, named and coloured from the dataframe's
        # own baseline rows rather than a literal here -- same reasoning as the sibling
        # figure. It is drawn per panel even though it is one number: it is what each
        # panel's boxes are read against.
        base_rows = base_df[base_df.noise_spec_tag .== spec, :]
        if !isempty(base_rows)
            hlines!(ax, [first(base_rows[:, metric])];
                color=method_color(first(base_rows.estimator)),
                linestyle=:dash, linewidth=3, label=first(base_rows.estimator))
        end

        _grouped_boxplot!(ax, sub, metric;
            group_col=:train_set_order, group_order_col=:train_set_order,
            show_outliers=show_outliers, show_points=show_points)

        legend_ax = ax
    end

    # ----- Share one y scale across panels -----
    # Same metric on the same test track, so a panel that autoscaled on its own would draw
    # its boxes at the same height as a panel whose errors are an order of magnitude
    # larger -- which is exactly the difference this figure is about.
    if length(axs) > 1
        linkyaxes!(axs...)
        for ax in axs[2:end]
            hideydecorations!(ax; grid=false)
        end
    end

    Label(fig[0, 1:n_panels], "Tested on $(test_name)"; fontsize=17, font=:bold)
    isnothing(legend_ax) || Legend(fig[2, 1:n_panels], legend_ax;
        orientation=:horizontal, tellwidth=false)

    isnothing(save_path) || save(save_path, fig)
    return fig
end
