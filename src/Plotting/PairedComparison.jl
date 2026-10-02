"""
Paired per-trial comparison against a baseline. Every sweep runs the same trials through
every estimator, so each box shows per-trial changes relative to the reference on that
same trial: walk-to-walk difficulty cancels, and the reference itself is the zero line.
Summaries are descriptive (median, bootstrap interval, win count), not significance tests.
"""

using Random

"""
    plot_train_ratio_paired_relative_change(paired, dataset_name; value_col=:rel_change_pct,
        metric=:rmse, save_path=nothing, show_outliers=true, show_points=false,
        series_colors=nothing)

Grouped boxplots of a `paired_estimator_contrast` frame, one group per `train_ratio`.
`value_col` is `:rel_change_pct` or `:delta`; `metric` only names the axis and must
match the one the contrast was computed with (not checked).
"""
function plot_train_ratio_paired_relative_change(
    paired::DataFrame,
    dataset_name::AbstractString;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
    series_colors::Union{Nothing,AbstractDict}=nothing,
)
    check_metric(metric)
    value_col in (:delta, :rel_change_pct) || throw(ArgumentError(
        "value_col must be :delta or :rel_change_pct, got :$value_col"))

    sub = paired[paired.dataset_name .== dataset_name, :]
    isempty(sub) && error("No rows found for dataset_name = $dataset_name")

    # Pairing is still one-to-one per (trial, noise spec, seed), but a box here
    # groups on train_ratio only, so several noise specs/draws would be pooled
    # into one box without saying so. Say so.
    for splitter in (:noise_spec_tag, :seed)
        hasproperty(sub, splitter) || continue
        n = length(unique(sub[!, splitter]))
        n > 1 && @warn "plot_train_ratio_paired_relative_change: pooling $n `$splitter` \
                        values into each box; filter first if that is not intended."
    end

    as_pct = value_col === :rel_change_pct
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Ground truth available online [%]",
        ylabel=as_pct ? rich("relative change in ", metric_symbol(metric), " [%]") :
               rich("change in ", metric_label(metric)),
        subtitlesize=10,
        xticklabelsize=14,
    )
    as_pct && (ax.ytickformat = vs -> [string(round(v; digits=1), "%") for v in vs])

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1)
    labeled = _grouped_boxplot!(ax, sub, value_col;
        group_col=:train_ratio, group_order_col=:train_ratio_order,
        series_colors=series_colors,
        show_outliers=show_outliers, show_points=show_points)

    # `_grouped_boxplot!` labels the groups with the raw Float64; a percentage
    # reads better against an axis titled "ground truth available".
    ratio_order = Dict(r.train_ratio => r.train_ratio_order for r in eachrow(sub))
    ratios = sort(unique(sub.train_ratio), by=r -> ratio_order[r])
    ax.xticks = (1:length(ratios), ["$(round(Int, 100r))%" for r in ratios])

    if !isempty(labeled)
        Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false)
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end

"""
    plot_dataset_paired_relative_change(paired; value_col=:rel_change_pct, metric=:rmse,
        reference_label="baseline", save_path=nothing, show_outliers=true,
        show_points=false, show_subtitle=true)

As [`plot_train_ratio_paired_relative_change`](@ref), one group per `dataset_name`.
`series_colors` overrides [`method_color`](@ref) for variant series names.
"""
function plot_dataset_paired_relative_change(
    paired::DataFrame;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
)
    check_metric(metric)
    value_col in (:delta, :rel_change_pct) || throw(ArgumentError(
        "value_col must be :delta or :rel_change_pct, got :$value_col"))
    isempty(paired) && error("No rows to plot")

    # A box here groups on dataset only, so several train ratios / noise specs /
    # draws would be pooled into one box without saying so. Say so.
    for splitter in (:train_ratio, :noise_spec_tag, :seed)
        hasproperty(paired, splitter) || continue
        n = length(unique(paired[!, splitter]))
        n > 1 && @warn "plot_dataset_paired_relative_change: pooling $n `$splitter` \
                        values into each box; filter first if that is not intended."
    end

    as_pct = value_col === :rel_change_pct
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Dataset",
        ylabel=as_pct ? rich("relative change in ", metric_symbol(metric), " [%]") :
               rich("change in ", metric_label(metric)),
        xticklabelsize=14,
    )
    as_pct && (ax.ytickformat = vs -> [string(round(v; digits=1), "%") for v in vs])

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1)
    labeled = _grouped_boxplot!(ax, paired, value_col;
        group_col=:dataset_name, group_order_col=:dataset_order,
        show_outliers=show_outliers, show_points=show_points)

    if !isempty(labeled)
        Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false)
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end

"""
    plot_learning_curve_relative_change(paired, dataset_name; value_col=:rel_change_pct,
        metric=:rmse, series_colors=nothing, save_path=nothing,
        show_outliers=true, show_points=false)

As [`plot_train_ratio_paired_relative_change`](@ref), one group per mocap budget, from a
`learning_curve_contrast` frame (every group scored on the same strides).
"""
function plot_learning_curve_relative_change(
    paired::DataFrame,
    dataset_name::AbstractString;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse,
    series_colors::Union{Nothing,AbstractDict}=nothing,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
)
    check_metric(metric)
    value_col in (:delta, :rel_change_pct) || throw(ArgumentError(
        "value_col must be :delta or :rel_change_pct, got :$value_col"))

    sub = paired[paired.dataset_name .== dataset_name, :]
    isempty(sub) && error("No rows found for dataset_name = $dataset_name")

    as_pct = value_col === :rel_change_pct
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Online training strides",
        ylabel=as_pct ? rich("relative change in ", metric_symbol(metric), " [%]") :
               rich("change in ", metric_label(metric)),
        subtitlesize=10,
        xticklabelsize=14,
    )
    as_pct && (ax.ytickformat = vs -> [string(round(v; digits=1), "%") for v in vs])

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1)
    labeled = _grouped_boxplot!(ax, sub, value_col;
        group_col=:train_strides, group_order_col=:train_strides_order,
        series_colors=series_colors,
        show_outliers=show_outliers, show_points=show_points)

    # A budget longer than a trial's pre-split prefix is skipped rather than clamped, so
    # the wide end of the axis can rest on fewer trials than the narrow end. Say how many.
    budget_order = Dict(r.train_strides => r.train_strides_order for r in eachrow(sub))
    budgets = sort(unique(sub.train_strides), by=b -> budget_order[b])
    n_trials = Dict(b => length(unique(sub[sub.train_strides .== b, :trial_id])) for b in budgets)
    ax.xticks = (1:length(budgets), ["$b\n(n=$(n_trials[b]))" for b in budgets])

    if !isempty(labeled)
        Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false)
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end

"""
    plot_learning_curve_absolute(df, dataset_name; metric=:rmse,
        series_colors=nothing, save_path=nothing, show_outliers=true, show_points=false)

Unpaired learning curve on the metric's own scale from a `run_online_learning_curve`
frame: one group per budget, every estimator including `"ZUPT only"`. With `_ylims`,
marks beyond the limits get an arrowhead at the frame.
"""
function plot_learning_curve_absolute(
    df::DataFrame,
    dataset_name::AbstractString;
    metric::Symbol=:rmse,
    series_colors::Optional{AbstractDict}=nothing,
    save_path::Optional{String}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
    _ylims::Optional{Tuple{Float64,Float64}}=nothing
)
    check_metric(metric)
    groups = df[df.dataset_name .== dataset_name, :]
    isempty(groups) && error("No rows found for dataset_name = $dataset_name")

    groups.group = string.(groups.train_strides)
    groups.group_order = groups.train_strides_order

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Online training strides",
        ylabel=metric_label(metric),
        xticklabelsize=14,
    )
    labeled = _grouped_boxplot!(ax, groups, metric;
        group_col=:group, group_order_col=:group_order,
        series_colors=series_colors,
        show_outliers=show_outliers, show_points=show_points, clip_lims=_ylims)

    # As in `plot_learning_curve_relative_change`: the wide budgets can rest on fewer trials.
    order = Dict(r.group => r.group_order for r in eachrow(groups))
    labels = sort(unique(groups.group), by=g -> order[g])
    n_trials = Dict(g => length(unique(groups[groups.group .== g, :trial_id])) for g in labels)
    ax.xticks = (1:length(labels), ["$g\n(n=$(n_trials[g]))" for g in labels])

    if !isempty(labeled)
        Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false)
    end
    if !isnothing(_ylims)
        ylims!(ax, _ylims[1], _ylims[2])
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end
