"""
Paired per-trial comparison against a baseline.

Every sweep in `scripts/5Results/` runs the *same* trials through every
estimator, which makes the design paired. Plotting the result as side-by-side
boxplots throws that pairing away and asks the reader to compare two clouds of
9-10 points by eye, which is exactly the situation where an eyeball comparison
is least reliable.

The paired view instead shows, per trial, the change the estimator produced
relative to the baseline on that same trial. Trial-to-trial difficulty (some
walks are simply longer or twistier) cancels, so the remaining spread is the
effect of the estimator rather than the spread of the dataset.

The summary reported is deliberately descriptive -- median difference, a
bootstrap interval, and a win count -- not a significance test. At n = 9 the
honest statement is "lower in 8 of 9 trials, median -0.004 m/m", and this
figure is built to let you write exactly that sentence.
"""

using Random

"""
    median_bootstrap_ci(x; level=0.95, n_boot=10_000, rng) -> (lo, hi)

Percentile bootstrap interval for the median of `x`. Returns `(NaN, NaN)` for
fewer than 3 samples, where an interval would be theatre rather than
information.
"""
function median_bootstrap_ci(x::AbstractVector{<:Real};
    level::Real=0.95, n_boot::Int=10_000,
    rng::Random.AbstractRNG=Random.Xoshiro(0xC0FFEE))
    n = length(x)
    n < 3 && return (NaN, NaN)
    boot = Vector{Float64}(undef, n_boot)
    idx = Vector{Int}(undef, n)
    for b in 1:n_boot
        rand!(rng, idx, 1:n)
        boot[b] = median(@view x[idx])
    end
    α = (1 - level) / 2
    return (quantile(boot, α), quantile(boot, 1 - α))
end


"""
    function plot_train_ratio_paired_relative_change(
        paired::DataFrame,
        dataset_name::AbstractString;
        value_col::Symbol=:rel_change_pct,
        metric::Symbol=:rmse,
        save_path::Union{String,Nothing}=nothing,
        show_outliers::Bool=true,
        show_points::Bool=false,
    )

`plot_noise_paired_relative_change` with `train_ratio` on the x axis instead of the
noise spec: same grouped-boxplot machinery, same pairing, but each group is one
ground-truth availability level rather than one noise realisation.

Each point is one trial's change against the reference estimator on the **same**
trial at the **same** `train_ratio` (from `paired_estimator_contrast`), so
walk-to-walk difficulty cancels and a box clear of zero is a consistent effect —
the claim `plot_corrector_boxplots` cannot make, because there the trial-to-trial
spread swamps the estimator difference. The reference estimator has no box: it is
the zero line.

# Arguments
- `paired`: output of `paired_estimator_contrast`.
- `dataset_name`: which `dataset_name` to filter to.
- `value_col`: `:rel_change_pct` (default) or `:delta` (the metric's own units).
- `metric`: only used to name the quantity in the axis label — it must be the one
  `paired_estimator_contrast` was called with, which is not checked.
- `show_points`: overlay the individual trials on each box. Worth turning on here:
  a box typically summarises ~10 trials.
- `show_subtitle`: draw the subtitle under the title (default `true`). Turn it off
  when the document's own caption carries the same information.
"""
function plot_train_ratio_paired_relative_change(
    paired::DataFrame,
    dataset_name::AbstractString;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse,
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
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
    plot_dataset_paired_relative_change(
        paired::DataFrame;
        value_col::Symbol=:rel_change_pct,
        metric::Symbol=:rmse,
        reference_label::AbstractString="baseline",
        save_path::Union{String,Nothing}=nothing,
        show_outliers::Bool=true,
        show_points::Bool=false,
        show_subtitle::Bool=true,
    )

`plot_train_ratio_paired_relative_change` with the dataset on the x axis: one group
per `dataset_name`, estimators side by side within it, so a frozen hyperparameter
set can be read across datasets in the same visual language as the noise and
train-ratio figures.

Each box spans the per-trial changes against the reference estimator on the **same**
trial (from `paired_estimator_contrast`), so walk-to-walk difficulty cancels and a
box clear of zero is a consistent effect. The reference estimator has no box: it is
the zero line.

# Arguments
- `paired`: output of `paired_estimator_contrast`.
- `value_col`: `:rel_change_pct` (default) or `:delta` (the metric's own units).
- `metric`: only used to name the quantity in the axis label — it must be the one
  `paired_estimator_contrast` was called with, which is not checked.
- `show_points`: overlay the individual trials on each box. Worth turning on here:
  a box typically summarises ~10 trials.
- `show_subtitle`: draw the subtitle under the title (default `true`). Turn it off
  when the document's own caption carries the same information.
"""
function plot_dataset_paired_relative_change(
    paired::DataFrame;
    value_col::Symbol=:rel_change_pct,
    metric::Symbol=:rmse,
    reference_label::AbstractString="baseline",
    save_path::Union{String,Nothing}=nothing,
    show_outliers::Bool=true,
    show_points::Bool=false,
    show_subtitle::Bool=true,
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
        title="Per-trial change vs \"$reference_label\"",
        subtitle=(!show_subtitle ? "" :
                  as_pct ? "(estimator − $reference_label) / |$reference_label|, per trial" :
                  "estimator − $reference_label, per trial"),
        subtitlesize=10,
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
