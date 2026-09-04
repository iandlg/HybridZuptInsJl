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
    plot_paired_relative_change(df; metric, baseline, group=nothing, train_ratio=nothing,
                                save_path=nothing, level=0.95)

Per-trial *relative* change in `metric` between each estimator and `baseline`, in
percent of the baseline — not a raw difference, hence the name.

`df` is the long DataFrame produced by `run_online_correction_sweep`, requiring
columns `trial_id`, `estimator` and `metric`. `group` optionally facets by
another column (`:dataset_name` is the usual one). If `df` spans several
`train_ratio` values, pass `train_ratio` to pin one, otherwise the rows would
not be paired one-to-one.

Negative values mean the estimator beat the baseline on that trial. Each
estimator's column shows every trial as a point labelled with its `trial_id`
(set `label_trials=false` to drop the labels), the median as a thick bar, and
a percentile-bootstrap interval for that median; the subtitle carries the win
count and n.

`show_subtitle=false` drops that subtitle and the explanatory banner above the
panels, for a figure whose caption in the document says the same thing.
"""
function plot_paired_relative_change(df::DataFrame;
    metric::Symbol=:rmse_rate,
    baseline::AbstractString,
    group::Union{Symbol,Nothing}=nothing,
    train_ratio::Union{Real,Nothing}=nothing,
    save_path::Union{String,Nothing}=nothing,
    level::Real=0.95,
    label_trials::Bool=true,
    show_subtitle::Bool=true,
    figsize::Tuple{Int,Int}=(1000, 520))

    check_metric(metric)
    work = copy(df)

    if !isnothing(train_ratio)
        work = work[work.train_ratio .== train_ratio, :]
    elseif hasproperty(work, :train_ratio) && length(unique(work.train_ratio)) > 1
        throw(ArgumentError(
            "df spans train_ratio values $(sort(unique(work.train_ratio))); " *
            "pass train_ratio= to pin one, otherwise trials are not paired."))
    end

    baseline in work.estimator ||
        throw(ArgumentError("baseline \"$baseline\" not found; have $(unique(work.estimator))"))

    # This figure pairs on `trial_id` alone, so a frame carrying replicates --
    # several noise specs, or the repeated noise draws added to
    # `run_online_correction_sweep` -- has more than one row per (trial,
    # estimator) and the Dict below would silently keep whichever row came last.
    # Refuse rather than plot a comparison of mismatched rows.
    for splitter in (:noise_spec_tag, :seed)
        hasproperty(work, splitter) || continue
        length(unique(work[!, splitter])) > 1 || continue
        throw(ArgumentError(
            "df spans $(length(unique(work[!, splitter]))) `$splitter` values, so trials are " *
            "not paired one-to-one. Filter to a single value first, or use " *
            "`paired_estimator_contrast` + `plot_noise_paired_relative_change`, which " *
            "pair on (trial, noise spec, noise draw)."))
    end

    groups = isnothing(group) ? ["All"] : unique(work[!, group])
    others = filter(!=(baseline), unique(work.estimator))
    isempty(others) && throw(ArgumentError("nothing to compare against the baseline"))

    fig = Figure(size=figsize)
    colors = Makie.wong_colors()
    axs = Axis[]

    for (gi, gname) in enumerate(groups)
        sub = isnothing(group) ? work : work[work[!, group] .== gname, :]

        base_rows = sub[sub.estimator .== baseline, :]
        base_by_trial = Dict(zip(base_rows.trial_id, base_rows[!, metric]))

        ax = Axis(fig[1, gi];
            title=string(gname),
            ylabel=gi == 1 ? "relative change in $(metric_quantity(metric))  (estimator − baseline) / |baseline|  [%]" : "",
            xticks=(1:length(others), others),
            xticklabelrotation=π / 6,
            ytickformat=vs -> [string(round(v; digits=1), "%") for v in vs])
        hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1)

        annotations = String[]
        for (ei, est) in enumerate(others)
            rows = sub[sub.estimator .== est, :]
            # pair strictly by trial: a trial missing from either arm is dropped
            # Denominator is the baseline, matching the axis label: dividing by the
            # estimator's own value (as this did) makes the same absolute
            # improvement read differently depending on which estimator produced it.
            paired = [(r.trial_id, 100 * (r[metric] - base_by_trial[r.trial_id]) / abs(base_by_trial[r.trial_id]))
                      for r in eachrow(rows) if haskey(base_by_trial, r.trial_id)]
            isempty(paired) && continue
            deltas = [d for (_, d) in paired]
            n = length(deltas)
            wins = count(<(0), deltas)
            med = median(deltas)
            lo, hi = median_bootstrap_ci(deltas; level=level)

            x = fill(float(ei), n) .+ (rand(Random.Xoshiro(ei), n) .- 0.5) .* 0.18
            scatter!(ax, x, deltas; color=(colors[mod1(ei+1, length(colors))], 0.75), markersize=9)
            if label_trials
                # pixel offset so the label clears the marker regardless of zoom
                text!(ax, x, deltas; text=[string(t) for (t, _) in paired],
                    align=(:left, :center), offset=(7, 0), fontsize=8,
                    color=(:black, 0.7))
            end
            if isfinite(lo)
                rangebars!(ax, [float(ei)], [lo], [hi];
                    color=:black, linewidth=2, whiskerwidth=12)
            end
            lines!(ax, [ei - 0.28, ei + 0.28], [med, med]; color=:black, linewidth=3)

            push!(annotations, "$est: better on $wins/$n, median $(round(med, sigdigits=3))%")
        end
        if show_subtitle
            ax.subtitle = join(annotations, "\n")
            ax.subtitlesize = 10
        end
        # labels hang to the right of the last column; leave them room
        # label_trials && xlims!(ax, 0.5, length(others) + 0.8)
        push!(axs, ax)
    end

    # Panels stay separate (own title, own x ticks, own subtitle) but share one
    # y scale: autoscaled independently, a -5% column in one dataset can be drawn
    # at the same height as a -30% column in the other, which is precisely the
    # false equivalence this figure exists to avoid.
    length(axs) > 1 && linkyaxes!(axs...)

    # Figure-level caption, gated by the same flag: with the per-estimator
    # subtitles off, a lone explanatory banner is the thing a thesis caption
    # would repeat.
    show_subtitle && Label(fig[0, :],
        "Paired per-trial relative change vs \"$baseline\", in %. " *
        "Bars: median and $(round(Int, 100 * level))% bootstrap interval.";
        fontsize=12, tellwidth=false)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end

"""
    function plot_train_ratio_paired_relative_change(
        paired::DataFrame,
        dataset_name::AbstractString;
        value_col::Symbol=:rel_change_pct,
        metric::Symbol=:rmse,
        reference_label::AbstractString="baseline",
        save_path::Union{String,Nothing}=nothing,
        show_outliers::Bool=true,
        show_points::Bool=false,
        show_subtitle::Bool=true,
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

`plot_paired_relative_change` shows the same contrast at a *single* train ratio in
more detail (per-trial labels, median with a bootstrap interval); this one trades
that detail for the sweep across ratios.

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
        xlabel="Ground truth available online (train_ratio)",
        ylabel=as_pct ? "relative change in $(metric_quantity(metric)) [%]" :
               "change in $(metric_label(metric))",
        title="Per-trial change vs \"$reference_label\" — $dataset_name",
        subtitle=(!show_subtitle ? "" :
                  as_pct ? "(estimator − $reference_label) / |$reference_label|, per trial" :
                  "estimator − $reference_label, per trial"
    ),
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
