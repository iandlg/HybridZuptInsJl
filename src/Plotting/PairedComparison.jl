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
"""
function plot_paired_relative_change(df::DataFrame;
    metric::Symbol=:rmse_rate,
    baseline::AbstractString,
    group::Union{Symbol,Nothing}=nothing,
    train_ratio::Union{Real,Nothing}=nothing,
    save_path::Union{String,Nothing}=nothing,
    level::Real=0.95,
    label_trials::Bool=true,
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
        ax.subtitle = join(annotations, "\n")
        ax.subtitlesize = 10
        # labels hang to the right of the last column; leave them room
        # label_trials && xlims!(ax, 0.5, length(others) + 0.8)
        push!(axs, ax)
    end

    # Panels stay separate (own title, own x ticks, own subtitle) but share one
    # y scale: autoscaled independently, a -5% column in one dataset can be drawn
    # at the same height as a -30% column in the other, which is precisely the
    # false equivalence this figure exists to avoid.
    length(axs) > 1 && linkyaxes!(axs...)

    Label(fig[0, :],
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
