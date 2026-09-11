"""
Probe-aware sensitivity figures.

Companion to `OnlineHpSensitivity.jl`, which draws the same sweep against a
multiplier axis recovered as `tested_value / base_value`. That axis is only
meaningful for a *scale* parameter. This file reads the `probe` / `probe_kind`
columns `vary_hsgp_parameters` now emits, so a location parameter is drawn on the
additive axis it was actually swept on, and it draws the across-trial band that
`sweep_over_trials` produces.

The older file is left in place: `2_hyp_sensitivity-dataset_comparison.jl` and
`2_hyp_sensitivity-yaw_length_scale_regression.jl` still use it, and a
probe-aware panel is a different function rather than a flag on the old one.

Shared vocabulary -- colours, labels, percent tick format, multiplier ticks --
comes from `OnlineHpSensitivity.jl` rather than being restated here.
"""

"""
X-axis label per probe kind and family. A multiplier needs no unit; an offset is
only interpretable with the scale it is quoted in, which is the whole point of
the additive probe.
"""
const _PROBE_XLABELS = Dict{String,String}(
    "input_mean" => "offset [σₓ]",
    "input_center" => "offset [z]",
)
const _PROBE_MULT_XLABEL = "multiplier"
const _PROBE_OFFSET_XLABEL = "offset"

function _probe_xlabel(probe_kind::AbstractString, type::AbstractString)::String
    probe_kind == "multiplicative" && return _PROBE_MULT_XLABEL
    key = get(_STAT_TYPE_ALIASES, String(type), String(type))
    return get(_PROBE_XLABELS, key, _PROBE_OFFSET_XLABEL)
end

# The probe value that means "unperturbed": ×1 for a multiplier, 0 for an offset.
_probe_identity(probe_kind::AbstractString)::Float64 =
    probe_kind == "multiplicative" ? 1.0 : 0.0

"""
    _probe_ticks(probes, probe_kind; max_ticks=5) -> (positions, labels)

Labelled ticks for a probe axis. Multiplicative reuses the geometric thinning and
`×`-prefixed labels of the multiplier axis, so a point carries between the two
files' figures. Additive is linear and evenly thinned, always keeping the 0 tick.
"""
function _probe_ticks(probes::AbstractVector{<:Real}, probe_kind::AbstractString;
    max_ticks::Int=5)
    probe_kind == "multiplicative" && return _hp_multiplier_ticks(probes; max_ticks=max_ticks)
    pos = sort(unique(float.(probes)))
    n = length(pos)
    if n > max_ticks && max_ticks >= 2
        anchor = argmin(abs.(pos))              # the 0 tick, or the nearest to it
        stride = cld(n - 1, max_ticks - 1)
        idx = sort(unique(vcat(collect(anchor:(-stride):1), collect(anchor:stride:n))))
        for e in (1, n)
            if !(e in idx) && minimum(abs.(idx .- e)) > 1
                push!(idx, e)
            end
        end
        pos = pos[sort!(idx)]
    end
    labels = map(p -> (r = round(p; sigdigits=3); isinteger(r) ? string(Int(r)) : string(r)), pos)
    return (pos, labels)
end

"""
    _probe_summary(sub) -> (probes, med, lo, hi, n_trials)

Collapse one parameter's rows to an across-trial median and inter-quartile band
at each probe value. A sweep with a single trial returns a degenerate band, which
the drawing code then skips rather than shading a zero-width ribbon.
"""
function _probe_summary(sub::AbstractDataFrame)
    has_trials = hasproperty(sub, :trial_id)
    n_trials = has_trials ? length(unique(sub.trial_id)) : 1
    g = combine(groupby(sub, :probe),
        :relative_change => (v -> 100 * median(v)) => :med,
        :relative_change => (v -> 100 * quantile(v, 0.25)) => :lo,
        :relative_change => (v -> 100 * quantile(v, 0.75)) => :hi)
    sort!(g, :probe)
    return (g.probe, g.med, g.lo, g.hi, n_trials)
end

"""
    _draw_probe_panel!(ax, sub; color, markersize, linewidth, max_ticks) -> (probes, med)

One parameter's sensitivity curve on its own probe axis: across-trial median with
an inter-quartile band, percent RMSE change against `probe`.

Unlike `_draw_hp_panel!` this never divides by `base_value`, so it is defined for
a parameter fitted to zero and points the same way for a negative base as for a
positive one.
"""
function _draw_probe_panel!(ax::Axis, sub::AbstractDataFrame;
    color=Makie.wong_colors()[1], markersize::Real=6, linewidth::Real=2,
    max_ticks::Int=5)

    probes, med, lo, hi, n_trials = _probe_summary(sub)
    kind = first(sub.probe_kind)

    hlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)
    vlines!(ax, _probe_identity(kind); color=:gray, linestyle=:dot, linewidth=1)

    if n_trials > 1
        band!(ax, probes, lo, hi; color=(color, 0.22))
    end
    lines!(ax, probes, med; color=color, linewidth=linewidth)
    scatter!(ax, probes, med; color=color, markersize=markersize)

    ax.xticks = _probe_ticks(probes, kind; max_ticks=max_ticks)
    return probes, med
end

"""
    plot_probe_sensitivity(df, grid; save_path=nothing, max_ticks=5)

One panel per swept parameter, laid out by `grid`, each on its own probe axis:
`log10` multiplier for the scale families, linear offset for the locations.

This is the figure the typed probe was for. Under the old multiplicative sweep a
row of location panels shared an axis label that meant a different excursion in
every panel -- `μ_x[1]` at ×10 is a 13-standard-deviation shift and `μ_x[3]` at
×10 is less than one -- and the `c_x[3]` panel ran right to left because its base
is negative. Here every panel of a family is the same probe in the same units.
"""
function plot_probe_sensitivity(
    df::DataFrame, grid::ParamGrid;
    save_path::Union{String,Nothing}=nothing,
    max_ticks::Int=5)

    n_rows, n_cols = size(grid.specs)
    fig = Figure(size=(400 * n_cols, 400 * n_rows))

    for row in 1:n_rows, col in 1:n_cols
        spec = grid.specs[row, col]
        isnothing(spec) && continue
        sub = df[df.parameter.==spec.name, :]
        if isempty(sub)
            @warn "Missing data for $(spec.name)"
            continue
        end
        kind = first(sub.probe_kind)
        ax = Axis(fig[row, col];
            xlabel=_probe_xlabel(kind, first(sub.type)),
            xscale=(kind == "multiplicative" ? log10 : identity),
            ytickformat=_HP_PCT_TICKFORMAT,
            title=hp_param_label(spec.name),
            xgridstyle=:dash, ygridstyle=:dash)
        _draw_probe_panel!(ax, sub; color=hp_param_color(spec.name), max_ticks=max_ticks)
    end

    Label(fig[1:n_rows, 0], "RMSE change vs baseline [%]", rotation=π / 2, fontsize=14)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
    end
    return fig
end

"""
    plot_box_exit(df, box_df; save_path=nothing, max_ticks=5)

Where the normalised features leave the fixed HSGP domain, against what the same
perturbation does to RMSE.

One panel per input-normalisation parameter. Left axis: across-trial median RMSE
change with its inter-quartile band. Right axis: the fraction of strides with any
`|z_d| > LL_d`. A dashed rule marks where `max_z_ratio` first crosses 1.

The point of the figure: `LL` is a stored field, fixed at training time, and is
*not* rescaled when `σ_x` is perturbed. So the outer end of a `σ_x` sweep is not
a length-scale result, it is the features leaving a box that did not move -- and
the rule says at exactly which multiplier. A panel with no rule is the other
useful outcome: that parameter stayed inside the domain over the whole probe, so
whatever RMSE did there is sensitivity rather than a basis-truncation artifact.

`box_df` may be swept over a wider probe range than `df` -- it costs no filter
runs -- in which case the left axis is drawn over the RMSE range and the right
axis over the full box range.
"""
function plot_box_exit(df::DataFrame, box_df::DataFrame;
    save_path::Union{String,Nothing}=nothing, max_ticks::Int=5)

    params = sort(unique(box_df.parameter))
    n_cols = 3
    n_rows = cld(length(params), n_cols)
    fig = Figure(size=(420 * n_cols, 380 * n_rows))
    exits = box_exit_points(box_df)

    for (i, pname) in enumerate(params)
        row, col = fldmod1(i, n_cols)
        bsub = sort(box_df[box_df.parameter.==pname, :], :probe)
        rsub = df[df.parameter.==pname, :]
        kind = first(bsub.probe_kind)
        scale = kind == "multiplicative" ? log10 : identity
        color = hp_param_color(pname)

        ax = Axis(fig[row, col];
            xlabel=_probe_xlabel(kind, first(bsub.type)),
            ylabel="RMSE change [%]",
            xscale=scale, ytickformat=_HP_PCT_TICKFORMAT,
            title=hp_param_label(pname), xgridstyle=:dash, ygridstyle=:dash)
        ax_box = Axis(fig[row, col];
            ylabel="strides outside box", yaxisposition=:right,
            xscale=scale, ygridvisible=false, xgridvisible=false)
        hidespines!(ax_box)
        hidexdecorations!(ax_box)
        linkxaxes!(ax, ax_box)

        # Box occupancy first, so the RMSE curve draws over it.
        lines!(ax_box, bsub.probe, bsub.frac_outside;
            color=(:black, 0.55), linewidth=2, linestyle=:dashdot)
        ylims!(ax_box, -0.02, 1.02)

        if isempty(rsub)
            @warn "plot_box_exit: no RMSE rows for $pname; drawing occupancy only"
        else
            _draw_probe_panel!(ax, rsub; color=color, max_ticks=max_ticks)
        end

        # The crossings, one rule per direction that actually leaves the box.
        # The values go in the subtitle rather than beside the rules: at three
        # panels per row two labels collide with each other and with the curve,
        # and a data-space y for the text is meaningless when the panels' RMSE
        # ranges differ by an order of magnitude.
        esub = exits[exits.parameter.==pname, :]
        crossings = [r.exit_probe for r in eachrow(esub) if !ismissing(r.exit_probe)]
        for x in crossings
            vlines!(ax, x; color=:firebrick, linestyle=:dash, linewidth=2)
        end
        ax.subtitle = isempty(crossings) ? "inside box over the probed range" :
                      "leaves box at " * join(
            [(kind == "multiplicative" ? hp_multiplier_label(x) : string(round(x; sigdigits=3)))
             for x in sort(crossings)], " and ")
        ax.subtitlesize = 10
        ax.subtitlecolor = isempty(crossings) ? (:black, 0.55) : :firebrick

        # The span the RMSE sweep actually covers. The box grid is deliberately
        # wider -- it costs no filter runs -- so without this the reader cannot
        # tell whether a crossing was inside the measured range or extrapolated
        # past it, which is exactly the difference between the input std rows and
        # the location rows.
        if !isempty(rsub)
            vspan!(ax, minimum(rsub.probe), maximum(rsub.probe); color=(:steelblue, 0.10))
        end
        ax.xticks = _probe_ticks(bsub.probe, kind; max_ticks=max_ticks)
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
    end
    return fig
end

"""
    plot_probe_signed_range(df; sort_by=:span, save_path=nothing, figsize=(900, 560))

Signed range of RMSE change per parameter: the figure to read first.

The bar is the range of the across-trial *median* curve; the open circles are
each individual trial's own extremes. A parameter whose circles straddle the
bar's zero crossing, or whose trials disagree in sign, has not been shown to
matter however wide its bar is.

Two things make this rankable that were not before. Every probe is now sized by
something physical -- a decade for a scale, a standard deviation for a location
-- rather than by the parameter's own base value, so the rows are commensurate.
And it reports the signed *range* rather than `maximum(relative_change)`: the
signed max is 0 for any parameter that only ever improves RMSE, which rendered
the yaw length scale, whose true range is about [-60%, 0], as a zero-height bar
indistinguishable from a hyperparameter that never reached the code.
"""
function plot_probe_signed_range(df::DataFrame;
    sort_by::Symbol=:span,
    save_path::Union{String,Nothing}=nothing,
    figsize::Tuple{Int,Int}=(900, 560))

    sort_by in (:span, :min, :max) ||
        throw(ArgumentError("sort_by must be :span, :min or :max, got :$sort_by"))

    work = df[df.parameter.!="baseline", :]
    isempty(work) && throw(ArgumentError("plot_probe_signed_range: no swept rows in frame"))
    work = copy(work)
    work.pct = 100 .* float.(work.relative_change)
    has_trials = hasproperty(work, :trial_id)

    # The bar: range of the across-trial median curve.
    med = combine(groupby(work, [:parameter, :type, :probe]),
        :pct => median => :pct)
    g = combine(groupby(med, [:parameter, :type]),
        :pct => minimum => :lo, :pct => maximum => :hi)
    g.span = g.hi .- g.lo
    key = sort_by === :span ? :span : (sort_by === :min ? :lo : :hi)
    sort!(g, key; rev=(sort_by !== :min))

    # The circles: each trial's own extremes.
    per_trial = has_trials ?
                combine(groupby(work, [:parameter, :trial_id]),
        :pct => minimum => :lo, :pct => maximum => :hi) :
                DataFrame(parameter=String[], lo=Float64[], hi=Float64[])

    n = nrow(g)
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        xlabel="RMSE change vs baseline [%]",
        xtickformat=_HP_PCT_TICKFORMAT,
        yticks=(1:n, [hp_param_label(p) for p in g.parameter]),
        title="Signed sensitivity range over the tested probes",
        xgridstyle=:dash, ygridvisible=false)
    vlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)

    for (i, r) in enumerate(eachrow(g))
        c = hp_param_color(r.parameter)
        lines!(ax, [r.lo, r.hi], [i, i]; color=c, linewidth=6)
        if has_trials
            t = per_trial[per_trial.parameter.==r.parameter, :]
            pts = vcat(t.lo, t.hi)
            scatter!(ax, pts, fill(i, length(pts));
                color=:white, strokecolor=c, strokewidth=1.2, markersize=7)
        end
    end

    types = _hp_ordered_types(unique(g.type))
    Legend(fig[1, 2],
        [PolyElement(color=hp_type_color(t)) for t in types],
        [_hp_type_label(t) for t in types];
        tellheight=false)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
    end
    return fig
end
