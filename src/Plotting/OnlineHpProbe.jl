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
    plot_probe_ranking(df; save_path=nothing, figsize=(940, 560), max_label=nothing)

Which parameters move RMSE, by how much, and whether the trials agree. The
overview figure, and the one to read first.

One row per parameter, sorted by the span of the across-trial median curve,
largest at the top. The bar is that median range; the thin whisker is the
across-trial inter-quartile range at the probe where the median moved furthest;
the right-hand column reports how many trials agreed on the sign there, starred
when an exact sign test puts it below 0.05.

**The x limits are set from the bars, not from the data.** This is the whole
point of the figure. Drawing every trial's own extremes, as the previous version
did, let a handful of trials set the scale -- on the 11-trial ANG2 sweep they run
to +458% while all fifteen median ranges fit inside -4% to +39%, so every bar
collapsed to at most 8% of the plot width and the ranking was unreadable. A
whisker that runs past the frame is clipped and marked with an arrowhead, so
"this parameter's trials disagree wildly" is still visible without those trials
dictating the axis.

Agreement is carried alongside magnitude because the two answer different
questions and a wide bar alone answers neither: the pipeline is deterministic,
so a large span with 6/11 agreement is eleven trials disagreeing, not an effect.
"""
function plot_probe_ranking(df::DataFrame;
    save_path::Union{String,Nothing}=nothing,
    figsize::Tuple{Int,Int}=(940, 560))

    work = df[df.parameter.!="baseline", :]
    isempty(work) && throw(ArgumentError("plot_probe_ranking: no swept rows in frame"))
    work = copy(work)
    work.pct = 100 .* float.(work.relative_change)

    agree = probe_agreement(df)            # already sorted by span, descending
    med = combine(groupby(work, [:parameter, :probe]), :pct => median => :med)
    bars = combine(groupby(med, :parameter),
        :med => minimum => :lo, :med => maximum => :hi)
    # `leftjoin` does not preserve row order, so the descending-span order
    # `probe_agreement` established has to be restored explicitly -- without this
    # the rows come back in sweep order and the figure is not a ranking at all.
    g = sort!(leftjoin(agree, bars; on=:parameter), :span; rev=true)
    n = nrow(g)

    # Axis limits from the bars alone, padded. Whiskers may exceed these; the
    # per-trial extremes certainly will, and neither is allowed to set the scale.
    lo = min(0.0, minimum(g.lo))
    hi = max(0.0, maximum(g.hi))
    pad = 0.15 * max(hi - lo, eps())
    xlo, xhi = lo - pad, hi + pad

    fig = Figure(size=figsize)
    # Largest span at the top: Makie's y increases upward, so row i of a
    # descending-sorted frame is drawn at n - i + 1.
    ypos(i) = n - i + 1
    ax = Axis(fig[1, 1];
        xlabel="RMSE change vs baseline [%]",
        xtickformat=_HP_PCT_TICKFORMAT,
        yticks=(1:n, [hp_param_label(g.parameter[ypos(i)]) for i in 1:n]),
        title="Sensitivity ranking: signed range of the across-trial median",
        xgridstyle=:dash, ygridvisible=false)
    vlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)

    for i in 1:n
        r = g[i, :]
        y = ypos(i)
        c = hp_param_color(r.parameter)
        # Whisker first, so the bar reads on top of it.
        wlo, whi = clamp(r.q25, xlo, xhi), clamp(r.q75, xlo, xhi)
        lines!(ax, [wlo, whi], [y, y]; color=(c, 0.5), linewidth=1.5)
        for (v, cl, mk) in ((r.q25, wlo, :ltriangle), (r.q75, whi, :rtriangle))
            isapprox(v, cl; rtol=1e-9) && continue
            scatter!(ax, [cl], [y]; color=(c, 0.7), marker=mk, markersize=9)
        end
        lines!(ax, [r.lo, r.hi], [y, y]; color=c, linewidth=7)
    end
    xlims!(ax, xlo, xhi)
    ylims!(ax, 0.5, n + 0.5)

    # Agreement column: its own axis so the text sits outside the data frame.
    ax_a = Axis(fig[1, 2]; title="agree", titlesize=11, titlegap=6)
    hidedecorations!(ax_a)
    hidespines!(ax_a)
    xlims!(ax_a, 0, 1)
    ylims!(ax_a, 0.5, n + 0.5)
    for i in 1:n
        r = g[i, :]
        star = r.p_value < 0.05 ? "*" : ""
        text!(ax_a, 0.5, ypos(i); text="$(r.n_agree)/$(r.n_trials)$star",
            align=(:center, :center), fontsize=10,
            font=(r.p_value < 0.05 ? :bold : :regular))
    end
    colsize!(fig.layout, 2, Fixed(56))

    types = _hp_ordered_types(unique(g.type))
    Legend(fig[1, 3],
        [PolyElement(color=hp_type_color(t)) for t in types],
        [_hp_type_label(t) for t in types];
        tellheight=false)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
    end
    return fig
end

"""
    plot_param_closeup(df, parameter; box_df=nothing, show_trials=true,
                       save_path=nothing, figsize=(620, 440))

One parameter, one figure, one conclusion -- the per-parameter extract of the
`plot_probe_sensitivity` grid, sized for the write-up rather than for an
appendix page.

Across-trial median with its inter-quartile band, faint per-trial lines beneath
it when `show_trials`, and a second axis on top carrying the parameter's
*absolute* values: a reader needs to know that ×2.15 on the yaw length scale
means 31.6, not only that it is ×2.15.

Pass `box_df` (from [`box_exit_over_trials`](@ref)) for a normalisation
parameter and the probe intervals where the features leave the fixed `±LL`
domain are shaded, so "the damage begins exactly where the basis stops being
able to represent the feature" is read off the figure rather than argued. When
no such interval falls inside the plotted range the subtitle says so instead of
the figure simply lacking shading -- for the location parameters the boundary is
at roughly ±5 and the probe only reaches ±2, and that the probe never leaves the
domain is the result.
"""
function plot_param_closeup(df::DataFrame, parameter::AbstractString;
    box_df::Union{DataFrame,Nothing}=nothing,
    show_trials::Bool=true,
    save_path::Union{String,Nothing}=nothing,
    max_ticks::Int=5,
    figsize::Tuple{Int,Int}=(700, 460))

    sub = df[df.parameter.==parameter, :]
    isempty(sub) && throw(ArgumentError(
        "plot_param_closeup: \"$parameter\" is not in the frame. Available: " *
        join(sort(unique(df.parameter[df.parameter.!="baseline"])), ", ")))

    kind = first(sub.probe_kind)
    color = hp_param_color(parameter)
    probes, med, qlo, qhi, n_trials = _probe_summary(sub)

    fig = Figure(size=figsize)
    ax = Axis(fig[2, 1];
        xlabel=_probe_xlabel(kind, first(sub.type)),
        ylabel="RMSE change vs baseline [%]",
        xscale=(kind == "multiplicative" ? log10 : identity),
        ytickformat=_HP_PCT_TICKFORMAT,
        xgridstyle=:dash, ygridstyle=:dash)

    # Out-of-box shading goes down first, under everything else.
    shaded = Tuple{Float64,Float64}[]
    if !isnothing(box_df) && parameter in box_df.parameter
        for (a, b) in box_outside_spans(box_df, parameter)
            # Only the part that overlaps the RMSE probe range is drawable.
            lo, hi = max(a, minimum(probes)), min(b, maximum(probes))
            lo < hi || continue
            push!(shaded, (lo, hi))
            vspan!(ax, lo, hi; color=(:firebrick, 0.12))
            vlines!(ax, a >= minimum(probes) ? a : b;
                color=(:firebrick, 0.6), linestyle=:dash, linewidth=1.5)
        end
    end

    if show_trials && hasproperty(sub, :trial_id)
        for t in groupby(sub, :trial_id)
            o = sortperm(t.probe)
            lines!(ax, t.probe[o], 100 .* float.(t.relative_change[o]);
                color=(color, 0.18), linewidth=1)
        end
    end

    hlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)
    vlines!(ax, _probe_identity(kind); color=:gray, linestyle=:dot, linewidth=1)
    n_trials > 1 && band!(ax, probes, qlo, qhi; color=(color, 0.25))
    lines!(ax, probes, med; color=color, linewidth=2.5)
    scatter!(ax, probes, med; color=color, markersize=9)
    ax.xticks = _probe_ticks(probes, kind; max_ticks=max_ticks)

    # Y limits from the band, not from the per-trial lines. Same reasoning as
    # plot_probe_ranking's x limits: on this sweep single trials reach +458%
    # while the IQR tops out near +80%, so letting the data set the range
    # flattens the median curve the figure exists to show. Trials outside the
    # frame are clipped; the band still states how wide the spread is.
    ylo = min(0.0, minimum(qlo), minimum(med))
    yhi = max(0.0, maximum(qhi), maximum(med))
    ypad = 0.12 * max(yhi - ylo, eps())
    ylims!(ax, ylo - ypad, yhi + ypad)

    # Absolute parameter values on a linked top axis. `tested_value` already
    # holds them, so the mapping needs no base value or probe unit.
    abs_at = Dict(round(r.probe; digits=9) => r.tested_value for r in eachrow(sub))
    tickpos = _probe_ticks(probes, kind; max_ticks=max_ticks)[1]
    ax_top = Axis(fig[2, 1];
        xscale=(kind == "multiplicative" ? log10 : identity),
        xaxisposition=:top, xgridvisible=false, ygridvisible=false,
        xticks=(tickpos, [string(round(abs_at[round(t; digits=9)]; sigdigits=3))
                          for t in tickpos]),
        xlabel="absolute value")
    hidespines!(ax_top)
    hideydecorations!(ax_top)
    linkxaxes!(ax, ax_top)

    # Caption line: what the reader needs to size the claim.
    ag = probe_agreement(df)
    a = ag[ag.parameter.==parameter, :]
    bits = String["baseline $(round(first(sub.base_value); sigdigits=4))"]
    n_trials > 1 && push!(bits, "$n_trials trials, median and IQR")
    if nrow(a) == 1
        star = a.p_value[1] < 0.05 ? ", p = $(round(a.p_value[1]; sigdigits=2))" : ""
        push!(bits, "$(a.n_agree[1])/$(a.n_trials[1]) agree at the largest effect$star")
    end
    if !isnothing(box_df) && parameter in box_df.parameter
        push!(bits, isempty(shaded) ?
                    "features stay inside ±LL over the whole probed range" :
                    "shaded: features outside ±LL")
    end
    # `tellwidth=false` on both: otherwise the caption -- which is long, and
    # longer still for a normalisation parameter carrying the containment note --
    # sets the column width, pushing the axis out of the frame until the negative
    # y-tick labels are clipped and the caption itself is truncated.
    Label(fig[1, 1], hp_param_label(parameter);
        fontsize=15, font=:bold, halign=:left, tellwidth=false)
    Label(fig[3, 1], join(bits, "  ·  ");
        fontsize=9, color=(:black, 0.65), halign=:left, tellwidth=false)
    rowgap!(fig.layout, 1, 2)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
    end
    return fig
end
