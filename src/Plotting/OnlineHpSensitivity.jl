"""
Hyperparameter sensitivity curves.

Both the full grid ([`plot_hp_sensitivity`](@ref)) and the single-parameter
close-up ([`plot_hp_param_sensitivity`](@ref)) draw the same curve through
`_draw_hp_panel!`, so a panel of the grid and the close-up of the same parameter
cannot disagree about axes or units.

Two conventions, shared by both:

* **x is the multiplier applied to the tested hyperparameter**, on a log-scaled
  axis with a tick at each value actually swept. `vary_hsgp_parameters` samples
  geometrically around the base value (`log_around`), so the spacing is the
  sweep's own spacing; only the labels change, from exponents to `×0.32`, `×1`,
  `×3.16`. The axis previously read `log₁₀(relative change)`, which was doubly
  misleading: the quantity plotted was `log10(tested/base)`, a multiplier, and
  `relative_change` is the name of a different column entirely.
* **y is the RMSE change against the baseline, in percent** -- the
  `relative_change` column scaled by 100, so 0% is the untouched hyperparameter
  and −60% means the perturbation cut RMSE by well over half. The grid used to
  plot `rmse_ratio` (1.0 = baseline), which is the same information on a scale
  that has to be translated before it can be quoted.
"""

"""
Math symbols for the three hyperparameters, keyed by their position in a
channel's parameter vector (the same index `_HYPERPARAM_TYPES` is keyed by, and
the one that appears in a parameter name like `yaw[2]`).

Written as `(base, subscript)` pairs rather than plain strings because Unicode
has no subscript `f`, so `σ_f` cannot be spelled with combining characters the
way `σₙ` can. Makie's `rich`/`subscript` renders all three consistently.
"""
const _HP_SYMBOL_PARTS = Dict{Int,Tuple{String,String}}(
    1 => ("σ", "n"),   # noise
    2 => ("ℓ", "s"),   # length scale
    3 => ("σ", "f"),   # signal variance
)

"""
Hyperparameter type name -> its index in a channel's parameter vector. The
inverse of `_HYPERPARAM_TYPES`, so the mapping is stated once.
"""
const _HP_TYPE_INDEX = Dict{String,Int}(v => k for (k, v) in _HYPERPARAM_TYPES)

"""
One colour per hyperparameter type, fixed by type rather than by plotting order,
so ℓ_s is the same colour in every sensitivity figure regardless of which
parameters a given sweep happened to contain. Keyed by the index in a channel's
parameter vector (1 = σ_n, 2 = ℓ_s, 3 = σ_f).

These are Wong 4-6. Wong 1-3 are spoken for: they are what the estimator figures
in `scripts/5Results/` give to the three correction types (ZUPT only / Static /
HSGP), and reusing them here would put the same colour on a correction method in
one figure and a hyperparameter in the next. Wong 7 is the yellow, which is too
light to read as a thin line on the light background these figures use.
"""
const _HP_COLOR_INDICES = Dict{Int,Int}(
    1 => 4,   # σ_n  -> Wong 4, #CC79A7
    2 => 5,   # ℓ_s  -> Wong 5, #56B4E9
    3 => 6,   # σ_f  -> Wong 6, #D55E00
)

const _HP_FALLBACK_COLOR = Makie.RGBAf(0.45, 0.45, 0.45, 1.0)

"""
    hp_type_color(idx_or_type)

Colour for a hyperparameter type, given either its index (1/2/3) or its type
name (`"noise"` / `"length_scale"` / `"signal_variance"`). Anything else -- the
input/output statistics from `make_stats_param_grid` -- gets a neutral grey
rather than silently borrowing another type's colour.
"""
hp_type_color(idx::Int) =
    haskey(_HP_COLOR_INDICES, idx) ? Makie.wong_colors()[_HP_COLOR_INDICES[idx]] : _HP_FALLBACK_COLOR
hp_type_color(type::AbstractString) = hp_type_color(get(_HP_TYPE_INDEX, String(type), 0))

"""
    hp_param_color(name)

Colour for a sweep parameter such as `"yaw[2]"`, resolved through its
hyperparameter type so that every figure agrees. Same channel guard as
[`hp_param_label`](@ref).
"""
function hp_param_color(name::AbstractString)
    parsed = _parse_hp_param(name)
    isnothing(parsed) && return _HP_FALLBACK_COLOR
    channel, idx = parsed
    channel in _OUTPUT_NAMES || return _HP_FALLBACK_COLOR
    return hp_type_color(idx)
end

"""
    _parse_hp_param(name) -> (channel, idx) or nothing

Split a parameter name such as `"yaw[2]"` into its channel and hyperparameter
index. Returns `nothing` for anything that is not of that shape (`"baseline"`,
the input/output-statistic names from `make_stats_param_grid`), so callers can
fall back to showing the raw name.
"""
function _parse_hp_param(name::AbstractString)
    mt = match(r"^(.*)\[(\d+)\]$", name)
    isnothing(mt) && return nothing
    return (String(mt.captures[1]), parse(Int, mt.captures[2]))
end

"""
    hp_param_label(name; with_channel=true) -> rich text

Display label for a sweep parameter: the hyperparameter's math symbol, with the
output channel it belongs to, e.g. `yaw[2]` renders as `ℓ_s (yaw)`. Falls back to
the raw name for parameters outside the `channel[idx]` scheme.

The channel must be one of `_OUTPUT_NAMES`. `make_stats_param_grid` emits names
of the same *shape* -- `input_mean[1]`, `output_std[2]` -- where the bracketed
number is an input dimension, not a hyperparameter kind; without this check
those would be relabelled as σ_n and friends, which would be wrong rather than
merely ugly.
"""
function hp_param_label(name::AbstractString; with_channel::Bool=true)
    parsed = _parse_hp_param(name)
    isnothing(parsed) && return rich(String(name))
    channel, idx = parsed
    (channel in _OUTPUT_NAMES && haskey(_HP_SYMBOL_PARTS, idx)) || return rich(String(name))
    base, sub = _HP_SYMBOL_PARTS[idx]
    return with_channel ?
           rich(base, subscript(sub), " ($channel)") :
           rich(base, subscript(sub))
end

"""
    _hp_type_label(type) -> rich text

Legend label for a hyperparameter `type` (`"length_scale"`, ...): the math symbol
followed by the name in words, e.g. `ℓ_s  length scale`. Unknown types -- the
input/output statistics from `make_stats_param_grid` -- pass through unchanged.
"""
function _hp_type_label(type::AbstractString)
    key = get(_HP_TYPE_INDEX, String(type), 0)
    haskey(_HP_SYMBOL_PARTS, key) || return rich(String(type))
    base, sub = _HP_SYMBOL_PARTS[key]
    return rich(base, subscript(sub), "  ", replace(String(type), "_" => " "))
end

"""
    _hp_tick_positions(mults; max_ticks=7) -> Vector{Float64}

Which of the swept multipliers get a labelled tick. A tick per swept value is
right for the 3-5 step sweeps, but a 15- or 21-step sweep overruns the axis and
the labels collide, so beyond `max_ticks` this thins them out.

Thinning is anchored on the ×1 tick and steps outwards by a constant stride, so
the baseline is always labelled and the kept ticks stay evenly spaced (the sweep
is geometric, so constant stride is constant distance on the log axis). The two
endpoints are added back when they are not already kept, since they are the
extremes of the range the figure is claiming -- but only when doing so leaves a
gap, otherwise re-adding them recreates the collision being avoided.
"""
function _hp_tick_positions(mults::AbstractVector{<:Real}; max_ticks::Int=7)
    pos = sort(unique(float.(mults)))
    n = length(pos)
    (n <= max_ticks || max_ticks < 2) && return pos

    anchor = argmin(abs.(log.(pos)))          # the ×1 tick, or the nearest to it
    stride = cld(n - 1, max_ticks - 1)
    idx = sort(unique(vcat(collect(anchor:-stride:1), collect(anchor:stride:n))))

    for e in (1, n)
        if !(e in idx) && minimum(abs.(idx .- e)) > 1
            push!(idx, e)
        end
    end
    return pos[sort!(idx)]
end

"""
    _hp_multiplier_ticks(mults; max_ticks=7) -> (positions, labels)

Labelled ticks for the multiplier axis, `×<value>`. The exact baseline tick is
labelled `×1` rather than `×1.0`, since it is the reference the reader looks for
first. See [`_hp_tick_positions`](@ref) for which values get a label.
"""
function _hp_multiplier_ticks(mults::AbstractVector{<:Real}; max_ticks::Int=7)
    pos = _hp_tick_positions(mults; max_ticks=max_ticks)
    return (pos, map(hp_multiplier_label, pos))
end

"""
    hp_multiplier_label(m) -> String

A single multiplier rendered the way the sensitivity axis renders it: `×0.1`, `×1`,
`×10`. The exact baseline is `×1`, not `×1.0`, since it is the reference the reader
looks for first.

Public, and separate from [`_hp_multiplier_ticks`](@ref), so a figure that labels
individual series by their multiplier spells them exactly as the axis of the sweep
figure does -- that identity is what lets a reader carry a point from one to the other.
"""
function hp_multiplier_label(m::Real)::String
    isapprox(m, 1.0; rtol=1e-6) && return "×1"
    r = round(m; sigdigits=3)
    # Whole multipliers lose the trailing `.0`: `×10`, not `×10.0`. Same reasoning as
    # the `×1` case -- these are the round numbers a reader anchors on.
    return isinteger(r) && abs(r) < 1e15 ? "×$(Int(r))" : "×$r"
end

"""
    _draw_hp_panel!(ax, sub; color, markersize, linewidth) -> (mults, pct)

Draw one parameter's sensitivity curve into `ax`: percent RMSE change against
the multiplier. `sub` is the subset of the `vary_hsgp_parameters` frame for a
single `parameter`. Returns the plotted vectors so callers can annotate them.
"""
function _draw_hp_panel!(ax::Axis, sub::AbstractDataFrame;
    color=Makie.wong_colors()[1], markersize::Real=6, linewidth::Real=2,
    max_ticks::Int=7)

    base_val = first(sub.base_value)
    order = sortperm(sub.tested_value)
    mults = float.(sub.tested_value[order]) ./ base_val
    pct = 100 .* float.(sub.relative_change[order])

    # Baseline reference: 0% change. There is deliberately no rule at ×1 -- the
    # point there is 0% by construction, so it already sits on this line, and the
    # ×1 tick marks the position.
    hlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)

    lines!(ax, mults, pct; color=color, linewidth=linewidth)
    scatter!(ax, mults, pct; color=color, markersize=markersize)

    ax.xticks = _hp_multiplier_ticks(mults; max_ticks=max_ticks)
    return mults, pct
end

"""
    _hp_xlims(log_range; pad=0.04) -> (lo, hi)

Multiplier-space limits for a sweep specified as log₁₀ exponents, with a little
padding so the markers at the ends of the range are not clipped by the frame.
"""
function _hp_xlims(log_range::Tuple{Float64,Float64}; pad::Real=0.04)
    lo, hi = log_range
    span = max(hi - lo, eps())
    return (10.0^(lo - pad * span), 10.0^(hi + pad * span))
end

const _HP_PCT_TICKFORMAT = vs -> [string(round(v; digits=1), "%") for v in vs]

# `+ 0.0` collapses IEEE negative zero, so a parameter that lands exactly on the
# baseline reads "0.0%" rather than "-0.0%".
_pct_str(x::Real) = string(round(x; digits=1) + 0.0, "%")

function plot_hp_sensitivity(
    df::DataFrame, grid::ParamGrid, log_range::Tuple{Float64,Float64};
    save_path::Union{String,Nothing}=nothing,
    max_ticks::Int=5)

    n_rows, n_cols = size(grid.specs)
    fig = Figure(size=(400 * n_cols, 400 * n_rows))

    for row in 1:n_rows
        for col in 1:n_cols
            spec = grid.specs[row, col]
            if isnothing(spec)
                # Skip empty cell – leave blank
                continue
            end
            # Find data for this parameter
            sub = df[df.parameter .== spec.name, :]
            if isempty(sub)
                @warn "Missing data for $(spec.name)"
                continue
            end
            ax = Axis(fig[row, col];
                xlabel="multiplier",
                xscale=log10,
                ytickformat=_HP_PCT_TICKFORMAT,
                title=hp_param_label(spec.name),
                xgridstyle=:dash,
                ygridstyle=:dash)
            # Colour by hyperparameter type, not by grid column: with
            # include_noise=false the columns shift left, so a column-indexed
            # palette would repaint the length scale in the noise term's colour.
            # Panels are only ~400 px wide, so they take fewer labels than the
            # single-parameter figure before the ×0.316-style labels collide.
            _draw_hp_panel!(ax, sub; color=hp_param_color(spec.name), max_ticks=max_ticks)
            xlims!(ax, _hp_xlims(log_range)...)
        end
    end

    # Add global y-label
    Label(fig[1:n_rows, 0], "RMSE change vs baseline [%]", rotation=π / 2, fontsize=14)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end

"""
    hp_param_name(channel, kind) -> String

The `parameter` string `vary_hsgp_parameters` uses for one hyperparameter, e.g.
`hp_param_name(:yaw, :length_scale) == "yaw[2]"`. Spelling the name out beats
hard-coding the index at the call site, where `"yaw[2]"` gives the reader no way
to tell a length scale from a signal variance.

`kind` is one of `:noise`, `:length_scale`, `:signal_variance` (the values of
`_HYPERPARAM_TYPES`).
"""
function hp_param_name(channel::Union{Symbol,AbstractString}, kind::Union{Symbol,AbstractString})::String
    want = String(kind)
    # The index in the parameter name is the hyperparameter's position in the
    # channel vector, which is what _HYPERPARAM_TYPES is keyed by.
    key = get(_HP_TYPE_INDEX, want, nothing)
    isnothing(key) && throw(ArgumentError(
        "unknown hyperparameter kind \"$want\"; have $(sort(collect(keys(_HP_TYPE_INDEX))))"))
    return "$(String(channel))[$key]"
end

"""
    plot_hp_param_sensitivity(df, parameter; log_range=nothing, save_path=nothing, ...)

Close-up of a *single* hyperparameter's sensitivity curve — the same curve
`plot_hp_sensitivity` draws in one cell of its grid, given the whole figure so
the shape between the swept points is actually readable.

This is the companion to [`plot_signed_relative_change`](@ref), which collapses
each parameter to the interval `[min, max]` of its RMSE change. That interval
answers "how much" but not "in which direction, and how does it get there": a
parameter whose curve falls monotonically as the multiplier grows and one whose
baseline sits on a local *maximum* — the documented case for the yaw length
scale, where both directions improve RMSE — produce the same bar. Plot the curve
for the parameter the argument turns on.

# Arguments
- `df`: frame from `vary_hsgp_parameters`.
- `parameter`: the parameter name, e.g. `hp_param_name(:yaw, :length_scale)`.
- `log_range`: the sweep's log₁₀ range, for x limits matching the grid figure.
  Defaults to the span of the data.
- `mark_best`: ring the lowest-RMSE point.
- `show_values`: print the percentage beside every swept point (default off --
  the points are readable off the axis, and the labels crowd a steep curve).
- `show_subtitle`: draw the subtitle line (default `true`). Turn it off for a
  figure whose caption in the document already carries the same information.
- `show_summary`: the line under the title carrying the baseline value, the best
  multiplier and the range (default off).
"""
function plot_hp_param_sensitivity(df::DataFrame, parameter::AbstractString;
    log_range::Union{Tuple{Float64,Float64},Nothing}=nothing,
    save_path::Union{String,Nothing}=nothing,
    color=nothing,   # defaults to the parameter's type colour
    mark_best::Bool=true,
    show_values::Bool=false,
    show_summary::Bool=false,
    show_subtitle::Bool=true,
    show_absolute_axis::Bool=true,
    max_ticks::Int=7,
    figsize::Tuple{Int,Int}=(760, 520))

    sub = df[df.parameter .== parameter, :]
    if isempty(sub)
        available = sort(unique(df.parameter[df.parameter .!= "baseline"]))
        throw(ArgumentError("no rows for parameter \"$parameter\"; have $available"))
    end
    nrow(sub) > 1 || throw(ArgumentError(
        "parameter \"$parameter\" has a single tested value; there is no curve to draw"))

    base_val = first(sub.base_value)
    label = hp_param_label(parameter)
    color = isnothing(color) ? hp_param_color(parameter) : color

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        xlabel=rich("multiplier on ", label),
        ylabel="RMSE change vs baseline [%]",
        xscale=log10,
        ytickformat=_HP_PCT_TICKFORMAT,
        xgridstyle=:dash,
        ygridstyle=:dash)

    mults, pct = _draw_hp_panel!(ax, sub; color=color, markersize=11, linewidth=2.5,
        max_ticks=max_ticks)

    annotations = ["baseline value $(round(base_val; sigdigits=4))"]
    if mark_best
        b = argmin(pct)
        scatter!(ax, [mults[b]], [pct[b]];
            color=:transparent, strokecolor=color, strokewidth=2.5, markersize=20)
        push!(annotations,
            "best ×$(round(mults[b]; sigdigits=3)) at $(_pct_str(pct[b])) " *
            "(value $(round(base_val * mults[b]; sigdigits=4)))")
    end
    push!(annotations, "range $(_pct_str(minimum(pct))) to $(_pct_str(maximum(pct)))")

    if show_values
        for (x, y) in zip(mults, pct)
            text!(ax, x, y; text=_pct_str(y),
                align=(:center, :bottom), offset=(0, 9), fontsize=9)
        end
    end

    xl = isnothing(log_range) ?
         _hp_xlims((log10(minimum(mults)), log10(maximum(mults)))) :
         _hp_xlims(log_range)
    xlims!(ax, xl...)

    # Second x axis carrying the absolute hyperparameter value at each swept
    # multiplier: the multiplier is what the experiment varied, but the value is
    # what anyone reproducing the result has to type into the parameter file.
    #
    # The title goes on whichever axis is topmost, so it stacks above the
    # secondary ticks instead of being drawn on top of them.
    title_ax = ax
    tickpos = _hp_tick_positions(mults; max_ticks=max_ticks)
    if show_absolute_axis
        ax_top = Axis(fig[1, 1];
            xlabel=rich(label, " value"),
            xscale=log10,
            xaxisposition=:top,
            # Same positions as the bottom axis: thinning them independently
            # would leave the two rows of labels out of register.
            xticks=(tickpos, ["$(round(base_val * m; sigdigits=3))" for m in tickpos]),
            xgridvisible=false,
            ygridvisible=false)
        hidespines!(ax_top)
        hideydecorations!(ax_top)
        linkxaxes!(ax, ax_top)
        xlims!(ax_top, xl...)
        title_ax = ax_top
    end

    title_ax.title = rich("Sensitivity to ", label)
    # `show_summary` decides whether there is a summary line at all;
    # `show_subtitle` is the flag every figure here takes to strip the subtitle
    # when the caption underneath the figure will say the same thing.
    if show_summary && show_subtitle
        title_ax.subtitle = join(annotations, "   |   ")
        title_ax.subtitlesize = 11
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end

"""
    plot_channel_boxplots(df; show_trials, save_path)

Box-plot the full raw-value distribution per channel (one box = all samples across all
trials), then overlay each trial's mean ± std as a scatter + error bar.
"""
function plot_channel_boxplots(
    df::DataFrame;
    show_trials::Bool=true,
    save_path::Union{String,Nothing}=nothing,
)
    input_df = subset(df, :io_type => x -> x .== :input)
    output_df = subset(df, :io_type => x -> x .== :output)

    fig = Figure(size=(1000, 500))
    ax_in = Axis(fig[1, 1], title="Input channels",
        xlabel="Channel", ylabel="Value")
    ax_out = Axis(fig[1, 2], title="Output channels",
        xlabel="Channel", ylabel="Value")

    # ------------------------------------------------------------------ #
    function draw_panel!(ax, panel_df, box_color)

        channels = sort(unique(panel_df.channel))
        ax.xticks = channels

        # ── 1. Box plot over ALL raw samples ──────────────────────────── #
        boxplot!(ax, panel_df.channel, panel_df.value;
            color=(box_color, 0.45),
            mediancolor=:white,
            whiskerwidth=0.5,
            show_outliers=true,
            outliercolor=(box_color, 0.3),
        )

        # ── 2. Per-trial mean ± std overlaid ──────────────────────────── #
        if show_trials
            trial_ids = sort(unique(panel_df.trial_id))

            # One distinct color per trial drawn from the :tableau_10 palette.
            palette = Makie.wong_colors()
            trial_color = Dict(t => palette[mod1(i, length(palette))]
                               for (i, t) in enumerate(trial_ids))

            # Global mean per channel (across all trials) — used to gauge how
            # "central" each trial point is and scale its jitter accordingly.
            ch_global_mean = Dict(
                ch => mean(panel_df.value[panel_df.channel .== ch])
                for ch in channels
            )
            all_trial_means = [
                mean(panel_df.value[panel_df.channel .== ch.&&panel_df.trial_id .== t])
                for ch in channels for t in trial_ids
            ]
            mean_min = minimum(all_trial_means)
            mean_max = maximum(all_trial_means)
            range_mean = max(mean_max - mean_min, eps(Float64))

            decay = 5.0   # jitter sharpness (3–10): higher → tighter near mean
            max_jitter = 0.4   # half-width at maximum spread

            for trial in trial_ids
                tdf = subset(panel_df, :trial_id => x -> x .== trial)
                color = trial_color[trial]

                xs = Float64[]
                mus = Float64[]
                sigmas = Float64[]

                for ch in channels
                    vals = tdf.value[tdf.channel .== ch]
                    isempty(vals) && continue

                    mu = mean(vals)

                    # Trials close to the channel mean get MORE jitter so they
                    # don't pile up; distant outliers need less spread.
                    d = clamp(abs(ch_global_mean[ch] - mu) / range_mean, 0.0, 1.0)
                    jitter_factor = exp(-decay * d)
                    jitter = (rand() - 0.5) * max_jitter * jitter_factor

                    push!(xs, ch + jitter)
                    push!(mus, mu)
                    push!(sigmas, std(vals))
                end

                errorbars!(ax, xs, mus, sigmas;
                    color=(color, 0.7),
                    whiskerwidth=3,
                    linewidth=1,
                )
                scatter!(ax, xs, mus;
                    markersize=6,
                    color=color,
                    strokewidth=0,
                )
                # Trial number label just above each mean dot
                for (x, mu) in zip(xs, mus)
                    text!(ax, x, mu;
                        text=string(trial),
                        offset=(0, 6),
                        align=(:right, :bottom),
                        fontsize=10,
                        color=color,
                    )
                end
            end
        end
    end
    # ------------------------------------------------------------------ #

    draw_panel!(ax_in, input_df, :steelblue)
    draw_panel!(ax_out, output_df, :coral)

    linkyaxes!(ax_in, ax_out)

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_max_relative_change(
        df::DataFrame;
        exclude_baseline::Bool=true,
        save_path::Union{String,Nothing}=nothing,
    )

Bar chart of the maximum (most positive) `relative_change` achieved per parameter,
as produced by `vary_hsgp_parameters`. Bars are sorted ascending left-to-right
(smallest max relative change on the left, largest on the right) and colored by
`type`.

# Arguments
- `df`: Output of `vary_hsgp_parameters` (needs `parameter`, `type`, `relative_change`).
- `exclude_baseline`: If `true` (default), drops the `parameter == "baseline"` row
  before computing per-parameter maxima.
- `save_path`: Optional path to save the figure.

# Returns
- A `Figure` object.
"""
function plot_max_relative_change(
    df::DataFrame;
    exclude_baseline::Bool=true,
    save_path::Union{String,Nothing}=nothing,
    show_text_val::Bool=false,
)
    plot_df = exclude_baseline ? df[df.parameter .!= "baseline", :] : df
    isempty(plot_df) && error("No rows to plot after filtering baseline")

    # Max relative_change per parameter, keeping the associated type. Percent,
    # matching every other figure in this file.
    gdf = combine(
        groupby(plot_df, :parameter),
        :relative_change => (v -> 100 * maximum(v)) => :max_pct_change,
        :type => first => :type,
    )

    # Sort ascending: smallest on the left, largest on the right
    sort!(gdf, :max_pct_change; rev=true)

    n = nrow(gdf)
    xs = 1:n

    types = unique(gdf.type)
    type_color = Dict(t => hp_type_color(t) for t in types)
    bar_colors = [type_color[t] for t in gdf.type]

    fig = Figure(size=(max(800, 40 * n), 500))
    ax = Axis(fig[1, 1];
        title="Maximum relative RMSE change per parameter",
        ylabel="Max RMSE change vs baseline [%]",
        ytickformat=_HP_PCT_TICKFORMAT,
        xticks=(xs, [hp_param_label(p) for p in gdf.parameter]),
        xticklabelrotation=π / 3,
        xgridvisible=false)

    barplot!(ax, xs, gdf.max_pct_change; color=bar_colors)
    hlines!(ax, 0.0; color=:gray, linestyle=:dash, linewidth=1)

    if show_text_val
        for (x, val) in zip(xs, gdf.max_pct_change)
            text!(ax, x, val;
                text=_pct_str(val),
                align=(:center, val >= 0 ? :bottom : :top),
                fontsize=8)
        end
    end
    legend_elems = [PolyElement(color=type_color[t]) for t in types]
    Legend(fig[1, 2], legend_elems, [_hp_type_label(t) for t in types]; tellheight=false)

    isnothing(save_path) || save(save_path, fig)
    return fig
end
"""
    plot_signed_relative_change(df; exclude_baseline=true, save_path=nothing,
                                sort_by=:span, show_text_val=false)

Signed tornado plot of the sensitivity sweep: for each parameter, the full range
of relative RMSE change observed, from the most-improving to the most-degrading
perturbation, with a marker at each tested point.

Complements [`plot_max_relative_change`](@ref), which reduces each parameter to
`maximum(relative_change)` -- a *signed* max. That discards direction, and
direction is frequently the whole result. Worked example from
`out/Results/2_HypSensitivity/SensitivityAnalysis/data/ANG215_..._2026-08-19T12:17:58.504.csv`:
the yaw length scale has `maximum(relative_change) = +0.014`, so it reads as a
1.4% effect in the max-bar chart, while its *minimum* is `-0.603` -- every
perturbation tested, in both directions, improved RMSE by ~60%. The baseline
value sits on a local maximum. The max-bar chart cannot show that; this one puts
it at the top.

`sort_by` is `:span` (widest total range first), `:min` (best improvement first)
or `:max`.

Axes match the sensitivity curves: parameters are labelled with their math
symbol ([`hp_param_label`](@ref)) and the x axis is percent change against the
baseline, so a range quoted off this figure and one quoted off
[`plot_hp_param_sensitivity`](@ref) are in the same units.
"""
function plot_signed_relative_change(df::DataFrame;
    exclude_baseline::Bool=true,
    save_path::Union{String,Nothing}=nothing,
    sort_by::Symbol=:span,
    show_text_val::Bool=false,
    figsize::Tuple{Int,Int}=(900, 520))

    work = exclude_baseline ? df[df.parameter .!= "baseline", :] : copy(df)
    isempty(work) && throw(ArgumentError("no rows to plot"))

    # Percent, to match the sensitivity curves and the paired-comparison figures.
    work = copy(work)
    work.pct_change = 100 .* float.(work.relative_change)

    g = combine(groupby(work, [:parameter, :type]),
        :pct_change => minimum => :lo,
        :pct_change => maximum => :hi)
    g.span = g.hi .- g.lo

    key = sort_by === :span ? :span : sort_by === :min ? :lo : :hi
    rev = sort_by !== :min
    sort!(g, key; rev=rev)

    # Fixed per type, so the colours mean the same thing here as in the
    # sensitivity curves -- and do not shuffle when a sweep omits a type.
    types = unique(g.type)
    tcolor = Dict(t => hp_type_color(t) for t in types)

    n = nrow(g)
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        xlabel="RMSE change vs baseline [%]",
        xtickformat=_HP_PCT_TICKFORMAT,
        yticks=(1:n, [hp_param_label(p) for p in g.parameter]),
        title="Signed sensitivity range over the tested multipliers")
    vlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1)

    for (i, row) in enumerate(eachrow(g))
        c = tcolor[row.type]
        lines!(ax, [row.lo, row.hi], [float(i), float(i)]; color=c, linewidth=6)
        # every individual tested point, so a flat parameter is visibly flat
        pts = work[work.parameter .== row.parameter, :pct_change]
        scatter!(ax, pts, fill(float(i), length(pts));
            color=:white, strokecolor=c, strokewidth=1.5, markersize=8)
        if show_text_val
            text!(ax, row.lo, float(i); text=_pct_str(row.lo),
                align=(:right, :center), offset=(-6, 0), fontsize=10)
        end
    end

    # Legend entries pair the symbol with its name: once the y ticks are symbols
    # alone, this is the only place the reader can learn that ℓ is the length
    # scale rather than guess it.
    Legend(fig[1, 2],
        [PolyElement(color=tcolor[t]) for t in types],
        [_hp_type_label(t) for t in types], "Parameter type"; framevisible=false)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    return fig
end
