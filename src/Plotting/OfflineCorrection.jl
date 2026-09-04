"""
    plot_line_with_std!(ax, t, data, data_std; color, label=nothing, linewidth=1.4, kwargs...)

Add a line plot to `ax` with two uncertainty bands (1σ and 2σ) drawn from `data_std`.
If `data_std` is `nothing`, only the line is plotted.

Only the line carries the `label` – the bands are silent in the legend.
"""
function plot_line_with_std!(ax, t, data, data_std; color, label=nothing, linewidth=1.4, kwargs...)
    if !isnothing(data_std)
        # 2σ band (lighter)
        # band!(ax, t, data .- 2data_std, data .+ 2data_std; color=(color, 0.1), label=label)
        # 1σ band
        band!(ax, t, data .- data_std, data .+ data_std; color=(color, 0.15), label=label)
    end
    lines!(ax, t, data; color=color, linewidth=linewidth, label=label, kwargs...)
end

"""
    plot_regression_results(
        pred_data::Union{Nothing,AbstractDict{String,Union{CorrectionIO, Dict}}},
        true_data::Union{CorrectionIO, Matrix{Float64}})

Plot regression results for yaw and position components (X, Y, Z) with RMSE annotations.
Supports both CorrectionIO objects and traditional dictionary/matrix formats.

# Arguments
- `pred_data`: Dictionary mapping method names to CorrectionIO
- `true_data`: CorrectionIO

# Keywords
- `channels`: which output channels to draw, as indices into the 4-row correction vector
  (`1:3` = position, `4` = yaw). `nothing` (default) draws them all. Units and panel
  titles follow the *original* channel index, so selecting `[4]` still labels the panel
  yaw/radians rather than treating it as the first position component.
- `show_std`: draw the ±1σ band around each prediction (default `true`). Worth turning off
  when one series has a far wider predictive variance than the others, since its band
  otherwise fills the panel and hides every line in it.
- `ylims`: `(lo, hi)` limits for every panel. Without this a single divergent series sets
  the scale and compresses the rest to a flat line at zero; the legend still reports each
  series' true RMSE, so clipping costs no information as long as the caption says so.
- `save_path`: write the figure here as well as returning it.

# Returns
- `Figure` object.
"""
function plot_regression_results(
    pred_data::Union{Nothing,AbstractDict{String,CorrectionIO}},
    true_data::Union{Nothing,CorrectionIO}=nothing;
    channels::Union{Nothing,AbstractVector{Int}}=nothing,
    show_std::Bool=true,
    ylims::Union{Nothing,Tuple{Real,Real}}=nothing,
    save_path::Union{String,Nothing}=nothing
)
    ref = isnothing(true_data) ? first(values(pred_data)) : true_data
    t = ref.t
    n_dim = size(ref.data, 1)
    @assert 1 <= n_dim <= 4

    rows = isnothing(channels) ? collect(1:n_dim) : collect(channels)
    all(r -> 1 <= r <= n_dim, rows) ||
        throw(ArgumentError("channels must index into 1:$n_dim, got $rows"))
    n_panels = length(rows)

    n_cols = min(2, n_panels)
    n_rows = ceil(Int, n_panels / n_cols)
    fig = Figure(size=n_panels == 1 ? (900, 500) : (1200, 900))
    axes = Vector{Axis}(undef, n_panels)

    for (idx, row_ch) in enumerate(rows)
        row = (idx - 1) ÷ n_cols + 1
        col = (idx - 1) % n_cols + 1
        ax = Axis(fig[row, col])

        # Keyed off the channel, not the panel position: with `channels` set they are no
        # longer the same number, and a yaw panel labelled "meters" is worse than no label.
        ax.ylabel = row_ch == 4 ? "radians" : "meters"
        ax.title = _OUTPUT_NAMES[row_ch]
        if row == n_rows
            ax.xlabel = "Time [s]"
        end
        axes[idx] = ax
    end

    # Ground truth
    if !isnothing(true_data)
        for (plot_idx, row) in enumerate(rows)
            std_vals = (show_std && !isnothing(true_data.data_std)) ? true_data.data_std[row, :] : nothing
            plot_line_with_std!(axes[plot_idx], true_data.t, true_data.data[row, :],
                std_vals;
                color=:black, linewidth=0.9, label="Target")
        end
    end

    # Predictions
    if !isnothing(pred_data)
        colors = [:red, :blue, :green, :orange, :purple]

        for (method_idx, (method_name, pred)) in enumerate(pred_data)
            color = colors[(method_idx-1)%length(colors)+1]

            # RMSE on overlapping interval
            rmse = fill(NaN, n_panels)
            if !isnothing(true_data)
                res = truncate_to_overlap(true_data, pred)
                if !isnothing(res)
                    gt_trim, pred_trim = res
                    is_compatible(gt_trim, pred_trim) ||
                        throw(ArgumentError(
                            "Series '$method_name' is not compatible with ground truth " *
                            "(different lengths or timestamps differ > 1e-9). " *
                            "Truncate first or resample."))
                    for (plot_idx, row) in enumerate(rows)
                        rmse[plot_idx] = sqrt(mean(
                            (pred_trim.data[row, :] .- gt_trim.data[row, :]) .^ 2))
                    end
                end
            end

            for (plot_idx, row) in enumerate(rows)
                rmse_str = isnan(rmse[plot_idx]) ? "" :
                           " (RMSE=$(round(rmse[plot_idx]; digits=4)))"
                # The mean std is reported in the legend even when the band is hidden: it is
                # the number that says how confident the model was, and it is precisely the
                # series with a huge one that show_std=false is used to get out of the way.
                raw_std = isnothing(pred.data_std) ? nothing : pred.data_std[row, :]
                std_vals = show_std ? raw_std : nothing
                std_str = isnothing(raw_std) ? "" : " (Mean Std=$(round(mean(raw_std);digits=3)))"
                label = "$method_name$rmse_str$std_str"
                plot_line_with_std!(axes[plot_idx], pred.t, pred.data[row, :],
                    std_vals;
                    color=color, label=label)
            end
        end
    end

    for ax in axes
        axislegend(ax; position=:rt, merge=true)
        ax.xgridvisible = true
        ax.ygridvisible = true
        isnothing(ylims) || ylims!(ax, ylims[1], ylims[2])
    end
    # Legend(fig[n_rows+1, :], [ax for ax in axes]; orientation=:horizontal, merge=true)

    isnothing(save_path) || save(save_path, fig)
    return fig
end


function plot_regression_results(
    pred_data::CorrectionIO,
    true_data::Union{Nothing,CorrectionIO}=nothing;
    kwargs...
)

    pred_data = Dict(
        "Prediction" => pred_data
    )
    return plot_regression_results(pred_data, true_data; kwargs...)
end
"""
    plot_input_features(
        features::Matrix{Float64}, t::Union{Vector{Float64},Nothing}=nothing;
        labels::Union{Vector{String},Nothing}=nothing,
        title::String="Input Features"
    )

Plot three-dimensional input features over time.

# Arguments
- `features`: Feature matrix of size `(3, N)`.
- `t`: Optional time vector of length `N`. If `nothing`, uses sample indices.
- `labels`: Optional legend labels for the three dimensions (default: `["x", "y", "z"]`).
- `title`: Plot title.

# Returns
- A `Figure` object.
"""
function plot_input_features(
    features::Matrix{Float64},
    t::Union{Vector{Float64},Nothing}=nothing;
    feature_std::Union{Nothing,Matrix{Float64}}=nothing,
    labels::Union{Vector{String},Nothing}=nothing,
    title::String="Input Features"
)
    N = size(features, 2)
    n_channel = size(features, 1)
    if t === nothing
        t = 1:N
    end
    if labels === nothing
        labels = ["Feature $i" for i in 1:n_channel]
    end

    palette = [:red, :blue, :green, :orange, :purple]
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1]; xlabel="Time [s]", ylabel="Value", title=title,
        xgridvisible=true, ygridvisible=true)

    # Plot every channel present. Previously hard-coded to 1:3, which silently
    # dropped channels 4+ for TWOD_STEP_YAW / THREED_STEP_DT_YAW features.
    for i in 1:n_channel
        color = palette[mod1(i, length(palette))]
        lines!(ax, t, features[i, :]; color=color, linewidth=1.2, label=labels[i])

        if !isnothing(feature_std)
            label = "Feature $i ±σ"
            band!(ax, t, features[i, :] .- feature_std[i, :], features[i, :] .+ feature_std[i, :];
                color=(color, 0.15), label=label)

        end
    end
    axislegend(ax; position=:rt)
    return fig
end

function plot_input_features(
    features::CorrectionIO
)
    plot_input_features(features.data, features.t; feature_std=features.data_std)
end

"""
Colour index into `Makie.wong_colors()` for each correction method, so a method keeps
one colour across every figure instead of getting whatever its position in a
`Dict` happened to earn. Wong 1-3 were already spoken for by these three by
convention (see `_HP_COLOR_INDICES` in `Plotting/OnlineHpSensitivity.jl`, which
avoids them for exactly this reason); this states the convention rather than
leaving it to iteration order.
"""
const _METHOD_COLOR_INDICES = Dict{String,Int}(
    "ZUPT only" => 1,
    "Static" => 2,
    "HSGP" => 3,
)

const _METHOD_FALLBACK_COLOR = Makie.RGBAf(0.45, 0.45, 0.45, 1.0)

"""
    method_color(name) -> colour

Canonical colour for a correction method name (`"ZUPT only"`, `"Static"`, `"HSGP"`).
Anything else gets a neutral grey rather than silently borrowing another method's colour.
"""
function method_color(name::AbstractString)
    idx = get(_METHOD_COLOR_INDICES, String(name), nothing)
    return isnothing(idx) ? _METHOD_FALLBACK_COLOR : Makie.wong_colors()[idx]
end

"""
    color_shades(base, n; spread=26) -> Vector

`n` shades of `base`, varying lightness only. Hue and chroma are held fixed, so the
shades read as "the same method, different setting" rather than as unrelated series --
which is the point when the variants being compared are one estimator at several
hyperparameter values.

Lightness is clamped into [15, 95]: outside that a shade is either black or invisible
against a light background, and two clamped shades would be indistinguishable.
"""
function color_shades(base, n::Int; spread::Real=26)
    n <= 0 && return Makie.RGBAf[]
    c = convert(LCHab, convert(RGB, Makie.to_color(base)))
    n == 1 && return [Makie.RGBAf(convert(RGB, c))]
    ls = range(clamp(c.l - spread, 15, 95), clamp(c.l + spread, 15, 95); length=n)
    return [Makie.RGBAf(convert(RGB, LCHab(l, c.c, c.h))) for l in ls]
end

"""
    plot_regression_comparison(pred_data, true_data; kwargs...)

One channel of the GP correction, with every method's regressed output on a single panel.

A sibling of [`plot_regression_results`](@ref) rather than more keywords on it: this one
draws a single channel over a chosen part of the track, defaults its annotations off, and
takes explicit per-series colours -- a different figure with a different job, sharing
`plot_line_with_std!` and the segment-window logic.

# Arguments
- `pred_data`: ordered map from series name to its predicted `CorrectionIO`.
- `true_data`: the target `CorrectionIO`. Optional, but see `segment` below.

# Keywords
- `channel`: which output channel to draw (`1:3` position, `4` yaw). Default `4`.
- `segment`: `:test` (default), `:train` or `:full`. `:train`/`:test` need `train_ratio`.
  The cut is placed by time across the *union* of every series' time span, because the
  predictions only start once the model is trained -- cutting on a prediction's own span
  would put the boundary in the wrong place.
- `train_ratio`: the fraction passed to the filter.
- `colors`: map from series name to colour. Names not present fall back to
  [`method_color`](@ref).
- `linestyles` / `linewidths`: maps from series name to `:solid`/`:dash`/`:dot` and to a
  line width. Colour alone cannot separate two series that lie on top of each other, which
  is exactly what happens when several settings of one estimator all collapse to nearly
  the same output; linestyle can, and survives greyscale printing besides. Use them to say
  which series is the real setting and which are perturbations of it.
- `labels`: map from series name to the label to show in the legend. Values may be
  `rich` text, which is how a series gets a real superscript (`×10¹`) rather than a
  typed-out `10^+1`. Names not present fall back to the series name itself.
- `time_window`: `(t_first, t_last)` in seconds, applied on top of `segment`, for zooming
  in. It filters the data rather than only setting `xlims`, so `clip_quantile` scales to
  what is actually in view.
- `dataset` / `trial_id`: named in the subtitle, so a figure lifted out of the output
  directory still says which walk it came from.
- `show_std`: draw the ±1σ predictive band (default `false`).
- `show_rmse` / `show_mean_std`: append the RMSE against the target, and the mean
  predictive std, to each legend label. Both default `false` -- with four series and long
  hyperparameter names the legend is otherwise unreadable, and this figure is about the
  *shape* of the correction, with the numbers reported elsewhere.
- `clip_quantile`: sets the y-axis from the *target* rather than from the drawn values.
  Below 1 it is a quantile: `0.95` gives the target's central 95%. At and above 1 it is a
  margin on the target's full range: `1.0` is exactly that range, `1.1` adds 10% (5% each
  side), `1.5` half again. The two regimes meet continuously at 1, and the value alone
  fixes the window -- there is no hidden padding on top of it.

  Scaling to the target rather than to the drawn values is what makes this usable: a
  divergent series contributes most of the spread, so pooling everything leaves the
  y-range set by the very series you wanted to stop dominating the figure. The target is
  the quantity the corrections are trying to reproduce, so its own range is the scale at
  which they should be compared. Falls back to the predictions when there is no
  `true_data`. `nothing` (default) autoscales. Clipping crops the view only; it never
  changes a reported number.
- `show_subtitle`: draw the subtitle under the title (default `true`). Turn it off
  when the document's own caption carries the same information.

# Returns
- A `Figure` object (Makie figure).
"""
function plot_regression_comparison(
    pred_data::AbstractDict{String,CorrectionIO},
    true_data::Union{Nothing,CorrectionIO}=nothing;
    channel::Int=4,
    segment::Symbol=:test,
    train_ratio::Union{Nothing,Real}=nothing,
    colors::Union{Nothing,AbstractDict}=nothing,
    linestyles::Union{Nothing,AbstractDict}=nothing,
    linewidths::Union{Nothing,AbstractDict}=nothing,
    labels::Union{Nothing,AbstractDict}=nothing,
    time_window::Union{Nothing,Tuple{Real,Real}}=nothing,
    dataset::Union{Nothing,AbstractString}=nothing,
    trial_id::Union{Nothing,Integer}=nothing,
    show_std::Bool=false,
    show_rmse::Bool=false,
    show_mean_std::Bool=false,
    clip_quantile::Union{Nothing,Real}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    show_subtitle::Bool=true,
    save_path::Union{String,Nothing}=nothing
)
    isempty(pred_data) && throw(ArgumentError("pred_data is empty"))
    all_series = CorrectionIO[values(pred_data)...]
    isnothing(true_data) || push!(all_series, true_data)
    1 <= channel <= minimum(size(s.data, 1) for s in all_series) ||
        throw(ArgumentError("channel $channel out of range for the supplied series"))

    # One cut time shared by every series, built from the union of all spans: a
    # prediction series only starts once the model is trained, so cutting on its own
    # span would put the train/test boundary in the wrong place.
    t_lo = minimum(minimum(s.t) for s in all_series)
    t_hi = maximum(maximum(s.t) for s in all_series)

    keep = if segment === :full
        _ -> true
    else
        segment in (:train, :test) ||
            throw(ArgumentError("segment must be :full, :train or :test, got :$segment"))
        isnothing(train_ratio) &&
            throw(ArgumentError("segment=:$segment needs train_ratio (the fraction passed to the filter)"))
        0 <= train_ratio <= 1 ||
            throw(ArgumentError("train_ratio must be in [0, 1], got $train_ratio"))
        t_cut = t_lo + train_ratio * (t_hi - t_lo)
        segment === :train ? (t -> t <= t_cut) : (t -> t > t_cut)
    end

    if !isnothing(time_window)
        time_window[1] < time_window[2] ||
            throw(ArgumentError("time_window must be (first, last) with first < last, got $time_window"))
        # Composed with the segment filter rather than replacing it, and applied to the
        # data rather than to xlims, so a zoomed view also rescales the y-axis instead of
        # keeping limits set by strides that are no longer on screen.
        seg_keep = keep
        keep = t -> seg_keep(t) && time_window[1] <= t <= time_window[2]
    end

    seg_name = segment === :train ? "Train" : segment === :test ? "Test" : "Full"

    # Same title/subtitle split the trajectory figures use: what the panel shows on the
    # title line, which data produced it underneath.
    provenance = String[]
    isnothing(dataset) || push!(provenance, String(dataset))
    isnothing(trial_id) || push!(provenance, "trial $trial_id")
    isnothing(time_window) || push!(provenance,
        @sprintf("%.0f-%.0f s", time_window[1], time_window[2]))

    fig = Figure(size=(1000, 560))
    ax = Axis(fig[1, 1];
        xlabel="Time [s]",
        ylabel=channel == 4 ? "Stride yaw error [rad]" : "Stride position error [m]",
        title=isnothing(title) ? "$(_OUTPUT_NAMES[channel]) correction — $(seg_name) segment" : title,
        subtitle=(show_subtitle && !isempty(provenance)) ? join(provenance, " · ") : "",
        subtitlesize=11,
        xgridvisible=true,
        ygridvisible=true)

    # Kept apart because the y-range is scaled to the target, not to the predictions.
    target_vals = Float64[]
    pred_vals = Float64[]

    if !isnothing(true_data)
        mask = keep.(true_data.t)
        if any(mask)
            vals = true_data.data[channel, mask]
            stds = (show_std && !isnothing(true_data.data_std)) ? true_data.data_std[channel, mask] : nothing
            plot_line_with_std!(ax, true_data.t[mask], vals, stds;
                color=:black, linewidth=0.9, label="Target")
            append!(target_vals, vals)
        end
    end

    for (name, pred) in pred_data
        mask = keep.(pred.t)
        any(mask) || continue
        c = isnothing(colors) ? method_color(name) : get(() -> method_color(name), colors, name)

        vals = pred.data[channel, mask]
        raw_std = isnothing(pred.data_std) ? nothing : pred.data_std[channel, mask]

        label = isnothing(labels) ? name : get(labels, name, name)
        annotations = String[]
        if show_rmse && !isnothing(true_data)
            res = truncate_to_overlap(true_data, pred)
            if !isnothing(res)
                gt_trim, pred_trim = res
                m = keep.(gt_trim.t)
                if any(m)
                    r = sqrt(mean((pred_trim.data[channel, m] .- gt_trim.data[channel, m]) .^ 2))
                    push!(annotations, "RMSE=$(round(r; digits=4))")
                end
            end
        end
        if show_mean_std && !isnothing(raw_std)
            push!(annotations, "Mean Std=$(round(mean(raw_std); digits=3))")
        end
        # Appended via `rich` rather than string concatenation: `label` may already be
        # rich text, which does not support `*`.
        isempty(annotations) ||
            (label = rich(label, " (", join(annotations, ", "), ")"))

        plot_line_with_std!(ax, pred.t[mask], vals, show_std ? raw_std : nothing;
            color=c, label=label,
            linewidth=isnothing(linewidths) ? 1.4 : get(linewidths, name, 1.4),
            linestyle=isnothing(linestyles) ? :solid : get(linestyles, name, :solid))
        append!(pred_vals, vals)
    end

    if !isnothing(clip_quantile)
        clip_quantile > 0 ||
            throw(ArgumentError("clip_quantile must be positive, got $clip_quantile"))
        scale_vals = isempty(target_vals) ? pred_vals : target_vals
        if !isempty(scale_vals)
            # Below 1 the value admits more of the target into the window; at or above 1
            # the window is the target's full range widened by that factor. Splitting it
            # this way keeps the knob monotonic -- larger always means more visible -- and
            # means the caller's number alone decides the limits.
            tail = (1 - min(clip_quantile, 1.0)) / 2
            lo = quantile(scale_vals, tail)
            hi = quantile(scale_vals, 1 - tail)
            mid = (lo + hi) / 2
            half = max((hi - lo) / 2, eps()) * max(clip_quantile, 1.0)
            ylims!(ax, mid - half, mid + half)
        end
    end

    # Below the panel rather than inside it: these series sit on top of each other around
    # zero, which is exactly where an in-axis legend would land, and the labels are long
    # enough that a boxed legend would cover a good part of the data.
    Legend(fig[2, 1], ax; orientation=:horizontal, nbanks=1,
        tellwidth=false, tellheight=true, framevisible=false, merge=true)

    isnothing(save_path) || save(save_path, fig)
    return fig
end
