"""
    plot_groundtruth_vs_inertial_positions(trajs, gt_traj)

Plot 2D positions (X-Y) of ground truth and one or more estimated trajectories.

# Arguments
- `trajs`: A `Trajectory` object (single estimate) or a `Dict{String,Trajectory}` (multiple).
- `gt_traj`: Ground truth `Trajectory`.

# Keywords
- `segment`: which part of the track to draw -- `:full` (default), `:train` or `:test`.
  `:train`/`:test` need `train_ratio` and cut the trajectory themselves, so callers do
  not have to work out stride indices by hand. The title is labelled to match.
- `train_ratio`: the same fraction passed to the filter. Required by `:train`/`:test`.
- `start`/`stop`: explicit index window, overriding `segment` on whichever end is given.
  Leave both `nothing` to let `segment` decide. Everything reported in the title and the
  legend (duration, distance, RMSE) is measured over the drawn window only, never over
  the full track.
- `marker_stride`: draw a dot every Nth stride along each track. `0` (default) draws none:
  on a looping walk the per-stride dots were the bulk of the clutter, and the start/stop
  markers still show the direction of travel. Set e.g. `5` to get a pace cue back.
- `segment_label`: overrides the name `segment` puts in the title.
- `legend_position`: `:outer_right` (default) or `:outer_bottom` put the legend outside
  the axis so it cannot cover the track. Any other symbol is passed to `axislegend` as an
  in-axis anchor (`:rt`, `:lt`, ...), which is compact but may overlap the trajectory.
- `show_subtitle`: draw the subtitle under the title (default `true`). Turn it off
  when the document's own caption carries the same information.
- `save_path`: write the figure here as well as returning it. Call inside
  `results_figure()` (`scripts/5Results/_common.jl`) so it is saved under the same
  CairoMakie theme as every other results figure.

# Returns
- A `Figure` object (Makie figure).
"""
function plot_groundtruth_vs_inertial_positions(
    trajs::Trajectory, gt_traj::Union{Nothing,Trajectory};
    segment::Symbol=:full,
    train_ratio::Union{Nothing,Real}=nothing,
    start=nothing,
    stop=nothing,
    show_heading=false,
    heading_stride=10,
    heading_length=0.15,
    marker_stride=0,
    segment_label::Union{Nothing,AbstractString}=nothing,
    legend_position::Symbol=:outer_right,
    show_subtitle::Bool=true,
    save_path::Union{String,Nothing}=nothing,
)
    plot_groundtruth_vs_inertial_positions(Dict("Estimation" => trajs), gt_traj;
        segment=segment,
        train_ratio=train_ratio,
        start=start,
        stop=stop,
        show_heading=show_heading,
        heading_stride=heading_stride,
        heading_length=heading_length,
        marker_stride=marker_stride,
        segment_label=segment_label,
        legend_position=legend_position,
        show_subtitle=show_subtitle,
        save_path=save_path,
    )
end

"""
    _segment_window(ref_traj, segment, train_ratio) -> (first, last)

Index range of `ref_traj` covered by `segment` (`:full`, `:train` or `:test`).

The train/test split is made on the IMU sample stream, which is uniformly sampled, so
`train_ratio` is really a fraction of elapsed *time*. The trajectories plotted here are
stride-indexed and strides are not uniform in time, so the cut is placed by time rather than
by scaling the stride count -- otherwise the drawn window drifts away from the split the
filter actually used, by more the more the walking pace varies.
"""
function _segment_window(ref_traj::Trajectory, segment::Symbol, train_ratio::Union{Nothing,Real})
    return _segment_window(ref_traj.t, segment, train_ratio)
end

# Core method on a bare time vector, so anything time-indexed -- a Trajectory, a
# CorrectionIO -- gets the same train/test cut from one implementation.
function _segment_window(t::AbstractVector{<:Real}, segment::Symbol, train_ratio::Union{Nothing,Real})
    n = length(t)
    segment === :full && return (1, n)
    segment in (:train, :test) ||
        throw(ArgumentError("segment must be :full, :train or :test, got :$segment"))
    isnothing(train_ratio) &&
        throw(ArgumentError("segment=:$segment needs train_ratio (the fraction passed to the filter)"))
    0 <= train_ratio <= 1 ||
        throw(ArgumentError("train_ratio must be in [0, 1], got $train_ratio"))

    t_cut = t[1] + train_ratio * (t[n] - t[1])
    n_train = something(findlast(<=(t_cut), t), 0)
    segment === :train && return (1, max(n_train, 1))
    return (min(n_train + 1, n), n)
end

"""
    _heading_arrows!(ax, traj, idx, len, color)

Draw yaw arrows at `idx` along `traj`. Shared so the ground-truth and estimate paths
cannot drift apart in how they read heading out of `R_nb`.
"""
function _heading_arrows!(ax, traj::Trajectory, idx, len::Real, color)
    pts = Point2f[]
    dirs = Vec2f[]
    for i in idx
        h = matrix_to_euler(traj.R_nb[:, :, i])[3]
        push!(pts, Point2f(traj.pos[1, i], traj.pos[2, i]))
        push!(dirs, len * Vec2f(cos(h), sin(h)))
    end
    arrows2d!(ax, pts, dirs; color=color)
end

"""
    _window_rmse(traj, gt_traj, window) -> Union{Nothing,Float64}

Horizontal RMSE of `traj` against `gt_traj` over `window`, or `nothing` when the two
cannot be compared there (window runs past either trajectory, or the time bases do not
line up). Returning `nothing` rather than throwing lets the caller simply drop the
number from a legend label instead of failing the whole plot.
"""
function _window_rmse(traj::Trajectory, gt_traj::Trajectory, window::AbstractRange)
    (last(window) <= length(traj.t) && last(window) <= length(gt_traj.t)) || return nothing
    tr_w, gt_w = traj[window], gt_traj[window]
    is_compatible(tr_w, gt_w) || return nothing
    return rmse(tr_w, gt_w)[end]
end

"""
    _draw_tracks!(ax, trajs, gt_traj, win_start, win_stop; kwargs...) -> (entries, labels, series)

Draw the ground truth and every estimated track onto `ax` over `win_start:win_stop`,
and return the legend entries/labels plus the `(traj, colour, last_index)` triples in
the order they were drawn.

Split out so the standalone top-down figure and the combined
[`plot_trajectory_and_distance_error`](@ref) draw the same tracks in the same colours
from one piece of code, and so the colour a series gets in one panel is by construction
the colour it gets in the other.
"""
function _draw_tracks!(
    ax,
    trajs::AbstractDict{String,Trajectory},
    gt_traj::Union{Nothing,Trajectory},
    win_start::Int,
    win_stop::Int;
    show_heading::Bool=false,
    heading_stride::Int=10,
    heading_length::Float64=0.15,
    marker_stride::Int=0
)
    # Casing colour: the axis background, not hard white, so the halo stays invisible
    # under both the default theme and theme_ggplot2's grey panel.
    casing_color = ax.backgroundcolor[]
    casing_width = 1.8
    line_width = 1.2

    # Legend entries are built from LineElement/MarkerElement rather than from plot
    # objects: the tracks are drawn as per-stride segments below, so there is no single
    # plot per series to point the legend at.
    entries = Vector{Any}[]
    labels = String[]
    # Markers and heading arrows go on last, after every line segment: drawn inline they
    # would be clipped by the casing of whatever is drawn after them.
    overlay_passes = Function[]

    series = Tuple{Trajectory,Any,Int}[]

    for ((key, traj), c) in zip(trajs, Iterators.cycle(Makie.wong_colors()))
        n = min(length(traj.t), win_stop)
        push!(series, (traj, c, n))

        window_rmse = isnothing(gt_traj) ? nothing : _window_rmse(traj, gt_traj, win_start:n)
        label = isnothing(window_rmse) ? key : @sprintf("%s - RMSE %.2f m", key, window_rmse)

        entry = Any[LineElement(color=c, linewidth=line_width)]
        marker_stride > 0 &&
            push!(entry, MarkerElement(color=c, marker=:circle, markersize=5))
        push!(entries, entry)
        push!(labels, label)

        push!(overlay_passes, function ()
            if marker_stride > 0
                idx = win_start:marker_stride:n
                scatter!(ax, traj.pos[1, idx], traj.pos[2, idx];
                    color=c, marker=:circle, markersize=5, alpha=0.6)
            end
            scatter!(ax, [traj.pos[1, win_start]], [traj.pos[2, win_start]];
                color=c, marker=:circle, markersize=12)
            scatter!(ax, [traj.pos[1, n]], [traj.pos[2, n]];
                color=c, marker=:rect, markersize=12)
            show_heading && _heading_arrows!(ax, traj, win_start:heading_stride:n, heading_length, c)
        end)
    end

    # Draw one time step at a time across ALL series, rather than one whole polyline per
    # series. Whole polylines make the z-order follow the order of `trajs`, so the last
    # entry sits on top of the others everywhere on the plot regardless of when each part
    # of the walk happened. Stepping through time instead means a later part of any track
    # covers an earlier part of any other, which is what the casing is meant to convey.
    for k in win_start:(win_stop-1)
        segs = Point2f[]
        cols = []
        for (traj, c, n) in series
            k + 1 <= n || continue
            push!(segs, Point2f(traj.pos[1, k], traj.pos[2, k]),
                Point2f(traj.pos[1, k+1], traj.pos[2, k+1]))
            push!(cols, c)
        end
        isempty(segs) && continue
        linesegments!(ax, segs; color=casing_color, linewidth=line_width + casing_width)
        linesegments!(ax, segs; color=repeat(cols, inner=2), linewidth=line_width)
    end

    if !isnothing(gt_traj)
        n = min(length(gt_traj.t), win_stop)
        idx = win_start:n

        # Ground truth stays a single uncased polyline on top of everything. It is the
        # reference the estimates are read against, so unlike the estimates it should not
        # be chopped up by anything -- and a dashed black line reads over them cleanly.
        lines!(ax, gt_traj.pos[1, idx], gt_traj.pos[2, idx];
            color=:black, linestyle=:dash, linewidth=line_width)

        gt_entry = Any[LineElement(color=:black, linestyle=:dash, linewidth=line_width)]
        marker_stride > 0 &&
            push!(gt_entry, MarkerElement(color=:black, marker=:circle, markersize=5))
        pushfirst!(entries, gt_entry)
        pushfirst!(labels, "Ground truth")

        push!(overlay_passes, function ()
            if marker_stride > 0
                m_idx = win_start:marker_stride:n
                scatter!(ax, gt_traj.pos[1, m_idx], gt_traj.pos[2, m_idx];
                    color=:black, marker=:circle, markersize=5, alpha=0.6)
            end
            scatter!(ax, [gt_traj.pos[1, win_start]], [gt_traj.pos[2, win_start]];
                color=:black, marker=:circle, markersize=12)
            scatter!(ax, [gt_traj.pos[1, n]], [gt_traj.pos[2, n]];
                color=:black, marker=:rect, markersize=12)
            show_heading && _heading_arrows!(ax, gt_traj, win_start:heading_stride:n, heading_length, :black)
        end)
    end

    foreach(f -> f(), overlay_passes)

    return entries, labels, series
end

function plot_groundtruth_vs_inertial_positions(
    trajs::AbstractDict{String,Trajectory},
    gt_traj::Union{Nothing,Trajectory};
    segment::Symbol=:full,
    train_ratio::Union{Nothing,Real}=nothing,
    start::Union{Nothing,Int}=nothing,
    stop::Union{Nothing,Int}=nothing,
    show_heading::Bool=false,
    heading_stride::Int=10,
    heading_length::Float64=0.15,
    marker_stride::Int=0,
    segment_label::Union{Nothing,AbstractString}=nothing,
    legend_position::Symbol=:outer_right,
    show_subtitle::Bool=true,
    save_path::Union{String,Nothing}=nothing
)

    # The window is common to every series, so measure it once on whichever trajectory
    # is available -- ground truth when there is one, otherwise the first estimate.
    ref_traj = isnothing(gt_traj) ? first(values(trajs)) : gt_traj
    n_ref = length(ref_traj.t)

    # `segment` picks the window; an explicit start/stop overrides it on that end.
    seg_start, seg_stop = _segment_window(ref_traj, segment, train_ratio)
    win_start = clamp(isnothing(start) ? seg_start : start, 1, n_ref)
    win_stop = clamp(isnothing(stop) ? seg_stop : stop, win_start, n_ref)
    window = win_start:win_stop
    n_strides = win_stop - win_start

    duration = ref_traj.t[win_stop] - ref_traj.t[win_start]
    distance = total_distance(ref_traj[window])

    name = isnothing(segment_label) ?
           (segment === :train ? "Train" : segment === :test ? "Test" : nothing) :
           segment_label

    title = isnothing(name) ?
            "Ground truth vs estimated trajectories" :
            "Ground truth vs estimated trajectories — $(name) segment"
    subtitle = @sprintf("%d strides · %.1f s · %.1f m travelled",
        n_strides, duration, distance)
    # if !isnothing(gt_traj)
    #     subtitle *= " · legend RMSE over this window"
    # end

    # An in-axis legend sooner or later sits on top of the track, since where the walk
    # goes is data-dependent. Default to a legend outside the axis instead, and widen
    # the figure to pay for it so the plotting area itself does not shrink.
    fig_size = legend_position === :outer_right ? (1050, 600) :
               legend_position === :outer_bottom ? (800, 700) : (800, 600)

    fig = Figure(size=fig_size)
    ax = Axis(fig[1, 1];
        xlabel="X (m)",
        ylabel="Y (m)",
        title=title,
        subtitle=show_subtitle ? subtitle : "",
        aspect=DataAspect(),
        xgridvisible=true)

    entries, labels, series = _draw_tracks!(ax, trajs, gt_traj, win_start, win_stop;
        show_heading=show_heading, heading_stride=heading_stride,
        heading_length=heading_length, marker_stride=marker_stride)

    if legend_position === :outer_right
        Legend(fig[1, 2], entries, labels; framevisible=false)
    elseif legend_position === :outer_bottom
        Legend(fig[2, 1], entries, labels; framevisible=false,
            orientation=:horizontal, nbanks=2, tellwidth=false, tellheight=true)
    else
        # Anything else is treated as an in-axis anchor (:rt, :lt, :rb, ...).
        axislegend(ax, entries, labels; position=legend_position)
    end

    # DataAspect() constrains what is *drawn*, not the layout cell: the cell keeps the
    # whole column width, so the axis floats in the middle of it and a legend in the next
    # column is pushed far off to the right by the leftover space. Size the column to the
    # data instead, then shrink the figure onto the layout so no gap survives. Only
    # :outer_right needs this -- a legend below is centred under the axis, so horizontal
    # slack never strands it.
    if legend_position === :outer_right
        reset_limits!(ax)
        lims = ax.finallimits[]
        if lims.widths[1] > 0 && lims.widths[2] > 0
            # Clamped: a long thin corridor would otherwise collapse the column to a
            # sliver too narrow to hold the title, which trades one bad figure for another.
            ratio = clamp(lims.widths[1] / lims.widths[2], 0.6, 3.0)
            colsize!(fig.layout, 1, Aspect(1, ratio))
            resize_to_layout!(fig)
        end
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end


"""
    plot_trajectory_and_distance_error(trajs, gt_traj; kwargs...)

Top-down 2D trajectory on the left, absolute horizontal distance error on the right, one
shared legend for both.

Both panels cover the *same* window of the track, chosen with `segment`/`train_ratio` (or
an explicit `start`/`stop`) exactly as in [`plot_groundtruth_vs_inertial_positions`](@ref),
so a bump in the error curve can be traced to a place on the map without the reader having
to reconcile two different index ranges. Series colours are shared for the same reason:
both panels are driven by one pass of [`_draw_tracks!`](@ref).

Ground truth appears in the left panel and in the legend but not in the error panel, where
its error is zero by construction.

# Keywords
Everything [`plot_groundtruth_vs_inertial_positions`](@ref) takes, plus:
- `gt_available`: per-step `Bool` vector marking steps where ground truth was fed to the
  filter; those points are dotted onto the error curve, as in
  [`plot_position_distance_error`](@ref). Indexed over the same steps as `trajs`.
- `legend_position`: `:outer_right` (default, a third column) or `:outer_bottom` (one
  horizontal legend spanning both panels).
- `show_subtitle`: drops the stride/duration/distance line from BOTH panels, so their
  plot areas stay aligned.

# Returns
- A `Figure` object (Makie figure).
"""
function plot_trajectory_and_distance_error(
    trajs::AbstractDict{String,Trajectory},
    gt_traj::Trajectory;
    segment::Symbol=:full,
    train_ratio::Union{Nothing,Real}=nothing,
    start::Union{Nothing,Int}=nothing,
    stop::Union{Nothing,Int}=nothing,
    show_heading::Bool=false,
    heading_stride::Int=10,
    heading_length::Float64=0.15,
    marker_stride::Int=0,
    gt_available::Union{Nothing,AbstractVector{Bool}}=nothing,
    segment_label::Union{Nothing,AbstractString}=nothing,
    legend_position::Symbol=:outer_right,
    show_subtitle::Bool=true,
    save_path::Union{String,Nothing}=nothing
)
    n_ref = length(gt_traj.t)
    seg_start, seg_stop = _segment_window(gt_traj, segment, train_ratio)
    win_start = clamp(isnothing(start) ? seg_start : start, 1, n_ref)
    win_stop = clamp(isnothing(stop) ? seg_stop : stop, win_start, n_ref)
    window = win_start:win_stop

    duration = gt_traj.t[win_stop] - gt_traj.t[win_start]
    distance = total_distance(gt_traj[window])

    name = isnothing(segment_label) ?
           (segment === :train ? "Train" : segment === :test ? "Test" : nothing) :
           segment_label
    title = isnothing(name) ?
            "Ground truth vs estimated trajectories" :
            "Ground truth vs estimated trajectories — $(name) segment"
    subtitle = @sprintf("%d strides · %.1f s · %.1f m travelled",
        win_stop - win_start, duration, distance)

    fig = Figure(size=legend_position === :outer_bottom ? (1200, 620) : (1500, 560))

    ax_traj = Axis(fig[1, 1];
        xlabel="X (m)",
        ylabel="Y (m)",
        title=title,
        subtitle=show_subtitle ? subtitle : "",
        aspect=DataAspect(),
        xgridvisible=true)

    entries, labels, series = _draw_tracks!(ax_traj, trajs, gt_traj, win_start, win_stop;
        show_heading=show_heading, heading_stride=heading_stride,
        heading_length=heading_length, marker_stride=marker_stride)

    ax_err = Axis(fig[1, 2];
        xlabel="Time (s)",
        ylabel="Position error (m)",
        title="Absolute distance error",
        # Blank, not absent: it keeps the two panels' plot areas aligned under
        # their titles -- but only while the left panel HAS a subtitle.
        subtitle=show_subtitle ? " " : "",
        xgridvisible=true)

    # Time is measured from the start of the window, not from the start of the recording,
    # so a test segment reads as "0 s in" rather than starting at some arbitrary offset.
    t0 = gt_traj.t[win_start]

    for (traj, c, n) in series
        idx = win_start:min(n, win_stop)
        diff = traj.pos[1:2, idx] .- gt_traj.pos[1:2, idx]
        dist = sqrt.(sum(diff .^ 2, dims=1))[:]

        lines!(ax_err, traj.t[idx] .- t0, dist; color=c, linewidth=1.2)

        if !isnothing(gt_available)
            # Steps where ground truth was actually fed to the filter. Restricted to the
            # drawn window, so this is empty on a :test segment -- which is correct, and
            # the reason it is worth showing on :full.
            marked = [i for i in idx if i <= length(gt_available) && gt_available[i]]
            if !isempty(marked)
                scatter!(ax_err, traj.t[marked] .- t0, dist[marked .- (win_start - 1)];
                    marker=:circle, color=:black, markersize=6,
                    strokewidth=1, strokecolor=:white)
            end
        end
    end

    if legend_position === :outer_bottom
        Legend(fig[2, 1:2], entries, labels; framevisible=false,
            orientation=:horizontal, nbanks=2, tellwidth=false, tellheight=true)
    else
        Legend(fig[1, 3], entries, labels; framevisible=false)
    end

    # Same DataAspect story as the single-panel figure: without this the map panel keeps a
    # full share of the width whatever its shape, and the error panel is squeezed for room
    # the map is not using. Sizing column 1 to the data hands that width back.
    reset_limits!(ax_traj)
    lims = ax_traj.finallimits[]
    if lims.widths[1] > 0 && lims.widths[2] > 0
        ratio = clamp(lims.widths[1] / lims.widths[2], 0.6, 3.0)
        colsize!(fig.layout, 1, Aspect(1, ratio))
        resize_to_layout!(fig)
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end

function plot_trajectory_and_distance_error(
    trajs::Trajectory, gt_traj::Trajectory; kwargs...
)
    plot_trajectory_and_distance_error(Dict("Estimation" => trajs), gt_traj; kwargs...)
end

"""
    plot_position_rmse(trajs::Union{Dict{String, Trajectory}, Trajectory}, gt_traj::Trajectory)

Plot the cumulative root-mean-square error (RMSE) of the horizontal position (x-y) over time
for one or more estimated trajectories against a ground truth trajectory.

# Arguments
- `trajs`: Either a single `Trajectory` (which will be labeled "Estimation") or a dictionary
  mapping labels (e.g. "Estimation", "Filter", etc.) to `Trajectory` objects. Each trajectory
  may optionally have a `name` field; if present it will be used in the legend instead of the
  dictionary key.
- `gt_traj`: Ground truth `Trajectory` (only its first two position coordinates are used).

# Returns
- A `Plots.Plot` object (the figure).

# Details
The horizontal RMSE at time step k is defined as:
RMSE(k) = sqrt( (1/k) * Σ_{i=1..k} ( (x̂_i - x_i)² + (ŷ_i - y_i)² ) )

Only the overlapping portion of the trajectories (minimum number of samples) is used.
"""
function plot_position_rmse(
    trajs::Union{AbstractDict{String,Trajectory},Trajectory},
    gt_traj::Trajectory;
    show_index_ticks::Bool=true
)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        xlabel="Time (s)",
        ylabel="RMSE (m)",
        title="Position RMSE over time",
        xgridvisible=true)

    # -- plot all trajectories, shifting time to start at 0 --
    for (key, traj) in trajs
        n = min(size(traj.pos, 2), size(gt_traj.pos, 2))
        cum_rmse = rmse(traj, gt_traj)
        Δrmse = cum_rmse[end] - cum_rmse[1]
        t_shifted = traj.t[1:n] .- traj.t[1]
        label = "$key, RMSE: $(round(cum_rmse[end], digits=3)), RMSE rate: $(@sprintf("%.2e", Δrmse/total_distance(gt_traj)))"
        lines!(ax, t_shifted, cum_rmse, label=label)
    end

    axislegend(ax; position=:lt)

    # -- add a twin x-axis for index (top) --
    if show_index_ticks
        first_traj = first(values(trajs))
        n = min(size(first_traj.pos, 2), size(gt_traj.pos, 2))
        t_shifted = first_traj.t[1:n] .- first_traj.t[1]

        step = max(1, n ÷ 10)
        idx_ticks = 1:step:n
        time_ticks = t_shifted[idx_ticks]

        ax_top = Axis(fig[1, 1];
            xaxisposition=:top,
            yaxisposition=:right,
            ylabelvisible=false,
            yticksvisible=false,
            yticklabelsvisible=false,
            xlabel="Sample index",
            xticks=(time_ticks, string.(collect(idx_ticks))),
            xticklabelrotation=0)

        linkxaxes!(ax, ax_top)
    end

    return fig
end

function plot_position_rmse(
    trajs::Union{AbstractDict{String,Trajectory},Trajectory},
    gt_traj::Trajectory,
    train_ratio::Float64;
    show_index_ticks::Bool=true
)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end
    N = length(gt_traj)
    n_train_cutoff = max(1, floor(Int, train_ratio * N))

    truncated_trajs = OrderedDict{String,Trajectory}()
    for (name, traj) in trajs
        @assert length(traj) == N "Incompatible trajectory $name with ground truth"
        truncated_trajs[name] = traj[n_train_cutoff:end]
    end
    plot_position_rmse(truncated_trajs, gt_traj[n_train_cutoff:end]; show_index_ticks=show_index_ticks)
end


"""
    plot_groundtruth_vs_inertial_orientations(trajs::Union{Dict{String,Trajectory},Trajectory},
                                               gt_traj::Trajectory)

Plot Euler angles (roll, pitch, yaw) from estimated trajectories against a ground truth.

# Arguments
- `trajs`: Either a single `Trajectory` (labelled "Estimation") or a dictionary mapping labels
  (e.g. "Estimation", "Filter") to `Trajectory` objects. Each trajectory must provide:
  - `t::Vector{Float64}` time vector
  - `euler_nb::Matrix{Float64}` Euler angles (3×N) in radians (roll=row1, pitch=row2, yaw=row3)
- `gt_traj`: Ground truth trajectory with the same `t` and `euler_nb` fields.

# Returns
- A `Figure` object with three side-by-side axis objects.
"""
function plot_groundtruth_vs_inertial_orientations(
    trajs::Union{AbstractDict{String,Trajectory},Trajectory},
    gt_traj::Trajectory
)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    labels = ["Roll", "Pitch", "Yaw"]
    fig = Figure(size=(1200, 400))

    # Create three axes in a row
    axs = [Axis(fig[1, i]; title=labels[i], xlabel="Time (s)", ylabel="Degrees",
        xgridvisible=true) for i in 1:3]

    # Storage for legend handles (one per estimated trajectory, using its first subplot colour)
    legend_handles = []

    for (key, ins_traj) in trajs
        ins_euler_deg = rad2deg.(matrix_to_euler(ins_traj.R_nb))
        # Plot all three angles for this trajectory
        for i in 1:3
            line = lines!(axs[i], ins_traj.t, ins_euler_deg[i, :];
                linewidth=0.8, label=(i == 1 ? key : ""))  # label only on first subplot
            if i == 1
                push!(legend_handles, line)
            end
        end
    end

    # Plot ground truth (dashed black line) on all subplots
    gt_euler_deg = rad2deg.(matrix_to_euler(gt_traj.R_nb))
    for i in 1:3
        lines!(axs[i], gt_traj.t, gt_euler_deg[i, :];
            color=:black, linestyle=:dash, linewidth=0.8,
            label=(i == 1 ? "Ground truth" : ""))
    end

    # Add legend on the first axis only
    # Automatic legend on the first axis (collects all lines with labels)
    axislegend(axs[1]; position=:rt)   # :rt = right top, you can also use :rb, :lt, etc.

    # Adjust layout to prevent legend overlap
    colgap!(fig.layout, 5)
    resize_to_layout!(fig)

    return fig
end

"""
    plot_groundtruth_vs_inertial_xyz(trajs::Union{AbstractDict{String,Trajectory},Trajectory},
                                           gt_traj::Trajectory)

Plot position components (X, Y, Z) from estimated trajectories against ground truth.

# Arguments
- `trajs`: Either a single `Trajectory` (labelled "Estimation") or a dictionary mapping labels
  (e.g. "Estimation", "Filter") to `Trajectory` objects. Each trajectory must provide:
  - `t::Vector{Float64}` time vector
  - `pos::Matrix{Float64}` position (3×N) in meters (x=row1, y=row2, z=row3)
- `gt_traj`: Ground truth trajectory with the same `t` and `pos` fields.

# Returns
- A `Figure` object with three side-by-side axis objects (X, Y, Z).
"""
function plot_groundtruth_vs_inertial_xyz(
    trajs::Union{AbstractDict{String,Trajectory},Trajectory},
    gt_traj::Trajectory
)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    labels = ["X (m)", "Y (m)", "Z (m)"]
    fig = Figure(size=(1200, 400))

    # Create three axes in a row
    axs = [Axis(fig[1, i]; title=labels[i], xlabel="Time (s)", ylabel="Position (m)",
        xgridvisible=true) for i in 1:3]

    legend_handles = []

    for (key, ins_traj) in trajs
        # Plot all three position components for this trajectory
        for i in 1:3
            line = lines!(axs[i], ins_traj.t, ins_traj.pos[i, :];
                linewidth=0.8, label=(i == 1 ? key : ""))
            if i == 1
                push!(legend_handles, line)
            end
        end
    end

    # Plot ground truth (dashed black line) on all subplots
    for i in 1:3
        lines!(axs[i], gt_traj.t, gt_traj.pos[i, :];
            color=:black, linestyle=:dash, linewidth=0.8,
            label=(i == 1 ? "Ground truth" : ""))
    end

    # Add legend on the first axis only
    axislegend(axs[1]; position=:rt)

    # Adjust layout
    colgap!(fig.layout, 5)
    resize_to_layout!(fig)

    return fig
end

"""
    plot_trajectory_3d(
    trajs::Union{Trajectory, Dict{String,Trajectory}};
    gt_traj::Union{Nothing,Trajectory}=nothing,
    start_marker::Bool=true, end_marker::Bool=true
)

Plot 3D positions (x, y, z) from one or more trajectories using an interactive GLMakie window.

# Arguments
- `trajs`: Either a single `Trajectory` (labelled "Estimation") or a dictionary mapping
  labels (e.g. "Estimation", "GP corrected") to `Trajectory` objects. Each trajectory
  must provide `pos::Matrix{Float64}` (3×N) with columns as time steps.
- `gt_traj`: Optional ground truth trajectory (plotted as a dashed black line).
- `start_marker`: If `true`, mark the start point of each trajectory with a green sphere.
- `end_marker`: If `true`, mark the end point with a red sphere.

# Returns
- A `Figure` object with a 3D axis.
"""
function plot_trajectory_3d(
    trajs::Union{Trajectory,AbstractDict{String,Trajectory}},
    gt_traj::Union{Nothing,Trajectory}=nothing;
    samples::Int=1000000000,
    start_marker::Bool=true,
    end_marker::Bool=true)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    fig = Figure(size=(1000, 800))
    ax = Axis3(fig[1, 1];
        title="3D Trajectory",
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        aspect=(1, 1, 1))

    colours = [:steelblue, :orange, :seagreen, :purple, :tomato]

    for (i, (label, traj)) in enumerate(trajs)
        pos = traj.pos
        colour = colours[(i-1)%length(colours)+1]
        n = min(samples, size(pos, 2))
        lines!(ax, pos[1, 1:n], pos[2, 1:n], pos[3, 1:n];
            color=colour, linewidth=2, label=label)

        if start_marker
            scatter!(ax, [pos[1, 1]], [pos[2, 1]], [pos[3, 1]];
                color=:green, markersize=12, marker=:circle,
                label=i == 1 ? "Start" : "")
        end
        if end_marker
            scatter!(ax, [pos[1, n]], [pos[2, n]], [pos[3, n]];
                color=:red, markersize=12, marker=:circle,
                label=i == 1 ? "End" : "")
        end
    end

    if gt_traj !== nothing
        pos_gt = gt_traj.pos
        n = min(samples, size(pos_gt, 2))
        lines!(ax, pos_gt[1, 1:n], pos_gt[2, 1:n], pos_gt[3, 1:n];
            color=:black, linewidth=1.5, linestyle=:dash, label="Ground truth")
    end

    axislegend(ax; position=:rb)
    return fig
end

"""
    plot_position_distance_error(trajs::Union{Dict{String,Trajectory},Trajectory},
                                 gt_traj::Trajectory)

Plot the horizontal position error (Euclidean distance) over time for one or more
estimated trajectories against a ground truth trajectory.

# Arguments
- `trajs`: Either a single `Trajectory` (labelled "Estimation") or a dictionary mapping
  labels (e.g. "Estimation", "Filter") to `Trajectory` objects. Each trajectory may
  optionally have a `name` field used in the legend.
- `gt_traj`: Ground truth trajectory (only the first two position coordinates are used).

# Returns
- A `Figure` object containing the distance plot.
"""
function plot_position_distance_error(
    trajs::Union{AbstractDict{String,Trajectory},Trajectory},
    gt_traj::Trajectory
)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        xlabel="Time (s)",
        ylabel="Position error (m)",
        title="Absolute Distance Error",
        xgridvisible=true)

    for (key, traj) in trajs
        n = min(size(traj.pos, 2), size(gt_traj.pos, 2))
        # Horizontal distance error per sample (no cumulative sum)
        diff = traj.pos[1:2, 1:n] .- gt_traj.pos[1:2, 1:n]
        dist = sqrt.(sum(diff .^ 2, dims=1))[:]   # (n,)

        # Use trajectory name if available, else the dictionary key
        lines!(ax, traj.t[1:n], dist; linewidth=1.2, label=key)
    end

    axislegend(ax; position=:rt)
    return fig
end
function plot_position_distance_error(
    trajs::Union{AbstractDict{String,Trajectory},Trajectory},
    gt_traj::Trajectory,
    bools::AbstractVector{Bool}
)
    # Wrap single trajectory in a dict
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    # Find indices where bools is true
    idxs = findall(bools)
    if isempty(idxs)
        @warn "No true values in bools; plotting without markers."
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        xlabel="Time (s)",
        ylabel="Position error (m)",
        title="Absolute Distance Error",
        xgridvisible=true)

    for (key, traj) in trajs
        n = min(size(traj.pos, 2), size(gt_traj.pos, 2))
        # Horizontal distance error per sample (no cumulative sum)
        diff = traj.pos[1:2, 1:n] .- gt_traj.pos[1:2, 1:n]
        dist = sqrt.(sum(diff .^ 2, dims=1))[:]   # (n,)

        # Plot the error line
        lines!(ax, traj.t[1:n], dist; linewidth=1.2, label=key)

        # Overlay dots at positions where bools is true and within the valid time range
        valid_idxs = filter(i -> i ≤ n, idxs)
        if !isempty(valid_idxs)
            scatter!(ax, traj.t[valid_idxs], dist[valid_idxs];
                marker=:circle, color=:black, markersize=8,
                strokewidth=1, strokecolor=:white)
        end
    end

    axislegend(ax; position=:rt)
    return fig
end

function plot_trajectory_xyz_euler(traj::Trajectory; figsize=(1200, 800))
    """
    Plot the XYZ position and Euler angles of a single trajectory.

    # Arguments
    - `traj`: Trajectory object with fields:
        - t::Vector{Float64}
        - pos::Matrix{Float64} (3×N)
        - euler_nb::Matrix{Float64} (3×N) angles in radians (roll, pitch, yaw)
    - `figsize`: (width, height) in pixels.
    """
    t = traj.t
    pos = traj.pos
    euler_deg = rad2deg.(matrix_to_euler(traj.R_nb))

    fig = Figure(size=figsize)
    # 2 rows, 3 columns
    ax_pos_x = Axis(fig[1, 1]; xlabel="Time (s)", ylabel="X (m)", title="Position components")
    ax_pos_y = Axis(fig[1, 2]; xlabel="Time (s)", ylabel="Y (m)")
    ax_pos_z = Axis(fig[1, 3]; xlabel="Time (s)", ylabel="Z (m)")

    ax_roll = Axis(fig[2, 1]; xlabel="Time (s)", ylabel="Roll (deg)", title="Euler angles")
    ax_pitch = Axis(fig[2, 2]; xlabel="Time (s)", ylabel="Pitch (deg)")
    ax_yaw = Axis(fig[2, 3]; xlabel="Time (s)", ylabel="Yaw (deg)")

    # Plot positions
    lines!(ax_pos_x, t, pos[1, :]; color=:blue, linewidth=1.5)
    lines!(ax_pos_y, t, pos[2, :]; color=:blue, linewidth=1.5)
    lines!(ax_pos_z, t, pos[3, :]; color=:blue, linewidth=1.5)

    # Plot Euler angles (converted to degrees)
    lines!(ax_roll, t, euler_deg[1, :]; color=:red, linewidth=1.5)
    lines!(ax_pitch, t, euler_deg[2, :]; color=:red, linewidth=1.5)
    lines!(ax_yaw, t, euler_deg[3, :]; color=:red, linewidth=1.5)

    return fig
end

function indices_and_ages_within_lifetime(times::Vector{Float64}, current_idx::Int, lifetime::Float64)
    current_time = times[current_idx]
    ages = current_time .- times[1:current_idx]
    mask = ages .<= lifetime
    return findall(mask), ages[mask]
end

function ring_points(center::Point3f, radius::Float32, z_offset::Float32)
    θs = range(0, 2π, length=64)
    push!(
        [Point3f(center[1] + radius * cos(θ), center[2] + radius * sin(θ), center[3] + z_offset) for θ in θs],
        Point3f(center[1] + radius, center[2], center[3] + z_offset)  # close the loop
    )
end

function make_ring_data(
    positions,
    times,
    footfall_idxs,
    current_idx,
    lifetime,
    inferno;
    radius0::Float32=0.05f0,
    radius_growth::Float32=0.15f0,
    radius_scale::Float32=1f0,
    alpha_fn=x -> 1f0 - x,
    z_phase::Float32=0.0f0
)
    ff = footfall_idxs[footfall_idxs .≤ current_idx]
    isempty(ff) && return (Point3f[], RGBAf[])

    ages = times[current_idx] .- times[ff]

    mask = ages .≤ lifetime
    within = ff[mask]
    within_ages = ages[mask]

    isempty(within) && return (Point3f[], RGBAf[])

    pts = Point3f[]
    colors = RGBAf[]

    for (j, age) in zip(within, within_ages)
        frac = Float32(age / lifetime)

        # Sample inferno: newest footfalls near 0.65, oldest near 0.0
        cmap_pos = 0.65f0 * (1f0 - frac)
        c = get(inferno, cmap_pos)
        # c = inferno(cmap_pos)

        radius = (radius0 + radius_growth * frac) * radius_scale
        z_offset = -Float32(0.02f0 * sin(age * 2pi / (lifetime / 1.7) + z_phase))
        α = clamp(alpha_fn(frac), 0f0, 1f0)
        color = RGBAf(c.r, c.g, c.b, α)

        ring = ring_points(Point3f(positions[:, j]...), radius, z_offset)
        append!(pts, ring)
        append!(colors, fill(color, length(ring)))

        push!(pts, Point3f(NaN, NaN, NaN))
        push!(colors, RGBAf(0, 0, 0, 0))
    end

    return (pts, colors)
end


function animate_trajectory(
    traj::Trajectory, segs::Vector{Int};
    lifetime::Float64=1.0,
    framerate::Int=60,
    filepath::String="trajectory_animation.mp4",
    padding::Float64=0.3,
    scale::Float32=2.1f0
)
    inferno = cgrad(:viridis)
    curr_color_frac = 0.9f0
    current_color = get(inferno, curr_color_frac)
    current_color_rgba = RGBAf(current_color.r, current_color.g, current_color.b, 1f0)

    positions = traj.pos
    times = traj.t
    N = length(times)
    footfall_idxs = segs

    current_idx = Observable(1)

    # ── Trail data ──────────────────────────────────────────────────────────
    trail_points = @lift begin
        trail_idxs, _ = indices_and_ages_within_lifetime(times, $current_idx, lifetime)
        length(trail_idxs) < 2 && return Point3f[]
        pts = Point3f[]
        for k in 1:(length(trail_idxs)-1)
            push!(pts,
                Point3f(positions[:, trail_idxs[k]]...),
                Point3f(positions[:, trail_idxs[k+1]]...))
        end
        pts
    end

    trail_colors = @lift begin
        trail_idxs, trail_ages = indices_and_ages_within_lifetime(times, $current_idx, lifetime)
        length(trail_idxs) < 2 && return RGBAf[]
        map(1:(length(trail_idxs)-1)) do k
            # Use the older end of each segment to determine color
            frac = Float32(trail_ages[k+1] / lifetime)
            # Sample inferno: newest near 0.95, oldest near 0.0
            cmap_pos = curr_color_frac * (1f0 - frac)
            c = get(inferno, cmap_pos)
            α = clamp(1f0 - frac, 0f0, 1f0)
            RGBAf(c.r, c.g, c.b, α)
        end
    end

    # ── Current marker ───────────────────────────────────────────────────────
    current_point = @lift [Point3f(positions[:, $current_idx]...)]

    # ── Footfall data ────────────────────────────────────────────────────────
    footfall_ring_data = @lift begin
        make_ring_data(positions, times, footfall_idxs, $current_idx, lifetime * 0.5, inferno;
            alpha_fn=x -> 1f0 - x,
            radius_scale=1f0 * scale,
            z_phase=0.0f0
        )
    end

    footfall_ring_data_soft = @lift begin
        make_ring_data(positions, times, footfall_idxs, $current_idx, lifetime * 0.5, inferno;
            alpha_fn=x -> 0.3f0 * (1f0 - x),
            radius_scale=1.3f0 * scale,
            z_phase=Float32(pi * 3 / 4)
        )
    end

    # ── Camera azimuth ───────────────────────────────────────────────────────────
    azimuth = @lift deg2rad(30 + 30 * sin(2π * $current_idx / (framerate * 30)))

    # ── Figure & plots ────────────────────────────────────────────────────────
    fig = Figure(size=(1000, 800))
    ax = Axis3(fig[1, 1]; aspect=:data, title="Trajectory Animation")
    limits!(ax,
        minimum(positions[1, :]) - padding, maximum(positions[1, :]) + padding,
        minimum(positions[2, :]) - padding, maximum(positions[2, :]) + padding,
        minimum(positions[3, :]) - padding, maximum(positions[3, :]) + padding
    )
    linesegments!(ax, trail_points; color=trail_colors, linewidth=4 * scale, transparency=true, fxaa=true, ssao=true)
    scatter!(ax, current_point; color=current_color_rgba, markersize=12 * scale)
    lines!(ax, @lift($footfall_ring_data[1]);
        color=@lift($footfall_ring_data[2]),
        linewidth=2 * scale,
        transparency=true, fxaa=true
    )
    lines!(ax, @lift($footfall_ring_data_soft[1]);
        color=@lift($footfall_ring_data_soft[2]),
        linewidth=6 * scale,
        transparency=true, fxaa=true
    )
    connect!(ax.azimuth, azimuth)

    record(fig, filepath, 1:N; framerate=framerate, profile="high", pixel_format="yuv420p") do i
        current_idx[] = i
    end

    return fig
end

function plot_trajectory(positions::Vector{Point3f}, Rs::Vector{<:AbstractMatrix}; axis_len=0.3, title="Trajectory")
    N = length(positions)
    @assert length(Rs) == N

    fig = Figure(size=(900, 700))
    ax = Axis3(fig[1, 1], aspect=:data, title=title)

    # slider – use update_while_dragging=false if desired
    sl = Slider(fig[2, 1], range=1:N, startvalue=1, update_while_dragging=true)

    # --- Trail: lift on slider value ---
    trail = lift(sl.value) do idx
        positions[1:idx]     # returns Vector{Point3f}
    end
    lines!(ax, trail, color=:dodgerblue, linewidth=3)

    # --- full path: lift on slider value ---
    trail = lift(sl.value) do idx
        positions[(idx):end]     # returns Vector{Point3f}
    end
    lines!(ax, trail, color=:gray70, linewidth=1)

    # --- Current point ---
    current_pos = lift(sl.value) do idx
        positions[idx]       # returns Point3f
    end
    scatter!(ax, current_pos, color=:black, markersize=12)

    # --- Body axes segments ---
    for (i, col) in enumerate(1:3)
        color = (:red, :green, :blue)[i]
        seg = lift(sl.value) do idx
            p = positions[idx]
            R = Rs[idx]
            dir = Point3f(R[:, col])            # force Float32
            Point3f[p, p+dir*Float32(axis_len)]   # concretely typed vector
        end
        lines!(ax, seg, color=color, linewidth=4)
    end

    # --- Label (can also use @lift or lift) ---
    lbl = lift(sl.value) do idx
        "Time step: $idx / $N"
    end
    Label(fig[0, 1], lbl, tellwidth=false)

    fig
end

function plot_trajectory(traj::Trajectory; axis_len=0.3, kwargs...)
    n = length(traj)
    points = Vector{Point3f}(undef, n)
    mats = Vector{Matrix{Float32}}(undef, n)
    for i in 1:n
        points[i] = Point3f(traj.pos[:, i])
        mats[i] = traj.R_nb[:, :, i]
    end
    plot_trajectory(points, mats; axis_len=axis_len, kwargs...)
end

"""
Plot multiple channels (rows of `data`) versus `time`.

# Arguments
- `data`: `n_channel × N` matrix, each row is a channel.
- `time`: time vector of length N (defaults to indices 1:N).
- `labels`: vector of strings of length `n_channel` for legend.
            Defaults to "ch1", "ch2", ...
- `offset`: if `nothing`, all lines are overlaid.
            If a number, each channel is shifted vertically by
            `(i-1) * offset` (e.g., for stacked EEG).
- `colormap`: any Makie colormap (Symbol or vector of colors) – 
                for categorical colors use `:tab10`, `:Set1`, etc.
- `kwargs...`: passed to `lines!` (e.g., `linewidth=1.5`).
"""
function plot_channels(data::Matrix, time::AbstractVector=axes(data, 2);
    labels::Union{AbstractVector,Nothing}=nothing,
    offset::Union{Real,Nothing}=nothing,
    colormap::Union{Symbol,AbstractVector}=:tab10,
    kwargs...)

    n_ch, N = size(data)
    labels = isnothing(labels) ? ["ch$i" for i in 1:n_ch] : labels
    time = collect(time)

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="Amplitude",
        title="Channel Plot")

    # Get categorical colors – this replaces the deprecated `to_colormap`
    colors = Makie.categorical_colors(colormap, n_ch)

    offsets = isnothing(offset) ? zeros(n_ch) : (0:(n_ch-1)) * offset

    for i in 1:n_ch
        y = data[i, :] .+ offsets[i]
        lines!(ax, time, y; color=colors[i], label=labels[i], kwargs...)
    end

    if n_ch > 1
        Legend(fig[2, :], ax, "Channels"; orientation=:horizontal, tellwidth=false)
    end

    return fig
end