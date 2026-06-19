"""
    plot_groundtruth_vs_inertial_positions(trajs, gt_traj)

Plot 2D positions (X‑Y) of ground truth and one or more estimated trajectories.

# Arguments
- `trajs`: A `Trajectory` object (single estimate) or a `Dict{String,Trajectory}` (multiple).
- `gt_traj`: Ground truth `Trajectory`.

# Returns
- A `Figure` object (Makie figure).
"""
function plot_groundtruth_vs_inertial_positions(
    trajs::Trajectory, gt_traj::Union{Nothing,Trajectory}; samples::Int=typemax(Int))
    plot_groundtruth_vs_inertial_positions(Dict("Estimation" => trajs), gt_traj; samples=samples)
end

function plot_groundtruth_vs_inertial_positions(
    trajs::AbstractDict{String,Trajectory},
    gt_traj::Union{Nothing,Trajectory};
    start::Int=1,
    stop::Int=typemax(Int)
)

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        xlabel="X (m)",
        ylabel="Y (m)",
        title="Ground truth vs estimated trajectories",
        aspect=DataAspect(),
        xgridvisible=true)

    if !isnothing(gt_traj)
        n = min(length(gt_traj.t), stop)
        start = min(length(gt_traj.t), start)
        lines!(ax, gt_traj.pos[1, start:n], gt_traj.pos[2, start:n];
            color=:black, linestyle=:dash, linewidth=1, label="Ground truth")
        scatter!(ax, gt_traj.pos[1, start:n], gt_traj.pos[2, start:n];
            color=:black, marker=:circle, markersize=5, alpha=0.6)

        scatter!(ax, [gt_traj.pos[1, start]], [gt_traj.pos[2, start]];
            color=:black, marker=:circle, markersize=12, label="Start")
        scatter!(ax, [gt_traj.pos[1, n]], [gt_traj.pos[2, n]];
            color=:black, marker=:rect, markersize=12, label="End")
    end

    colors = Makie.wong_colors()
    color_cycle = Iterators.cycle(colors)

    for (i, (key, traj)) in enumerate(trajs)
        c = first(color_cycle)
        color_cycle = Iterators.drop(color_cycle, 1)

        n = min(length(traj.t), stop)

        # Line
        lines!(ax, traj.pos[1, start:n], traj.pos[2, start:n];
            color=c, linewidth=1, label=key)

        # Small markers at every data point
        scatter!(ax, traj.pos[1, start:n], traj.pos[2, start:n];
            color=c, marker=:circle, markersize=5, alpha=0.6)

        # Start marker (larger circle)
        scatter!(ax, [traj.pos[1, start]], [traj.pos[2, start]];
            color=c, marker=:circle, markersize=12)

        # End marker (larger square)
        scatter!(ax, [traj.pos[1, n]], [traj.pos[2, n]];
            color=c, marker=:rect, markersize=12)
    end

    axislegend(ax; position=:rt)
    return fig
end


"""
    plot_position_rmse(trajs::Union{Dict{String, Trajectory}, Trajectory}, gt_traj::Trajectory)

Plot the cumulative root‑mean‑square error (RMSE) of the horizontal position (x‑y) over time
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

    # -- plot all trajectories --
    for (key, traj) in trajs
        n = min(size(traj.pos, 2), size(gt_traj.pos, 2))
        cum_rmse = rmse(traj, gt_traj)  # presumably computes cumulative RMSE
        label = "$key, RMSE: $(round(cum_rmse[end], digits=3)), RMSE rate: $(@sprintf("%.2e", cum_rmse[end]/total_distance(gt_traj)))"
        lines!(ax, traj.t[1:n], cum_rmse, label=label)
    end
    axislegend(ax)

    # -- add a twin x-axis for index (top) --
    if show_index_ticks
        # pick the first trajectory to get a representative time vector
        first_traj = first(values(trajs))
        n = min(size(first_traj.pos, 2), size(gt_traj.pos, 2))
        t_vec = first_traj.t[1:n]

        # choose a reasonable number of index ticks (e.g., ~10)
        step = max(1, n ÷ 10)
        idx_ticks = 1:step:n
        time_ticks = t_vec[idx_ticks]

        # create the top axis, linked to the main axis
        ax_top = Axis(fig[1, 1];
            xaxisposition=:top,
            yaxisposition=:right,        # avoids overlapping y-axis
            ylabelvisible=false,
            xlabel="Sample index",
            xticks=(time_ticks, string.(idx_ticks)),
            xticklabelrotation=0,
        )
        linkxaxes!(ax, ax_top)
    end

    return fig
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
- A `Figure` object with three side‑by‑side axis objects.
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
- A `Figure` object with three side‑by‑side axis objects (X, Y, Z).
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
    ff = footfall_idxs[footfall_idxs.≤current_idx]
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
    set_theme!(theme_black())

    inferno = cgrad(:inferno)

    current_color = get(inferno, 0.95)
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
        for k in 1:length(trail_idxs)-1
            push!(pts,
                Point3f(positions[:, trail_idxs[k]]...),
                Point3f(positions[:, trail_idxs[k+1]]...))
        end
        pts
    end

    trail_colors = @lift begin
        trail_idxs, trail_ages = indices_and_ages_within_lifetime(times, $current_idx, lifetime)
        length(trail_idxs) < 2 && return RGBAf[]
        map(1:length(trail_idxs)-1) do k
            # Use the older end of each segment to determine color
            frac = Float32(trail_ages[k+1] / lifetime)
            # Sample inferno: newest near 0.95, oldest near 0.0
            cmap_pos = 0.95f0 * (1f0 - frac)
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
    fig = Figure(size=(800, 600))
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

    record(fig, filepath, 1:N; framerate=framerate) do i
        current_idx[] = i
    end

    return fig
end