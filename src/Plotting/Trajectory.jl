"""
    plot_groundtruth_vs_inertial_positions(trajs, gt_traj)

Plot 2D positions (X‑Y) of ground truth and one or more estimated trajectories.

# Arguments
- `trajs`: A `Trajectory` object (single estimate) or a `Dict{String,Trajectory}` (multiple).
- `gt_traj`: Ground truth `Trajectory`.

# Returns
- A `Figure` object (Makie figure).
"""
function plot_groundtruth_vs_inertial_positions(trajs::Trajectory, gt_traj::Trajectory; samples::Int=20)
    plot_groundtruth_vs_inertial_positions(Dict("Estimation" => trajs), gt_traj; samples=samples)
end

function plot_groundtruth_vs_inertial_positions(
    trajs::AbstractDict{String,Trajectory},
    gt_traj::Trajectory;
    samples::Int=20
)
    # Find common length (shortest trajectory)
    n = size(gt_traj.pos, 2)
    for traj in values(trajs)
        n = min(n, size(traj.pos, 2))
    end
    n = min(n, samples)

    # Create figure and axis
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        xlabel="X (m)",
        ylabel="Y (m)",
        title="Ground truth vs estimated trajectories",
        aspect=DataAspect(),
        xgridvisible=true)

    # Ground truth line (dashed black)
    lines!(ax, gt_traj.pos[1, 1:n], gt_traj.pos[2, 1:n];
        color=:black, linestyle=:dash, linewidth=1, label="Ground truth")

    # Ground truth start (circle) and end (square)
    scatter!(ax, [gt_traj.pos[1, 1]], [gt_traj.pos[2, 1]];
        color=:black, marker=:circle, markersize=12, label="Start")
    scatter!(ax, [gt_traj.pos[1, n]], [gt_traj.pos[2, n]];
        color=:black, marker=:rect, markersize=12, label="End")

    # Colour palette for the estimated trajectories (tab10 equivalent)
    colors = Makie.wong_colors()   # gives 9 distinct colours (or use ColorSchemes.tab10)
    # If more trajectories than colors, cycle
    color_cycle = Iterators.cycle(colors)

    for (i, (key, traj)) in enumerate(trajs)
        c = first(color_cycle)   # get next colour
        color_cycle = Iterators.drop(color_cycle, 1)

        # Line
        lines!(ax, traj.pos[1, 1:n], traj.pos[2, 1:n];
            color=c, linewidth=1, label=key)

        # Start marker (circle)
        scatter!(ax, [traj.pos[1, 1]], [traj.pos[2, 1]];
            color=c, marker=:circle, markersize=12)

        # End marker (square)
        scatter!(ax, [traj.pos[1, n]], [traj.pos[2, n]];
            color=c, marker=:rect, markersize=12)
    end

    # Add legend
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
function plot_position_rmse(trajs::Union{AbstractDict{String,Trajectory},Trajectory}, gt_traj::Trajectory)
    if trajs isa Trajectory
        trajs = Dict("Estimation" => trajs)
    end

    # Create figure and axis
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        xlabel="Time (s)",
        ylabel="RMSE (m)",
        title="Position RMSE over time",
        xgridvisible=true)

    for (key, traj) in trajs
        # Number of common time steps
        n = min(size(traj.pos, 2), size(gt_traj.pos, 2))
        # Horizontal errors at each time step (squared, per sample)
        cum_rmse = rmse(traj, gt_traj)
        @show typeof(cum_rmse)
        @show size(cum_rmse)

        # Legend label: use traj.name if present, otherwise create from key
        if hasproperty(traj, :name) && !isnothing(traj.name)
            label = traj.name
        else
            label = "$key, RMSE: $(round(cum_rmse[end], digits=3))"
        end

        lines!(ax, traj.t[1:n], cum_rmse, label=label)
    end
    axislegend()

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
        N = size(pos, 2)
        colour = colours[(i-1)%length(colours)+1]

        lines!(ax, pos[1, :], pos[2, :], pos[3, :];
            color=colour, linewidth=2, label=label)

        if start_marker
            scatter!(ax, [pos[1, 1]], [pos[2, 1]], [pos[3, 1]];
                color=:green, markersize=12, marker=:circle,
                label=i == 1 ? "Start" : "")
        end
        if end_marker
            scatter!(ax, [pos[1, end]], [pos[2, end]], [pos[3, end]];
                color=:red, markersize=12, marker=:circle,
                label=i == 1 ? "End" : "")
        end
    end

    if gt_traj !== nothing
        pos_gt = gt_traj.pos
        lines!(ax, pos_gt[1, :], pos_gt[2, :], pos_gt[3, :];
            color=:black, linewidth=1.5, linestyle=:dash, label="Ground truth")
    end

    axislegend(ax; position=:rb)
    return fig
end