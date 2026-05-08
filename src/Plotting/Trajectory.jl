"""
    plot_groundtruth_vs_inertial_positions(trajs, gt_traj)

Plot 2D positions (X‑Y) of ground truth and one or more estimated trajectories.

# Arguments
- `trajs`: A `Trajectory` object (single estimate) or a `Dict{String,Trajectory}` (multiple).
- `gt_traj`: Ground truth `Trajectory`.

# Returns
- A `Figure` object (Makie figure).
"""
function plot_groundtruth_vs_inertial_positions(trajs::Trajectory, gt_traj::Trajectory)
    plot_groundtruth_vs_inertial_positions(Dict("Estimation" => trajs), gt_traj)
end

function plot_groundtruth_vs_inertial_positions(trajs::Dict{String,Trajectory}, gt_traj::Trajectory)
    # Find common length (shortest trajectory)
    n = size(gt_traj.pos, 2)
    for traj in values(trajs)
        n = min(n, size(traj.pos, 2))
    end

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