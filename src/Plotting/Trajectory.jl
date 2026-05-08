"""
    plot_groundtruth_vs_inertial_positions(trajs, gt_traj)

Plot 2D positions (X‑Y) of ground truth and one or more estimated trajectories.

# Arguments
- `trajs`: A `Trajectory` object (single estimate) or a `Dict{String,Trajectory}` (multiple).
- `gt_traj`: Ground truth `Trajectory`.

# Returns
- A `Plots.Plot` object.
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

    # Ground truth line (dashed black)
    p = plot(gt_traj.pos[1, 1:n], gt_traj.pos[2, 1:n];
        label="Ground truth",
        line=(:black, :dash),
        linewidth=1,
        aspect_ratio=:equal,
        grid=true,
        xlabel="X (m)",
        ylabel="Y (m)")

    # Ground truth start (circle) and end (square)
    scatter!(p, [gt_traj.pos[1, 1]], [gt_traj.pos[2, 1]];
        label="Start", color=:black, marker=:circle, markersize=6)
    scatter!(p, [gt_traj.pos[1, n]], [gt_traj.pos[2, n]];
        label="End", color=:black, marker=:square, markersize=6)

    # Colour palette for the estimated trajectories
    n_traj = length(trajs)
    colors = palette(:tab10, n_traj)

    for (i, (key, traj)) in enumerate(trajs)
        c = colors[i]
        # Line
        plot!(p, traj.pos[1, 1:n], traj.pos[2, 1:n];
            label=key, linecolor=c, linewidth=1)
        # Start marker (circle)
        scatter!(p, [traj.pos[1, 1]], [traj.pos[2, 1]];
            label=nothing, color=c, marker=:circle, markersize=6)
        # End marker (square)
        scatter!(p, [traj.pos[1, n]], [traj.pos[2, n]];
            label=nothing, color=c, marker=:square, markersize=6)
    end

    return p
end