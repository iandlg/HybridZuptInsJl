"""
    plot_step_lengths(trajs, gt_traj, segs) -> Figure

3D step lengths between consecutive step ends `segs` for one or more `Trajectory`s and
the ground truth.
"""
function plot_step_lengths(
    trajs::Union{Vector{Trajectory},Trajectory},
    gt_traj::Union{Nothing,Trajectory},
    segs::Vector{Int}
)
    if trajs isa Trajectory
        trajs = [trajs]
    end


    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1];
        title="Length of steps",
        xlabel="Time [s]",
        ylabel="Step length (m)",
        xgridvisible=true)

    # Plot ground truth
    if !isnothing(gt_traj)
        # Compute ground‑truth step lengths (dashed black line)
        gt_lengths = step_lengths(gt_traj, segs)

        gt_times = gt_traj.t[segs[1:(end-1)]]   # time of each step start (or end? Python uses segs[:-1])
        lines!(ax, gt_times, gt_lengths;
            color=:black, linestyle=:dash, linewidth=1, label="Ground truth")
    end

    # Plot each estimated trajectory
    for (i, traj) in enumerate(trajs)
        est_lengths = step_lengths(traj, segs)
        est_times = traj.t[segs[1:(end-1)]]
        label = hasproperty(traj, :name) && !isnothing(traj.name) ? traj.name : "Trajectory $(i+1)"
        lines!(ax, est_times, est_lengths; linewidth=1, label=label)
    end

    axislegend(ax; position=:rt)   # legend at right top
    return fig
end

