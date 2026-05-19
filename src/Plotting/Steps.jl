"""
    plot_step_lengths(trajs::Union{Vector{Trajectory}, Trajectory},
                      gt_traj::Trajectory, segs::Vector{Int})

Plot the 3D step lengths between consecutive step‑end indices for one or more estimated
trajectories and the ground truth.

# Arguments
- `trajs`: Either a single `Trajectory` (wrapped into a 1‑element vector) or a vector
  of `Trajectory` objects. Each trajectory may have a `name` field used in the legend.
- `gt_traj`: Ground truth `Trajectory`.
- `segs`: Vector of indices (1‑based) where each step ends. Step lengths are computed
  between `segs[i]` and `segs[i+1]`.

# Returns
- A `Figure` object with a single axis showing step lengths over time.
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
        xlabel="Time (s)",
        ylabel="Step length (m)",
        xgridvisible=true)

    # Plot ground truth
    if !isnothing(gt_traj)
        # Compute ground‑truth step lengths (dashed black line)
        gt_lengths = step_lengths(gt_traj, segs)

        gt_times = gt_traj.t[segs[1:end-1]]   # time of each step start (or end? Python uses segs[:-1])
        lines!(ax, gt_times, gt_lengths;
            color=:black, linestyle=:dash, linewidth=1, label="Ground truth")
    end

    # Plot each estimated trajectory
    for (i, traj) in enumerate(trajs)
        est_lengths = step_lengths(traj, segs)
        est_times = traj.t[segs[1:end-1]]
        label = hasproperty(traj, :name) && !isnothing(traj.name) ? traj.name : "Trajectory $(i+1)"
        lines!(ax, est_times, est_lengths; linewidth=1, label=label)
    end

    axislegend(ax; position=:rt)   # legend at right top
    return fig
end

function plot_inertialdata_and_stepsegm(
    inertial::InertialData,
    segs::Vector{Int};
    zupt::Union{Nothing,BitVector}=nothing
)
    # Compute squared sum of the first three channels
    y = vec(sum(inertial.u[1:3, :] .^ 2, dims=1))   # shape: (n_samples,)

    # Create figure and axis
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Time", ylabel="Squared sum", title="Inertial Data")
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Plot the continuous signal
    lines!(ax, inertial.t, y; linewidth=0.5)

    # Overlay scatter markers at the given segment indices
    if !isempty(segs)
        scatter!(ax, inertial.t[segs], fill(100.0, length(segs));
            marker=:x, color=:red, markersize=15)
    end
    if !isnothing(zupt)
        scatter!(ax, inertial.t, zupt .* 50;
            color=:green, markersize=5)
    end

    return fig
end

