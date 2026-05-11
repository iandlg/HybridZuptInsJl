include("../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_dir = "data/angermann_high_precision"
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)

Δt = -0.05
gt = HybridZuptInsJl.Trajectory(
    (gt.t .+ Δt),
    gt.pos,
    gt.R_nb,
    gt.vel
)

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, 15; inertial=imu, gt_traj=gt
)
##
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(ins_traj_aligned[1:1000], gt_traj_aligned[1:1000])
fig_rmse = HybridZuptInsJl.plot_position_rmse(ins_traj_aligned, gt_traj_aligned)
fig_orientation = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(ins_traj_aligned, gt_traj_aligned)
fig_steps = HybridZuptInsJl.plot_step_lengths(ins_traj_aligned, gt_traj_aligned, segs)