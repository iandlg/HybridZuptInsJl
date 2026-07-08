include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_key = "DCSC"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "MT" => "data/mti-100-recordings",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]
trial_id = 8

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)


##
fig_segs = HybridZuptInsJl.plot_inertialdata_and_stepsegm(inertial_updated, segs; zupt=zupt)
fig_3d = HybridZuptInsJl.plot_trajectory_3d(ins_traj_aligned)
fig_steps = HybridZuptInsJl.plot_step_lengths(ins_traj_aligned, nothing, segs)
fig_rmse = HybridZuptInsJl.plot_position_rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])
fig_dist = HybridZuptInsJl.plot_position_distance_error(ins_traj_aligned, gt_traj_aligned)
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(ins_traj_aligned, gt_traj_aligned; stop=2000)
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(ins_traj_aligned, gt_traj_aligned)