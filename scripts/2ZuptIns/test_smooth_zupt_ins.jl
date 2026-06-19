include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]
trial_id = 4
inertial = HybridZuptInsJl.InertialData(data_dir, trial_id)
gt_traj = HybridZuptInsJl.Trajectory(data_dir, trial_id)

sim_config = HybridZuptInsJl.InsConfig()

# Truncate to overlapping time window and align ground truth to IMU timestamps
inertial_trunc, gt_traj_trunc = HybridZuptInsJl.truncate_to_overlap(inertial, gt_traj)
gt_traj_aligned = HybridZuptInsJl.temporal_alignment(gt_traj_trunc, inertial_trunc.t)
fig_gt = HybridZuptInsJl.plot_trajectory_xyz_euler(gt_traj_aligned)

# Compute INS trajectory from inertial data
zupt, ins_traj, segs = HybridZuptInsJl.smoothed_zupt_aided_ins(inertial_trunc, sim_config)
fig_ins = HybridZuptInsJl.plot_trajectory_xyz_euler(ins_traj)
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(ins_traj, nothing)

# Calibration window: first point that exceeds the calibration distance from start.
# Distances are computed from the first position (column 1) to all points,
# then restricted to the step‑end indices `segs`.
pos_start = ins_traj.pos[:, 1:1]      # 3×1 matrix
distances = sqrt.(sum((pos_start .- ins_traj.pos) .^ 2, dims=1))[:]   # length N vector
dist_at_segs = distances[segs]        # distances only at step ends
idx_in_segs = findfirst(>(1.55), dist_at_segs)
if isnothing(idx_in_segs)
    error("No step end exceeds the calibration distance of $(sim_config.calibration_distance_m) m")
end
b = segs[idx_in_segs]   # index in the original array

# Calibration indices: all indices up to `b` that are **not** ZUPT frames
calib_idxs = [i for i in 1:b if !zupt[i]]

@info "Pecentage of track used for trajectory alignement : $(round(calib_idxs[end]/length(zupt);digits=3))"
calib_idxs[end] / length(zupt) > 0.3 && @warn "More than 0.3 was used for alignement"


# Rigidly align position and orientation to ground truth
ins_traj_aligned, _, _ = HybridZuptInsJl.transform_position(ins_traj, gt_traj_aligned, calib_idxs)
ins_traj_aligned, _ = HybridZuptInsJl.transform_orientation(ins_traj_aligned, gt_traj_aligned,
    zupt, zeros(3), calib_idxs)

fig_ins_align = HybridZuptInsJl.plot_trajectory_xyz_euler(ins_traj_aligned)


##
fig = HybridZuptInsJl.plot_inertialdata_and_stepsegm(inertial_trunc, segs; zupt=zupt)
fig = HybridZuptInsJl.plot_position_distance_error(ins_traj_aligned, gt_traj_aligned)
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(ins_traj_aligned, gt_traj_aligned)
fig_calib = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(ins_traj_aligned[1:(calib_idxs[end]+500)], gt_traj_aligned[1:calib_idxs[end]])
fig_rmse = HybridZuptInsJl.plot_position_rmse(ins_traj_aligned, gt_traj_aligned)
fig_orientation = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(ins_traj_aligned, gt_traj_aligned)
fig_steps = HybridZuptInsJl.plot_step_lengths(ins_traj_aligned, gt_traj_aligned, segs)