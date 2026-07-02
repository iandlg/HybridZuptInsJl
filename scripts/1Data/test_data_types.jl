# scripts/main.jl
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using Printf

data_key = "DCSC"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "MTI" => "data/mti-100-recordings",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]
trial_id = 7
imu = HybridZuptInsJl.InertialData(data_dir, trial_id)
gt = HybridZuptInsJl.Trajectory(data_dir, trial_id)

fig = HybridZuptInsJl.plot_inertial_data(imu)
fig = HybridZuptInsJl.plot_trajectory_xyz_euler(gt)
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(gt, nothing)
fig3d = HybridZuptInsJl.plot_trajectory_3d(gt)

imu_ov, gt_ov = HybridZuptInsJl.truncate_to_overlap(imu, gt)
fig = HybridZuptInsJl.plot_inertial_data(imu_ov)
fig = HybridZuptInsJl.plot_trajectory_xyz_euler(gt_ov)

println("IMU:  $(length(imu)) → $(length(imu_ov)) samples")
println("GT:   $(length(gt))  → $(length(gt_ov))  samples")
@printf("Overlap: %.3fs - %.3fs\n", imu_ov.t[begin], imu_ov.t[end])

display(imu.u[:, end])
display(gt.R_nb[:, :, end])
display(gt.pos[:, 3])