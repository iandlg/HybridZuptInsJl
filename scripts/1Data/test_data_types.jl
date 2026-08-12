# scripts/main.jl
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using Printf, LinearAlgebra
using GLMakie, Statistics


data_key = "DCSC"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "MTI" => "data/mti-100-recordings",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]
trial_id = 16
imu = HybridZuptInsJl.InertialData(data_dir, trial_id)
gt = HybridZuptInsJl.Trajectory(data_dir, trial_id)
# gt = HybridZuptInsJl.Trajectory(gt.t, gt.pos[[1, 3, 2], :], gt.R_nb)

ω = HybridZuptInsJl.angular_velocity_from_rotations(gt)
fig = HybridZuptInsJl.plot_inertial_data(gt.t[2:(end-1)], ω)
fig = HybridZuptInsJl.plot_inertial_data(imu)
fig = HybridZuptInsJl.plot_trajectory_xyz_euler(gt)
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(gt, nothing)
fig3d = HybridZuptInsJl.plot_trajectory_3d(gt)
fig = HybridZuptInsJl.plot_trajectory(gt)
##
gt1 = HybridZuptInsJl.Trajectory(data_dir, 19)
gt2 = HybridZuptInsJl.Trajectory(data_dir, 20)
display(GLMakie.Screen(), HybridZuptInsJl.plot_trajectory(gt1; title="T1"))
display(GLMakie.Screen(), HybridZuptInsJl.plot_trajectory(gt2; title="T2"))

##

imu_sync, gt_sync, lag = HybridZuptInsJl.synchronize(imu, gt; fs_resample=200.0)
fig = HybridZuptInsJl.plot_inertial_data(imu_sync)
fig = HybridZuptInsJl.plot_trajectory_xyz_euler(gt_sync)

# find rotation matrix
gt_aligned = HybridZuptInsJl.temporal_alignment(gt_sync, imu_sync.t)
ω_opti = HybridZuptInsJl.angular_velocity_from_rotations(gt_aligned)
ω_imu = imu.u[4:6, 2:(end-1)]
R = HybridZuptInsJl.fit_constant_rotation(ω_imu, ω_opti)
residuals = ω_opti .- R * ω_imu
rms_error = sqrt(mean(sum(residuals .^ 2, dims=1)))

display(GLMakie.Screen(), HybridZuptInsJl.plot_inertial_data(gt_aligned.t[2:(end-1)], ω_opti, R * ω_imu))
display(GLMakie.Screen(), HybridZuptInsJl.plot_inertial_data(imu_sync.t[2:(end-1)], R * ω_imu))
mats = similar(gt_aligned.R_nb)
for i in axes(mats, 3)
    mats[:, :, i] = gt_aligned.R_nb[:, :, i] * R
end
gt_aligned = HybridZuptInsJl.Trajectory(gt_aligned.t, gt_aligned.pos, mats)
fig = HybridZuptInsJl.plot_trajectory(gt_aligned)
