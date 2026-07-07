# scripts/main.jl
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using Printf, LinearAlgebra
using GLMakie, Statistics


data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "MTI" => "data/mti-100-recordings",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]
trial_id = 14
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
# Compare the norm of ground-truth angular velocity with the IMU gyro norm
# ω_norm_gt = vec(norm.(eachcol(ω)))
# imu_ω_norm = vec(norm.(eachcol(imu.u[4:6, :])))
# fig_norm = Figure(resolution=(1000, 500))
# ax_norm = Axis(fig_norm[1, 1];
#     xlabel="Time (s)",
#     ylabel="Angular rate norm (rad/s)",
#     title="Ground-truth ω norm vs IMU gyro norm",
#     titlesize=20)
# lines!(ax_norm, gt.t[2:(end-1)], ω_norm_gt, label="GT ω norm", color=:blue)
# lines!(ax_norm, imu.t, imu_ω_norm, label="IMU gyro norm", color=:red)
# axislegend(ax_norm)
# display(fig_norm)

# imu_ov, gt_ov = HybridZuptInsJl.truncate_to_overlap(imu, gt)
# fig = HybridZuptInsJl.plot_inertial_data(imu_ov)
# fig = HybridZuptInsJl.plot_trajectory_xyz_euler(gt_ov)



# println("IMU:  $(length(imu)) → $(length(imu_ov)) samples")
# println("GT:   $(length(gt))  → $(length(gt_ov))  samples")
# @printf("Overlap: %.3fs - %.3fs\n", imu_ov.t[begin], imu_ov.t[end])

# display(imu.u[:, end])
# display(gt.R_nb[:, :, end])
# display(gt.pos[:, 3])

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

fig = HybridZuptInsJl.plot_inertial_data(gt_aligned.t[2:(end-1)], ω_opti, R * ω_imu)
fig = HybridZuptInsJl.plot_inertial_data(imu_sync.t[2:(end-1)], R * ω_imu)

mats = similar(gt_aligned.R_nb)
for i in axes(mats, 3)
    mats[:, :, i] = gt_aligned.R_nb[:, :, i] * R
end
gt_aligned = HybridZuptInsJl.Trajectory(gt_aligned.t, gt_aligned.pos, mats)
fig = HybridZuptInsJl.plot_trajectory(gt_aligned)
