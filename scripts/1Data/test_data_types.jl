# scripts/main.jl
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using Printf

data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "MTI" => "data/mti-100-recordings",
    "ANG2" => "data/angermann_v2"
)[data_key]
trial_id = 6
imu = HybridZuptInsJl.InertialData(data_dir, trial_id)
gt = HybridZuptInsJl.Trajectory(data_dir, trial_id)

imu_ov, gt_ov = HybridZuptInsJl.truncate_to_overlap(imu, gt)

println("IMU:  $(length(imu)) → $(length(imu_ov)) samples")
println("GT:   $(length(gt))  → $(length(gt_ov))  samples")
@printf("Overlap: %.3fs - %.3fs\n", imu_ov.t[begin], imu_ov.t[end])

# Same spot-checks as Python __main__
display(imu.u[:, 10000])
display(gt.R_nb[:, :, 10000])
display(gt.pos[:, 10000])         # x

fig = HybridZuptInsJl.plot_inertial_data(imu_ov)
display(fig)   # opens an interactive window