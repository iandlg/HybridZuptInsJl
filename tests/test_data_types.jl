# scripts/main.jl
include("../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using Printf

data_dir = "data/angermann_high_precision"
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)

imu_ov, gt_ov = HybridZuptInsJl.truncate_to_overlap(imu, gt)

println("IMU:  $(length(imu)) → $(length(imu_ov)) samples")
println("GT:   $(length(gt))  → $(length(gt_ov))  samples")
@printf("Overlap: %.3fs – %.3fs\n", imu_ov.t[begin], imu_ov.t[end])

# Same spot-checks as Python __main__
display(imu.u[:, 10000])
display(gt.R_nb[:, :, 10000])
display(gt.pos[:, 10000])         # x
