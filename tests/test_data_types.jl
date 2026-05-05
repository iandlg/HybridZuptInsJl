# scripts/main.jl
include("../src/HybridZuptIns.jl");
using .HybridZuptInsJl;

data_dir = "data/angermann_high_precision"
imu = InertialData(data_dir, 15)
gt = Trajectory(data_dir, 15)

imu_ov, gt_ov = truncate_to_overlap(imu, gt)

println("IMU:  $(length(imu)) → $(length(imu_ov)) samples")
println("GT:   $(length(gt))  → $(length(gt_ov))  samples")
println("Overlap: $(imu_ov.t[begin]:.3f)s – $(imu_ov.t[end]:.3f)s")

# Same spot-checks as Python __main__
println(imu.u[:, 10000])          # Julia is 1-indexed
println(gt.R_nb[:, :, 10000])
println(gt.pos[1, 10000])         # x
println(gt.pos[3, 10000])         # z