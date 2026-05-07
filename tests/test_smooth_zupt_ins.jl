include("../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_dir = "data/angermann_high_precision"
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)

simdata = HybridZuptInsJl.InsConfig()
imu_ov, gt_ov = HybridZuptInsJl.truncate_to_overlap(imu, gt)

HybridZuptInsJl.smoothed_zupt_aided_ins(imu_ov, simdata)
