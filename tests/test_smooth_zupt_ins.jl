include("../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_dir = "data/angermann_high_precision"
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)

simdata = HybridZuptInsJl.InsConfig()
imu_ov, gt_ov = HybridZuptInsJl.truncate_to_overlap(imu, gt)

zupt, traj, step_segs = HybridZuptInsJl.smoothed_zupt_aided_ins(imu_ov, simdata)

fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(traj, gt_ov)
display(fig)

fig = HybridZuptInsJl.plot_inertialdata_and_stepsegm(imu_ov, step_segs)
display(fig)