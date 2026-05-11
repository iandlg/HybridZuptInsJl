include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using JSON

data_dir = "data/angermann_high_precision"
trial_id = 15
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)

vect_hyp = HybridZuptInsJl.read_hyperparameters_json("./out/hyperparameters/jl_hypers_body_3d_list.json")

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)
ins_rmse = HybridZuptInsJl.rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])[end]

rmse_per_fold, corrected_trajs, best_hyper, best_non_outlier_hyper = HybridZuptInsJl.evaluate_hyperparameter_variability(
    ins_traj_aligned, gt_traj_aligned, segs, vect_hyp; ref_frame=HybridZuptInsJl.BODY
)

HybridZuptInsJl.plot_hyperparameter_rmse_variability(
    rmse_per_fold, ins_rmse, 0.0
)
