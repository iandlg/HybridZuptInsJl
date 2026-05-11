include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using JSON

data_dir = "data/angermann_high_precision"
trial_id = 15
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(
    (gt.t .+ -0.05),
    gt.pos,
    gt.R_nb,
    gt.vel
)

vect_hyp = HybridZuptInsJl.read_hyperparameters_json("./out/hyperparameters/fixed_jl_hypers_body_3d_list.json")

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; inertial=imu, gt_traj=gt
)
ins_rmse = HybridZuptInsJl.rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])[end]

rmse_per_fold, corrected_trajs, best_hyper, best_non_outlier_hyper = HybridZuptInsJl.evaluate_hyperparameter_variability(
    ins_traj_aligned, gt_traj_aligned, segs, vect_hyp; ref_frame=HybridZuptInsJl.BODY
)

HybridZuptInsJl.plot_hyperparameter_rmse_variability(
    rmse_per_fold, ins_rmse, 0.682
)

## Save best non outlier hyperparameter
HybridZuptInsJl.to_json("out/3OfflineCorrection/VariabilityResults/nn_outlier.json", best_non_outlier_hyper)