include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using JSON

data_dir = "data/angermann_high_precision"
trial_id = 15

vect_hyp, meta, _ = HybridZuptInsJl.from_json(Vector{HybridZuptInsJl.SeHyperparams},
    "out/3OfflineCorrection/OptimizationResults/ANG15_BODY_THREED_STEP_2026-05-13T13:22:26.181.json")

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)
ins_rmse = HybridZuptInsJl.rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])[end]

rmse_per_fold, corrected_trajs, best_hyper, b_idx, best_non_outlier_hyper, bno_idx = HybridZuptInsJl.evaluate_hyperparameter_variability(
    ins_traj_aligned, gt_traj_aligned, segs, vect_hyp; ref_frame=HybridZuptInsJl.BODY
)

HybridZuptInsJl.plot_hyperparameter_rmse_variability(
    rmse_per_fold, ins_rmse, 0.731
)

## Save best non outlier hyperparameter
outdir = "out/3OfflineCorrection/VariabilityResults"
time = string(Dates.now())
filename = "$(meta["data_key"])$(meta["trial_id"])_$(meta["ref_frame"])_$(meta["feature_type"])_$(time).json"

HybridZuptInsJl.to_json(joinpath(outdir, filename), best_non_outlier_hyper;
    metadata=Dict{String,Any}(
        "data_key" => meta["data_key"],
        "trial_id" => meta["trial_id"],
        "ref_frame" => meta["ref_frame"],
        "feature_type" => meta["feature_type"],
        "n_restarts_optimizer" => meta["n_restarts_optimizer"],
        "log_kern_bounds" => meta["log_kern_bounds"],
        "log_noise_bounds" => meta["log_noise_bounds"],
        "rmse" => rmse_per_fold[bno_idx],
        "method" => meta["method"],
        "normalize_input" => meta["normalize_input"],
        "normalize_output" => meta["normalize_output"])
)

