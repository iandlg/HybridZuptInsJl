include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using JSON

hp_optim_key = 31
hp_optim_path = Dict{Int,String}(
    1 => "out/3OfflineCorrection/1_OptimizationResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T12:10:40.080.json",
    2 => "out/3OfflineCorrection/1_OptimizationResults/ANG15_BODY_THREED_STEP_2026-05-13T13:22:26.181.json",
    4 => "out/3OfflineCorrection/1_OptimizationResults/ANG15_BODY_THREED_STEP_2026-05-15T16:14:27.824.json",
    31 => "out/3OfflineCorrection/1_OptimizationResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-22T10:52:10.193.json"
)[hp_optim_key]

vect_hyp, meta, _ = HybridZuptInsJl.from_json(Vector{HybridZuptInsJl.SeHyperparams}, hp_optim_path)

# Extract run parameters
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[meta["data_key"]]
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])


ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, meta["trial_id"]
)
ins_rmse = HybridZuptInsJl.rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])[end]

rmse_per_fold, perm, corrected_trajs = HybridZuptInsJl.evaluate_hyperparameter_variability(
    ins_traj_aligned, gt_traj_aligned, segs, vect_hyp; ref_frame=FRAME, feature_type=FEATURE_TYPE
)

HybridZuptInsJl.plot_hyperparameter_rmse_variability(
    rmse_per_fold, ins_rmse, meta["rmse"]["model + static"]
)

## Save best non outlier hyperparameter
outdir = "out/3OfflineCorrection/2_VariabilityResults"
time = string(Dates.now())
filename = "$(meta["data_key"])$(meta["trial_id"])_$(meta["ref_frame"])_$(meta["feature_type"])_$(time).json"

perm_idx = 1
HybridZuptInsJl.to_json(joinpath(outdir, filename), vect_hyp[perm[perm_idx]];
    metadata=Dict{String,Any}(
        "data_key" => meta["data_key"],
        "trial_id" => meta["trial_id"],
        "ref_frame" => meta["ref_frame"],
        "feature_type" => meta["feature_type"],
        "n_restarts_optimizer" => meta["n_restarts_optimizer"],
        "log_kern_bounds" => meta["log_kern_bounds"],
        "log_noise_bounds" => meta["log_noise_bounds"],
        "rmse" => rmse_per_fold[perm[perm_idx]],
        "method" => meta["method"],
        "normalize_input" => meta["normalize_input"],
        "normalize_output" => meta["normalize_output"])
)

