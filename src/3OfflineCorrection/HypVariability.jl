"""
    evaluate_hyperparameter_variability(
        ins_traj_aligned::Trajectory,
        gt_traj_aligned::Trajectory,
        segs::Vector{Int},
        hyperparams_vec::Vector{SeHyperparams};
        ref_frame::ReferenceFrame = BODY
    )

For each of the hyperparameter sets (one per fold), fix the GP kernel (no re‑fitting),
compute corrections and apply them, then record the 2‑D horizontal position RMSE.

# Arguments
- `ins_traj_aligned`: INS trajectory already rigidly aligned to ground truth.
- `gt_traj_aligned`: Ground truth trajectory sampled at the same timestamps.
- `segs`: Step‑segment indices into the trajectories.
- `hyperparams_vec`: Vector of `SeHyperparams` (length = number of folds).
  Each `SeHyperparams` contains `pos_1`, `pos_2`, `pos_3`, `yaw` fields,
  each being a 3‑vector [`σ_f`, `length_scale`, `σ_n`] for that output.
- `ref_frame`: Reference frame for step vectors (default: `BODY`).

# Returns
- `rmse_per_fold`: Vector of horizontal RMSE (metres) for each fold.
- `p` : permutation vector of the smallest to largest RMSE
- `corrected_trajs`: List of step‑level corrected trajectories (one per fold).
"""
function evaluate_hyperparameter_variability(
    ins_traj_aligned::Trajectory,
    gt_traj_aligned::Trajectory,
    segs::Vector{Int},
    hyperparams_vec::Vector{SeHyperparams};
    ref_frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
)

    # 1. Training data (same for every fold)
    output, input_feature = compute_training_io(
        ins_traj_aligned, gt_traj_aligned, segs;
        ref_frame=ref_frame, feature_type=feature_type
    )

    n_folds = length(hyperparams_vec)
    rmse_per_fold = Vector{Float64}(undef, n_folds)
    corrected_trajs = Vector{Trajectory}(undef, n_folds)

    # Ground truth trajectory at step indices
    gt_step_traj = gt_traj_aligned[segs]

    # 2. Loop over folds
    for fold_idx in 1:n_folds
        hp = hyperparams_vec[fold_idx]

        # GP predictions (returns CorrectionIO)
        pred, _ = compute_corrections(
            input_feature,
            output,
            compute_gp_corrections,
            hp;
            n_restarts_optimizer=0
        )

        # Apply corrections to obtain step‑level trajectory
        gp_traj = apply_corrections(
            ins_traj_aligned,
            pred,
            segs;
            ref_frame=ref_frame,
        )
        corrected_trajs[fold_idx] = gp_traj

        # 2‑D horizontal RMSE (x, y) against ground truth step trajectory
        rmse_per_fold[fold_idx] = rmse(gp_traj, gt_step_traj)[end]
    end

    # # 3. Outlier detection (IQR)
    # q1, q3 = quantile(rmse_per_fold, [0.25, 0.75])
    # iqr = q3 - q1
    # lower = q1 - 1.5 * iqr
    # upper = q3 + 1.5 * iqr
    # is_outlier = (rmse_per_fold .< lower) .| (rmse_per_fold .> upper)

    # Best fold overall
    p = zeros(Int, n_folds)
    sortperm!(p, rmse_per_fold)

    return rmse_per_fold, p, corrected_trajs
end