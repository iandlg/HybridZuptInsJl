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
  Each `SeHyperparams` contains `yaw`, `pos_1`, `pos_2`, `pos_3` fields,
  each being a 3‑vector [`σ_f`, `length_scale`, `σ_n`] for that output.
- `ref_frame`: Reference frame for step vectors (default: `BODY`).

# Returns
- `rmse_per_fold`: Vector of horizontal RMSE (metres) for each fold.
- `corrected_trajs`: List of step‑level corrected trajectories (one per fold).
- `best_hyper`: `SeHyperparams` of the fold with lowest RMSE.
- `best_non_outlier_hyper`: `SeHyperparams` of the best fold that is **not** an
  outlier (IQR criterion). If all folds are outliers, the best overall is returned.
"""
function evaluate_hyperparameter_variability(
    ins_traj_aligned::Trajectory,
    gt_traj_aligned::Trajectory,
    segs::Vector{Int},
    hyperparams_vec::Vector{SeHyperparams};
    ref_frame::ReferenceFrame=BODY
)

    # 1. Training data (same for every fold)
    output, input_feature = compute_training_io(
        ins_traj_aligned, gt_traj_aligned, segs; ref_frame=ref_frame
    )

    n_folds = length(hyperparams_vec)
    rmse_per_fold = Vector{Float64}(undef, n_folds)
    corrected_trajs = Vector{Trajectory}(undef, n_folds)

    # Ground truth trajectory at step indices
    gt_step_traj = gt_traj_aligned[segs]

    # 2. Loop over folds
    for fold_idx in 1:n_folds
        hp = hyperparams_vec[fold_idx]

        # GP predictions
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
            pred["yaw"],
            pred["pos"], segs;
            ref_frame=ref_frame
        )
        corrected_trajs[fold_idx] = gp_traj

        # 2‑D horizontal RMSE (x, y) against ground truth step trajectory
        rmse_per_fold[fold_idx] = rmse(gp_traj, gt_step_traj)[end]
    end

    # 3. Outlier detection (IQR)
    q1, q3 = quantile(rmse_per_fold, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    is_outlier = (rmse_per_fold .< lower) .| (rmse_per_fold .> upper)

    # Best fold overall
    best_idx = argmin(rmse_per_fold)
    best_hyper = hyperparams_vec[best_idx]

    # Best non‑outlier fold
    non_outlier_idx = findall(.!is_outlier)
    if isempty(non_outlier_idx)
        best_non_outlier_idx = best_idx
    else
        best_non_outlier_idx = non_outlier_idx[argmin(rmse_per_fold[non_outlier_idx])]
    end
    best_non_outlier_hyper = hyperparams_vec[best_non_outlier_idx]

    return rmse_per_fold, corrected_trajs, best_hyper, best_non_outlier_hyper
end