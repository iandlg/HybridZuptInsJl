# Section 2 (generalisation): do hyperparameters trained on ANG2 transfer to a
# different dataset (different IMU, different user)?
#
# Design: one frozen hyperparameter set, applied to both datasets, 9 + 10 trials,
# train_ratio = 0.5, no injected noise. Varied: dataset x estimator.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames

# 1. Datasets / trials
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    "Angermann" => (data_dir("ANG2"), trial_ids("ANG2")),
    "TuDCSC" => (data_dir("DCSC"), trial_ids("DCSC")),
)

# 2. Align INS / GT trajectories for every trial
aligned = HybridZuptInsJl.collect_aligned_trajectories(data_dict)

## 3. Hyperparameters (trained on ANG2 -- that is the point of this figure)
m = 200
hsgp_p_key = 42
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Correction methods to compare
estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    # "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    # "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)

output_channels = [:pos_1, :pos_2, :yaw]
train_ratios = [0.5]

## 5. Run the sweep
results_df = HybridZuptInsJl.run_online_correction_sweep(
    aligned,
    FRAME,
    FEATURE_TYPE,
    hsgp_p,
    train_ratios,
    estimators,
    output_channels,
)

## 6. Plot
# n per box is small (9 and 10 trials), so the paired view below is the one to
# read for a claim; the boxplot is the distributional summary.
const SECTION = "2_HypSensitivity/DatasetComparison"

results_figure() do
    HybridZuptInsJl.boxplot_dataset_comparison(
        results_df;
        metric=:rmse_rate,
        train_ratio=0.5,
        save_path=stamped(SECTION, "dataset_comparison"),
    )
end

# Same trials go through every estimator, so the design is paired. Plot the
# per-trial difference against the uncorrected baseline rather than reading two
# independent-looking boxes side by side.
results_figure() do
    HybridZuptInsJl.plot_paired_relative_change(
        results_df;
        metric=:rmse,
        baseline="ZUPT only",
        group=:dataset_name,
        save_path=stamped(SECTION, "dataset_comparison_paired"),
    )
end
