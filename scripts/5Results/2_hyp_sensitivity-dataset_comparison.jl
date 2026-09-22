# Section 2 (generalisation): do hyperparameters trained on ANG2 transfer to a
# different dataset (different IMU, different user)?
#
# Design: one frozen hyperparameter set, applied to both datasets, 9 + 10 trials,
# train_ratio = 0.5, no injected noise. Varied: dataset x estimator.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames
import CSV

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
# Correction filter (see CORRECTION_FILTERS in _common.jl). Its tag goes into
# every output file name, and picks the correctors below (CORRECTORS).
filter_tag = "V4"
# V4 stride-noise arm (`StrideNoise`): `:process_only` is σ_w = σ_n fixed from the
# hyperparameters, no split and no online estimate. It goes into every file stem here
# for the same reason `filter_tag` does.
noise_mode = :process_only

estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    "Static" => CORRECTORS[filter_tag].static,
    # "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
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
    output_channels;
    correction_filter=CORRECTION_FILTERS[filter_tag],
    estimator_kwargs=(noise_mode=noise_mode,),
)

## 6. Plot
# n per box is small (9 and 10 trials), so the paired view below is the one to
# read for a claim; the boxplot is the distributional summary.
const SECTION = "2_HypSensitivity/DatasetComparison"
# Figures in the section directory, scores table in its data/ subdirectory.
const DATA_SECTION = "$(SECTION)/data"
const RUN_STEM = "$(filter_tag)_$(noise_mode)_key$(hsgp_p_key)"

score_cols = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :noise_spec_tag, :noise_spec_order, :seed,
    :rmse, :rmse_rate, :rmse_yaw]
CSV.write(stamped(DATA_SECTION, "results_$(RUN_STEM)"; ext="csv"), results_df[:, score_cols])

# Same trials go through every estimator, so the design is paired. Box the
# per-trial difference against the uncorrected baseline rather than reading two
# independent-looking boxes side by side.
paired = HybridZuptInsJl.paired_estimator_contrast(
    results_df; metric=:rmse, reference_estimator="ZUPT only")

results_figure() do
    HybridZuptInsJl.plot_dataset_paired_relative_change(
        paired;
        metric=:rmse,
        show_points=false,
        show_outliers=true,
        save_path=stamped(SECTION, "dataset_comparison_paired_rmse_$(RUN_STEM)"),
    )
end

paired = HybridZuptInsJl.paired_estimator_contrast(
    results_df; metric=:rmse_yaw, reference_estimator="ZUPT only")

results_figure() do
    HybridZuptInsJl.plot_dataset_paired_relative_change(
        paired;
        metric=:rmse_yaw,
        show_points=false,
        show_outliers=true,
        save_path=stamped(SECTION, "dataset_comparison_paired_rmse_yaw_$(RUN_STEM)"),
    )
end