# THESIS SECTION 1 (headline): performance vs how much ground truth is available
# online (train_ratio), for every corrector, across all trials of a dataset.
#
# Structurally this is the strongest sweep in 5Results: a genuinely swept
# continuous x-axis with per-trial replication at every level.
#
# It now runs through `collect_aligned_trajectories` + `run_online_correction_sweep`,
# the same path as every other 5Results script, instead of the older
# `collect_dataset`/`performance_dataframe` pair. That path took *instances* of the
# estimators and re-aligned each trial once per (corrector, train_ratio) cell, i.e.
# it repeated the INS/GT alignment |correctors| x |train_ratios| times per trial for
# no reason; the sweep aligns each trial once and reuses it. It also scores yaw
# (`rmse_yaw`) and keeps the raw filter outputs in the frame, neither of which the
# old path returned.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames
import CSV

# 1. Dataset / trials.
# Every trial of the dataset, not the curated TRIAL_IDS list: this figure is the
# "across all trials" claim.
data_key = "ANG2"
data_dir_path = data_dir(data_key)
ids = trial_ids(data_key) #  HybridZuptInsJl.list_trial_ids(data_dir_path; foot="R")

data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    "Angerman" => (data_dir_path, ids),
)

# 2. Align INS / GT trajectories once per trial
aligned = HybridZuptInsJl.collect_aligned_trajectories(data_dict)

## 3. Hyperparameters
hsgp_p_key = 42
m = 200   # WAS 300 here and 200 in every other script, so the headline figure
# was not directly comparable with the rest of the chapter.

hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Correction methods to compare
# WAS: "Static" => HybridZuptInsJl.StaticCorrectorV2(300)
# StaticCorrectorV2 does not exist anywhere in src/ -- this cell could not run
# as written, which is why the committed figure dates from 26 June while the
# rest of the chapter is from August. Names also updated from the old
# Default/Static/Split/Slam vocabulary to the one every other script uses, so
# the same estimator is called the same thing in every figure.
estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    # "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)

output_channels = [:pos_1, :pos_2, :yaw]
train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,

## 5. Run the sweep
results_df = HybridZuptInsJl.run_online_correction_sweep(
    aligned,
    FRAME,
    FEATURE_TYPE,
    hsgp_p,
    train_ratios,
    estimators,
    output_channels;
    estimator_alloc=300,
)

## 6. Save the scores
# Only the scalar columns: the sweep also carries the raw zupt/step_seg/corr_traj/
# io_data/model objects, which have no CSV representation.
const SECTION = "1_Performance"

score_cols = [:dataset_name, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :rmse, :rmse_rate, :rmse_yaw]
csv_path = stamped(SECTION, "results_$(data_key)_$(FRAME)_$(FEATURE_TYPE)"; ext="csv")
CSV.write(csv_path, results_df[:, score_cols])
@info "Saved results table: $csv_path"

## 7. Plot
# WAS: this cell re-read a hard-coded CSV path from June while stamping the
# output filenames with the data_key/FRAME/FEATURE of whatever the compute cell
# above had set -- so the figure legend could describe a different run than the
# data plotted. It now plots the DataFrame just computed. To re-plot an older
# run, set `results_df = CSV.read(<path>, DataFrame)` here deliberately.
corrector_names = collect(keys(estimators))

for metric in (:rmse, :rmse_rate), show_outliers in (true, false)
    suffix = show_outliers ? "" : "_nooutliers"
    results_figure() do
        HybridZuptInsJl.plot_corrector_boxplots(
            results_df, metric;
            show_outliers=show_outliers,
            corrector_names=corrector_names,
            save_path=stamped(SECTION, "$(uppercase(string(metric)))$(suffix)"),
        )
    end
end

## Paired view at the operating point used by the rest of the chapter.
# Same trials at every train_ratio, so per-trial differences are the honest
# summary at n ≈ 16.
# results_figure() do
#     HybridZuptInsJl.plot_paired_relative_change(
#         results_df;
#         metric=:rmse,
#         baseline="ZUPT only",
#         train_ratio=0.5,
#         label_trials=true,
#         save_path=stamped(SECTION, "paired_vs_baseline_tr0.5"),
#     )
# end

## Paired view across the whole train_ratio sweep.
# The cell above pins one operating point; this is the same paired contrast at
# every ratio, in the layout 5_noise_robustness.jl uses for its noise specs
# (train_ratio on the x axis, one box per estimator per group). Each point is a
# per-trial difference against "ZUPT only" on the SAME trial at the SAME ratio,
# so walk-to-walk difficulty cancels and the trend across ratios is readable --
# which it is not in the unpaired boxplots of step 7, where the trial-to-trial
# spread dominates.
const BASE_ESTIMATOR = "ZUPT only"
const DATASET = first(keys(data_dict))

for metric in (:rmse, :rmse_rate, :rmse_yaw)
    paired = HybridZuptInsJl.paired_estimator_contrast(
        results_df; metric=metric, reference_estimator=BASE_ESTIMATOR)
    results_figure() do
        HybridZuptInsJl.plot_train_ratio_paired_relative_change(
            paired, DATASET;
            metric=metric,
            reference_label=BASE_ESTIMATOR,
            show_outliers=true,
            show_points=false,
            show_subtitle=false,
            save_path=stamped(SECTION, "train_ratio_paired_$(metric)"),
        )
    end
end
