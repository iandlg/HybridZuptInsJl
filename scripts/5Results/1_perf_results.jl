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

const SECTION = "1_Performance"

# Set this to the file name of a scores CSV under out/Results/1_Performance/ to re-plot a
# finished sweep instead of recomputing it, e.g.
results_csv = "results_ANG2_HEADING_TWOD_STEP_YAW_2026-09-04T16:54:43.420.csv"
# `nothing` runs the sweep and writes a fresh CSV.
# results_csv = nothing

# 2. Align INS / GT trajectories once per trial.
# Skipped when re-plotting from CSV: this and the sweep are the whole cost of the
# script, and nothing downstream of the scores table needs the trajectories.
aligned = isnothing(results_csv) ? HybridZuptInsJl.collect_aligned_trajectories(data_dict) : nothing

## 3. Hyperparameters
hsgp_p_key = 42
m = 200   # WAS 300 here and 200 in every other script, so the headline figure
# was not directly comparable with the rest of the chapter.

hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Correction methods to compare
estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
)

output_channels = [:pos_1, :pos_2, :yaw]
train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,

## 5/6. Run the sweep and save the scores, or read a finished run back
# Only the scalar columns are written: the sweep also carries the raw zupt/step_seg/
# corr_traj/io_data/model objects, which have no CSV representation -- so a re-read frame
# has the score columns and nothing else, which is all the plots below use.
# dataset_order/noise_spec_tag/noise_spec_order/seed are here because they are the keys
# `paired_estimator_contrast` pairs on: without them a re-read CSV plots the boxplots but
# throws in the paired cells below.
score_cols = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :noise_spec_tag, :noise_spec_order, :seed,
    :rmse, :rmse_rate, :rmse_yaw]

if isnothing(results_csv)
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
    csv_path = stamped(SECTION, "results_$(data_key)_$(FRAME)_$(FEATURE_TYPE)"; ext="csv")
    CSV.write(csv_path, results_df[:, score_cols])
    @info "Saved results table: $csv_path"
else
    csv_path = results_path(SECTION, results_csv)
    results_df = CSV.read(csv_path, DataFrame)
    # CSVs written before the pairing keys joined `score_cols` hold the scores only. This
    # script runs one dataset, no noise and one seed, so each missing key has exactly one
    # value across the whole file -- filling it in restores it rather than inventing it.
    for (col, val) in (:dataset_order => 1, :noise_spec_tag => "NoiseSpec",
        :noise_spec_order => 1, :seed => 123)
        hasproperty(results_df, col) || (results_df[!, col] .= val)
    end
    @info "Loaded results table: $csv_path" nrow(results_df)
end

## 7. Plot
# WAS: this cell re-read a hard-coded CSV path from June while stamping the
# output filenames with the data_key/FRAME/FEATURE of whatever the compute cell
# above had set -- so the figure legend could describe a different run than the
# data plotted. It now plots whatever `results_df` holds: the sweep just computed,
# or the CSV named by `results_csv` at the top of the script.
#
# Corrector order comes from the frame rather than from `estimators`, so a re-read CSV
# keeps the order (and hence the colours) of the run that produced it, whether or not it
# swept the same estimators as the cell above.
corrector_names = unique(sort(results_df, :estimator_order).estimator)

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


## Paired view across the whole train_ratio sweep.
# The cell above pins one operating point; this is the same paired contrast at
# every ratio, in the layout 5_noise_robustness.jl uses for its noise specs
# (train_ratio on the x axis, one box per estimator per group). Each point is a
# per-trial difference against "ZUPT only" on the SAME trial at the SAME ratio,
# so walk-to-walk difficulty cancels and the trend across ratios is readable --
# which it is not in the unpaired boxplots of step 7, where the trial-to-trial
# spread dominates.
const BASE_ESTIMATOR = "ZUPT only"
# From the frame, not `data_dict`, for the same reason as `corrector_names`.
const DATASET = first(unique(results_df.dataset_name))

for metric in (:rmse, :rmse_rate, :rmse_yaw)
    paired = HybridZuptInsJl.paired_estimator_contrast(
        results_df; metric=metric, reference_estimator=BASE_ESTIMATOR, train_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
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
