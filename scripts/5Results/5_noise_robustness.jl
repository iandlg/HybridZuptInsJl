# Section 5 (noise robustness): how much ground-truth noise can the correction
# tolerate?
#
# Two figures per metric, both with the noise specs on the x axis and each box
# spanning the trials:
#
#   1. Unpaired -- ZUPT only, Static and HSGP side by side, in the metric's own
#      units. Shows the absolute error level at each noise level.
#   2. Paired -- Static and HSGP only, as the per-trial relative change against
#      ZUPT only on the SAME trial and the SAME noise realisation. ZUPT only is
#      the zero line. Walk-to-walk difficulty cancels here, so a box clear of zero is a
#      consistent effect; in figure 1 the same effect can hide inside the spread.
#
# N_NOISE_DRAWS below controls how many noise realisations each (trial, noise
# spec) gets: one per seed in SEEDS. At one seed the spread in both figures is
# purely trial-to-trial; with more, each box also carries the draw-to-draw
# variability, at n_trials x N_NOISE_DRAWS points per box.
#
# Each realisation comes from its own Xoshiro(seed), drawn once per
# (trial, noise spec, seed) and shared by every estimator in that cell -- that is
# what makes figure 2 a paired comparison.
#
# Set `results_csv` below to re-plot a finished sweep from its scores CSV instead
# of paying for it again.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames
import CSV

# Figures go in the section directory, scores tables in its data/ subdirectory.
# results_path mkpaths whatever nested section it is handed, as the sibling
# 5_noise_robustness-more_data.jl already relies on.
const SECTION = "5_NoiseRobustness/NoiseSweep"
const DATA_SECTION = "$(SECTION)/data"

# 1. Define datasets / trials to process
data_key = "DCSC"
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    # "Angermann" => (data_dir(data_key), trial_ids(data_key)),
    "DCSC" => (data_dir(data_key), trial_ids(data_key)),
)

# Set this to the file name of a scores CSV under
# out/Results/5_NoiseRobustness/NoiseSweep/data/ to re-plot a finished sweep instead
# of recomputing it, e.g.
# results_csv = "noise_results_ANG2_HEADING_TWOD_STEP_YAW_5draws_2026-09-11T09:12:33.123.csv"
# `nothing` runs the sweep and writes a fresh CSV.
results_csv = "noise_results_DCSC_HEADING_TWOD_STEP_YAW_5draws_2026-09-11T12:50:51.024.csv"

# 2. Align INS / GT trajectories for every trial
# Skipped when re-plotting from CSV: this and the sweep are the whole cost of the
# script, and nothing downstream of the scores table needs the trajectories.
aligned = isnothing(results_csv) ? HybridZuptInsJl.collect_aligned_trajectories(data_dict) : nothing

## 3. Load HSGP hyperparameters / Input feature type
m = 200
hsgp_p_key = 46
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Define correction methods to compare
estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    # "Joint static bias" => HybridZuptInsJl.JointStaticEstimator,
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    # "Joint HSGP" => JointHsgpEstimator,
)

output_channels = [:pos_1, :pos_2, :yaw]

train_ratios = [0.5]
noise_specs = [
    # Clean reference, so the figure carries its own no-noise baseline instead
    # of requiring the reader to compare against a different figure.
    HybridZuptInsJl.NoiseSpec(; pos_std=0.0, att_std=0.0, tag="No noise"),
    HybridZuptInsJl.NoiseSpec(; pos_std=0.05, att_std=0.0, tag="Position Noise Only (0.05m)"),
    HybridZuptInsJl.NoiseSpec(; pos_std=0.1, att_std=0.0, tag="Position Noise Only (0.1m)"),
    HybridZuptInsJl.NoiseSpec(; pos_std=0.0, att_std=5*pi/180, tag="Heading Noise Only (5°)"),
    HybridZuptInsJl.NoiseSpec(; pos_std=0.0, att_std=10*pi/180, tag="Heading Noise Only (10°)"),
    HybridZuptInsJl.NoiseSpec(; pos_std=0.05, att_std=5*pi/180, tag="Position & Heading Noise (0.05m, ±5°)"),
    HybridZuptInsJl.NoiseSpec(; pos_std=0.1, att_std=10*pi/180, tag="Position & Heading Noise (0.1m, ±10°)"),
    HybridZuptInsJl.NoiseSpec(; pos_std=1.0, att_std=10*pi/180, tag="Position & Heading Noise (1.0m, ±10°)"),
]
## 5. Run the sweep
# Each draw is shared by all estimators in its cell, so the paired figure below
# compares like with like. keep_artifacts=false because the raw trajectory/model
# objects cost ~2.5 MB per row and nothing here reads them.
#
# Cost is trials x (1 + n_noisy_specs x N_NOISE_DRAWS) x estimators runs at
# ~1.2 s each: 231 runs (~5 min) at 1 draw, ~2000 (~40 min) at 10.
N_NOISE_DRAWS = 5
SEEDS = collect(1:N_NOISE_DRAWS)

# Only the scalar columns are written: the sweep also carries the raw zupt/step_seg/
# corr_traj/io_data/model objects and the pos/att std/bias columns, which have no CSV
# representation -- so a re-read frame has the score columns and nothing else, which is
# all the plots below use. noise_spec_tag/noise_spec_order/seed are here because they
# are the keys `paired_estimator_contrast` pairs on: without them a re-read CSV plots
# the boxplots but throws in the paired cells below.
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
        noise_specs=noise_specs,
        seeds=SEEDS,
        keep_artifacts=false,
    )
    # The draw count is in the name because a 1-draw and a 10-draw file are different
    # artifacts: at one seed the spread is trial-to-trial only, and that is precisely
    # the distinction this script exists to make.
    csv_path = stamped(DATA_SECTION,
        "noise_results_$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(N_NOISE_DRAWS)draws"; ext="csv")
    CSV.write(csv_path, results_df[:, score_cols])
    @info "Saved results table: $csv_path" nrow(results_df)
else
    csv_path = results_path(DATA_SECTION, results_csv)
    results_df = CSV.read(csv_path, DataFrame)
    @info "Loaded results table: $csv_path" nrow(results_df)
end

##
const DATASET = data_key
const BASE_ESTIMATOR = "ZUPT only"

for metric in (:rmse, :rmse_yaw)
    # (1) Absolute level: every estimator, per noise spec, boxed over trials.
    results_figure() do
        HybridZuptInsJl.plot_noise_sweep_boxplots(
            results_df, DATASET;
            metric=metric,
            show_outliers=false,
            save_path=stamped(SECTION, "noise_sweep_$(metric)"),
        )
    end

    # (2) Same layout, paired: relative change vs Base on the same trial.
    paired = HybridZuptInsJl.paired_estimator_contrast(
        results_df; metric=metric, reference_estimator=BASE_ESTIMATOR)
    results_figure() do
        HybridZuptInsJl.plot_noise_paired_relative_change(
            paired, DATASET;
            metric=metric,
            reference_label=BASE_ESTIMATOR,
            show_outliers=false,
            save_path=stamped(SECTION, "noise_paired_$(metric)"),
        )
    end
end
