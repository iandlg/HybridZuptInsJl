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
# what makes figure 2 a paired comparison. WAS: add_gaussian_noise was called
# without an rng and fell back to a fixed Xoshiro(123), so every trial and every
# estimator saw one identical draw.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames

# 1. Define datasets / trials to process
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    "Angermann" => (data_dir("ANG2"), trial_ids("ANG2")),
    # "TuDCSC" => (data_dir("DCSC"), trial_ids("DCSC")),
)

# 2. Align INS / GT trajectories for every trial
aligned = HybridZuptInsJl.collect_aligned_trajectories(data_dict)

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



##
# WAS: called without save_path, so this script produced no artifact at all and
# out/Results/5_NoiseRobustness/ sat empty while the thesis cited noise figures
# pasted in as PNGs from a REPL session.
const SECTION = "5_NoiseRobustness"
const DATASET = "Angermann"
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
