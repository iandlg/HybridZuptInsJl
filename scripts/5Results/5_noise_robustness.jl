# Section 5 (noise robustness): how much ground-truth noise can the correction
# tolerate?
#
# CAVEAT, and it is a big one: add_gaussian_noise defaults to a fixed
# Xoshiro(123), and nothing here overrides it. Every trial and every estimator
# therefore sees the SAME noise realisation, so the spread in the boxplot is
# trial-to-trial variability, not noise variability. Read the boxes as "how do
# different walks respond to this one noise draw", not as an error bar on the
# noise effect. A real Monte-Carlo needs a seed axis threaded through
# NoiseSpec / run_online_correction_sweep -- add_gaussian_noise already accepts
# an `rng` argument, so no new machinery is required, only a loop.

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
hsgp_p_key = 44
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Define correction methods to compare
estimators = OrderedDict(
    "Base (no correction)" => HybridZuptInsJl.BaseEstimator,
    # "Joint static bias" => HybridZuptInsJl.JointStaticEstimator,
    "Decoupled Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
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
results_df = HybridZuptInsJl.run_online_correction_sweep(
    aligned,
    FRAME,
    FEATURE_TYPE,
    hsgp_p,
    train_ratios,
    estimators,
    output_channels;
    noise_specs=noise_specs
)

##
# WAS: called without save_path, so this script produced no artifact at all and
# out/Results/5_NoiseRobustness/ sat empty while the thesis cited noise figures
# pasted in as PNGs from a REPL session.
const SECTION = "5_NoiseRobustness"

for metric in (:rmse_rate, :rmse_yaw)
    results_figure() do
        HybridZuptInsJl.plot_noise_sweep_boxplots(
            results_df, "Angermann";
            metric=metric,
            show_outliers=false,
            save_path=stamped(SECTION, "noise_sweep_$(metric)"),
        )
    end
end