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
# of paying for it again. The figures are then named after that CSV's stem, so a
# re-plot stays traceable to the sweep it came from.

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
    data_key => (data_dir(data_key), trial_ids(data_key)),
)

# noise_results_ANG2_HEADING_TWOD_STEP_YAW_5draws_2026-09-12T12:31:55.563.csv
# noise_results_DCSC_HEADING_TWOD_STEP_YAW_5draws_2026-09-11T16:55:33.086.csv

results_csv = nothing

# 2. Align INS / GT trajectories for every trial
# Skipped when re-plotting from CSV: this and the sweep are the whole cost of the
# script, and nothing downstream of the scores table needs the trajectories.
aligned = isnothing(results_csv) ? HybridZuptInsJl.collect_aligned_trajectories(data_dict) : nothing

## 3. Load HSGP hyperparameters / Input feature type
m = 200
hsgp_p_key = 47
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Define correction methods to compare
# Correction filter (see CORRECTION_FILTERS in _common.jl). Its tag goes into
# every output file name, and picks the correctors below (CORRECTORS).
filter_tag = "V4"

# The baseline runs through the SAME filter as the corrections: the sweep hands
# `correction_filter` every estimator in this dict, `BaseEstimator` included, so
# "ZUPT only" here is that filter's own uncorrected run -- same corrector start
# (V4 starts it on the mocap pose at k=1), same fix path, same scoring. A
# baseline from a different filter is not the thing the corrections are adding to.
estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    # "Joint static bias" => HybridZuptInsJl.JointStaticEstimator,
    "Static" => CORRECTORS[filter_tag].static,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
    # "Joint HSGP" => JointHsgpEstimator,
)

# Does every filter get told how noisy the mocap it is handed actually is?
#
# `false` is what the earlier runs did: noise goes into the ground truth, while
# `sigma_groundtruth` stays at the clean 1cm/0.001rad, so at pos_std=1.0 every
# filter is handed an R ~100x too tight. That measures tolerance of a
# mis-specified R -- which V4 wins by re-estimating it online (notes/016) -- and
# it drags the "ZUPT only" baseline down with it (0.25m clean -> 1.05m).
# `true` gives every estimator the true noise, so what is left to measure is the
# correction model. V4's online jitter estimate floors at this value, so it can
# still inflate above the truth, just not profit from discovering it.
match_gt_sigma = true
sigma_tag = match_gt_sigma ? "matchedR" : "assumedR"

# The baseline is only a baseline if it ran through the same filter; assert it
# rather than trusting the dict above to have been edited in step with the tag.
@assert CORRECTION_FILTERS[filter_tag] === HybridZuptInsJl.hybrid_zupt_aided_insv4
@assert estimators["Static"] === CORRECTORS[filter_tag].static
@assert estimators["HSGP"] === CORRECTORS[filter_tag].hsgp

output_channels = [:pos_1, :pos_2, :yaw]

train_ratios = [0.5]
noise_specs = [
    # Clean reference, so the figure carries its own no-noise baseline instead
    # of requiring the reader to compare against a different figure.
    HybridZuptInsJl.NoiseSpec(; pos_std=0.0, att_std=0.0, tag="No noise"),
    # HybridZuptInsJl.NoiseSpec(; pos_std=0.05, att_std=0.0, tag="Position Noise Only (0.05m)"),
    # HybridZuptInsJl.NoiseSpec(; pos_std=0.1, att_std=0.0, tag="Position Noise Only (0.1m)"),
    # HybridZuptInsJl.NoiseSpec(; pos_std=0.0, att_std=5*pi/180, tag="Heading Noise Only (5°)"),
    # HybridZuptInsJl.NoiseSpec(; pos_std=0.0, att_std=10*pi/180, tag="Heading Noise Only (10°)"),
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
N_NOISE_DRAWS = 2
SEEDS = collect(1:N_NOISE_DRAWS)

# Only the scalar columns are written: the sweep also carries the raw zupt/step_seg/
# corr_traj/io_data/model objects and the pos/att std/bias columns, which have no CSV
# representation -- so a re-read frame has the score columns and nothing else, which is
# all the plots below use. noise_spec_tag/noise_spec_order/seed are here because they
# are the keys `paired_estimator_contrast` pairs on: without them a re-read CSV plots
# the boxplots but throws in the paired cells below.
score_cols = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :noise_spec_tag, :noise_spec_order, :seed,
    # The R each row ran under, so a CSV says on its face whether it is a
    # matched-R sweep -- the noise columns alone cannot tell you that.
    :gt_sigma_pos, :gt_sigma_yaw,
    :rmse, :rmse_rate, :rmse_yaw]

# Stem shared by the scores table and every figure drawn from it. On a re-plot it
# is recovered from the CSV's own name, so the figures carry the identifier of the
# sweep they came from instead of a fresh timestamp that says nothing about it.
const CSV_PREFIX = "noise_results"

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
        correction_filter=CORRECTION_FILTERS[filter_tag],
        match_gt_sigma=match_gt_sigma,
    )

    # Dead-reckoning reference: the same corrector with no mocap fix at all. It
    # bounds the comparison from the other side -- once the mocap is noisy enough,
    # the question is not which correction is best but whether using the fixes
    # beats ignoring them.
    #
    # Run ONCE per trial: with the fixes off, the noise never enters the filter,
    # so every (spec, seed) would reproduce this run exactly. The rows are then
    # copied onto each (spec, seed) key so `paired_estimator_contrast` can pair
    # them inside every cell. Those copies are one measurement repeated, not
    # repeated measurements: its box has no spread beyond the trial-to-trial one.
    nomocap_df = HybridZuptInsJl.run_online_correction_sweep(
        aligned,
        FRAME,
        FEATURE_TYPE,
        hsgp_p,
        train_ratios,
        OrderedDict("ZUPT only (no mocap)" => HybridZuptInsJl.BaseEstimator),
        output_channels;
        noise_specs=[first(noise_specs)],
        seeds=SEEDS[1:1],
        keep_artifacts=false,
        correction_filter=CORRECTION_FILTERS[filter_tag],
        match_gt_sigma=match_gt_sigma,
        posyaw_measurement_update=false,
    )
    nomocap_df.estimator_order .= length(estimators) + 1
    results_df = vcat(results_df, [
        let d = copy(nomocap_df)
            d.noise_spec_tag .= spec.tag
            d.noise_spec_order .= order
            d.seed .= seed
            d
        end
        for (order, spec) in enumerate(noise_specs)
        for seed in (HybridZuptInsJl.is_noiseless(spec) ? SEEDS[1:1] : SEEDS)
    ]...)

    # The draw count is in the stem because a 1-draw and a 10-draw file are different
    # artifacts: at one seed the spread is trial-to-trial only, and that is precisely
    # the distinction this script exists to make.
    # The hyperparameter key is part of the identity of a run, not a detail: the
    # same dataset under key 47 and key 42 gives a different answer on the yaw
    # channel (47's yaw prior underflows, 014), and without the key in the name
    # the two files differ only by timestamp.
    run_stem = "$(filter_tag)_$(sigma_tag)_$(data_key)_key$(hsgp_p_key)_$(FRAME)_$(FEATURE_TYPE)_$(N_NOISE_DRAWS)draws_$(Dates.now())"
    csv_path = results_path(DATA_SECTION, "$(CSV_PREFIX)_$(run_stem).csv")
    CSV.write(csv_path, results_df[:, score_cols])
    @info "Saved results table: $csv_path" nrow(results_df)
else
    csv_path = results_path(DATA_SECTION, results_csv)
    results_df = CSV.read(csv_path, DataFrame)
    run_stem = chopprefix(file_stem(csv_path), "$(CSV_PREFIX)_")
    @info "Loaded results table: $csv_path" nrow(results_df)
end

##
const DATASET = results_df[1, :dataset_name]
const BASE_ESTIMATOR = "ZUPT only"

# Which of the swept noise specs reach the figure. Indexed into `noise_specs` so
# the tags cannot drift from the specs that were actually run;
# `eachindex(noise_specs)` is all of them.
spec_indexes = eachindex(noise_specs) # [1, 6, 7, 8] #
plot_specs = noise_specs[spec_indexes]

for metric in (:rmse,)
    # paired relative change vs Base on the same trial.
    paired = HybridZuptInsJl.paired_estimator_contrast(
        results_df; metric=metric, reference_estimator=BASE_ESTIMATOR,
        noise_spec_tags=[spec.tag for spec in plot_specs])
    results_figure() do
        HybridZuptInsJl.plot_noise_paired_relative_change(
            paired, DATASET;
            metric=metric,
            show_outliers=true,
            _ylims=(-99.0, 300.0),
            figsize=(550, 500),
            save_path=results_path(SECTION, "noise_paired_$(metric)_$(run_stem).pdf"),
        )
    end
end
