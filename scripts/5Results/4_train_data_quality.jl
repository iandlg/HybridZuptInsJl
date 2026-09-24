# Section 4 (training-data quality): does varied movement during training
# produce a better frozen model?
#
# Protocol: train on ONE track with full GT, freeze the model, then run it on
# each test track with only 10% GT. One run per (estimator, train, test) cell.
#
# CAVEAT worth stating in the thesis: the hyperparameters below are trained on
# ANG2 but applied to DCSC data, so a cross-dataset transfer confound sits
# inside what is meant to be a training-data experiment. Use key 43 (DCSC) to
# remove it, or state it explicitly.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics
import CSV

data_key = "DCSC"
data_dir_path = data_dir(data_key)

# estimators = OrderedDict(
#     "DecoupledStatic" => HybridZuptInsJl.JointStaticEstimator,
#     "DecoupledHsgp" => HybridZuptInsJl.DecoupledHsgpEstimator,
# )
# Correction filter (see CORRECTION_FILTERS in _common.jl). Its tag goes into
# every output file name, and picks the correctors below (CORRECTORS).
filter_tag = "V4"
# V4 stride-noise arm (`StrideNoise`): `:process_only` is σ_w = σ_n fixed from the
# hyperparameters, no split and no online estimate. Both the training and the frozen
# test corrector are built with it (`training_data_quality_analysis`).
noise_mode = :process_only

estimators = OrderedDict(
    "Static" => CORRECTORS[filter_tag].static,
    # "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
    # "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)
train_labels = OrderedDict(
    6 => "CW Rectangle Long",
    3 => "CCW Rectangle Long",
    4 => "Figure Eight Long",
    5 => "S Shape Long",
    # 12 => "Mixed"
)
test_labels = OrderedDict(
    1 => "CW Rectangle Short",
    14 => "CCW Rectangle Short",
    2 => "Figure Eight Short",
)
# Choose Parameters file
hsgp_p_key = 42
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

params, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=200)

const SECTION = "4_TrainDataQuality"
# Figure in the section directory, numbers in its data/ subdirectory.
const DATA_SECTION = "$(SECTION)/data"
const RUN_STEM = "$(filter_tag)_$(noise_mode)_key$(hsgp_p_key)_$(data_key)"

# Set this to the file name of a CSV under out/Results/4_TrainDataQuality/data/ to re-plot a
# finished sweep instead of recomputing it, e.g.
# results_csv = "train_data_quality_V4_process_only_key42_DCSC_2026-09-24T12:32:06.567.csv"
# `nothing` runs the sweep and writes a fresh CSV.
results_csv = nothing

## Run the sweep and save the numbers, or read a finished run back
if isnothing(results_csv)
    df = HybridZuptInsJl.training_data_quality_analysis(
        data_dir_path, estimators, train_labels, test_labels, params;
        frame=FRAME, feature_type=FEATURE_TYPE,
        test_tr_ratio=0.0,# no measurements; initialised from rigid alignment
        corrected_channels=output_channels,
        correction_filter=CORRECTION_FILTERS[filter_tag],
        estimator_kwargs=(noise_mode=noise_mode,))
    # The baseline rows carry `nothing` in the train_* columns, which CSV writes as empty cells.
    csv_path = stamped(DATA_SECTION, "train_data_quality_$(RUN_STEM)"; ext="csv")
    CSV.write(csv_path,
        select(df, Not(intersect(names(df), ["corr_traj", "io_data", "model", "zupt", "step_seg"])));
        transform=(col, val) -> something(val, missing))
    @info "Saved results table: $csv_path"
else
    csv_path = results_path(DATA_SECTION, results_csv)
    df = CSV.read(csv_path, DataFrame; stringtype=String)
    # Back to `nothing`: `plot_train_data_quality` finds the baseline rows by `isnothing(train_id)`.
    for col in (:estimator_order, :train_id, :train_order, :train_name)
        df[!, col] = [ismissing(v) ? nothing : v for v in df[!, col]]
    end
    @info "Loaded results table: $csv_path" nrow(df)
end

results_figure() do
    HybridZuptInsJl.plot_train_data_quality(df; metric=:rmse,
        # Named after the CSV it plots, timestamp included.
        save_path=results_path(SECTION, "$(file_stem(csv_path)).pdf"))
end

"""Median % change in `metric` against the ZUPT-only run on the same test track, over
the test tracks, per training track and corrector."""
function print_median_change(df::DataFrame, metric::Symbol)
    base = Dict(r.test_id => r[metric] for r in eachrow(df) if isnothing(r.train_id))
    trained = df[.!isnothing.(df.train_id), :]
    trained.rel_change_pct = [100 * (r[metric] - base[r.test_id]) / abs(base[r.test_id])
                              for r in eachrow(trained)]
    sort!(trained, [:train_order, :estimator_order])
    out = combine(groupby(trained, [:train_name, :estimator]; sort=false),
        :rel_change_pct => median => :median_rel_change_pct, nrow => :n_tests)
    println("\n── $metric: median change against ZUPT only, per training track ──")
    show(stdout, out; allrows=true)
    println()
end

print_median_change(df, :rmse)