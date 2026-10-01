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
using OrderedCollections, DataFrames
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
## Run sweep
df = HybridZuptInsJl.training_data_quality_analysis(
    data_dir_path, estimators, train_labels, test_labels, params;
    frame=FRAME, feature_type=FEATURE_TYPE,
    corrected_channels=output_channels,
    correction_filter=CORRECTION_FILTERS[filter_tag],
    estimator_kwargs=(noise_mode=noise_mode,))
##
const SECTION = "4_TrainDataQuality"
# Figure in the section directory, numbers in its data/ subdirectory.
const DATA_SECTION = "$(SECTION)/data"
const RUN_STEM = "$(filter_tag)_$(noise_mode)_key$(hsgp_p_key)_$(data_key)"

results_figure() do
    HybridZuptInsJl.plot_train_data_quality(df; metric=:rmse,
        save_path=stamped(SECTION, "train_data_quality_$(RUN_STEM)"))
end

## Persist the numbers next to the figure.
# WAS: CSV.write("train_test_variability.csv", df) -- no `import CSV` in this
# script (so it only worked if a previous REPL cell had loaded it), and it wrote
# into the repository root rather than out/.
CSV.write(stamped(DATA_SECTION, "train_data_quality_$(RUN_STEM)"; ext="csv"),
    select(df, Not(intersect(names(df), ["corr_traj", "io_data", "model", "zupt", "step_seg"])));
    transform=(col, val) -> something(val, missing))