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
estimators = OrderedDict(
    "Decoupled Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)
train_labels = OrderedDict(
    6 => "CWRectangle_long",
    3 => "CCWRectangle_long_A",
    4 => "FigureEight_long",
    5 => "S_shape_long",
)
test_labels = OrderedDict(
    1 => "CWRectangle_short",
    14 => "CCWRectangle_long_B",
    2 => "FigureEight_short",
)
# Choose Parameters file
hsgp_p_key = 42
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

params, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=200)

df = HybridZuptInsJl.training_data_quality_analysis(
    data_dir_path, estimators, train_labels, test_labels, params;
    frame=FRAME, feature_type=FEATURE_TYPE,
    corrected_channels=output_channels)
##
const SECTION = "4_TrainDataQuality"

results_figure() do
    HybridZuptInsJl.plot_train_data_quality(df; metric=:rmse_rate,
        save_path=stamped(SECTION, "train_data_quality"))
end

## Persist the numbers next to the figure.
# WAS: CSV.write("train_test_variability.csv", df) -- no `import CSV` in this
# script (so it only worked if a previous REPL cell had loaded it), and it wrote
# into the repository root rather than out/.
CSV.write(stamped(SECTION, "train_data_quality"; ext="csv"),
    select(df, Not(intersect(names(df), ["corr_traj", "io_data", "model", "zupt", "step_seg"]))))