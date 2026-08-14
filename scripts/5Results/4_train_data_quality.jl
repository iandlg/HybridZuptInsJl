include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames

data_key = "DCSC" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

estimators = OrderedDict(
    "DecoupledStatic" => HybridZuptInsJl.JointStaticEstimator,
    "DecoupledHsgp" => HybridZuptInsJl.DecoupledHsgpEstimator,
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

hsgp_p_path = Dict{Int,String}(
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no ouptut norm; DCSC
)[hsgp_p_key]

output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

# Load parameters with corresponding metatdata
params, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)

# Get feature type and reference frame
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

df = HybridZuptInsJl.training_data_quality_analysis(
    data_dir, estimators, train_labels, test_labels, params;
    frame=FRAME, feature_type=FEATURE_TYPE,
    corrected_channels=output_channels)
##
import CairoMakie, Dates
base_dir = "out/Results/4_TrainDataQuality"

time = string(Dates.now())
filename = "$(time)_train_data_quality.svg"
path = joinpath(base_dir, filename)

CairoMakie.with_theme(CairoMakie.theme_ggplot2()) do
    fig = HybridZuptInsJl.plot_train_data_quality(df; metric=:rmse_rate,
        save_path=path)
end

##
CSV.write("train_test_variability.csv", df)