include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections, DataFrames

data_key = "DCSC" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

train_ids = [6, 3, 5] #  , 4, 5
test_ids = [1] # , 14, 2

train_labels = Dict(
    6 => "6_CWRectangle_long",
    3 => "3_CCWRectangle_long",
    4 => "4_VariedSpeedWalk_FigureEight",
    5 => "5_Walk_S_shape",
)
test_labels = Dict(
    1 => "1_Walk_CWRectangle",
    14 => "14_Right_CCWRectangle_long",
    2 => "2_Walk_FigureEight",
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
    data_dir, train_ids, test_ids, params;
    train_labels=train_labels, test_labels=test_labels,
    frame=FRAME, feature_type=FEATURE_TYPE,
    hsgp_estimator_factories=[HybridZuptInsJl.DecoupledHsgpEstimator],
    corrected_channels=output_channels
)
##
fig = HybridZuptInsJl.plot_train_data_quality(df; metric=:rmse_rate,
    save_path="train_test_variability.png")
display(fig)
##
CSV.write("train_test_variability.csv", df)