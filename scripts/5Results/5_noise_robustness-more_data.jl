include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames

data_key = "DCSC" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

# estimators = OrderedDict(
#     "DecoupledStatic" => HybridZuptInsJl.JointStaticEstimator,
#     "DecoupledHsgp" => HybridZuptInsJl.DecoupledHsgpEstimator,
# )
estimators = OrderedDict(
    "Decoupled Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    # "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    # "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)
train_labels = Dict(
    "ANG2" => OrderedDict(
        4 => "Walk_8",
        13 => "Walk_withoutCarpetShape",
        14 => "Walk_Patrick_long",
        1 => "Walk_rectangles",
        2 => "Walk_rectangles_otherdir",
        3 => "Walk_straight",
    ),
    "DCSC" => OrderedDict(
        3 => "CCWRectangle_long_A",
        4 => "FigureEight_long",
        5 => "S_shape_long",
        6 => "CWRectangle_long",
        12 => "Mixed",
        8 => "CW_Rect",
        # 10 => "FigureEightLong"
    )
)[data_key]
test_labels = Dict(
    "ANG2" => OrderedDict(
        15 => "Walk_Patrick_mixed",
    ),
    "DCSC" => OrderedDict(
        1 => "CWRectangle_short",
        14 => "CCWRectangle_long_B",
        2 => "FigureEight_short",
    )
)[data_key]
# Choose Parameters file
hsgp_p_key = 44

hsgp_p_path = Dict{Int,String}(
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no ouptut norm; DCSC
    44 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-19T12:56:03.443.json", # same; ANG2 higer lower noise bound
    45 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-19T16:42:30.821.json", # slightly changed hyps compared to 44
)[hsgp_p_key]

output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

# Load parameters with corresponding metatdata
params, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
@show params
new_hp = HybridZuptInsJl.SeHyperparams(
    [5e-1, 2.0, 0.09],
    [5e-1, 2.0, 0.09],
    [5e-1, 2.0, 0.09],
    [0.146, 24, 2.8]
)
params = HybridZuptInsJl.basecopy(params; new_hp=new_hp)

noise = HybridZuptInsJl.NoiseSpec(; pos_std=0.1, att_std=5*pi/180, tag="Position & Heading Noise (0.1m, ±5°)")

# Get feature type and reference frame
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

df_results = HybridZuptInsJl.multi_track_training_analysis(
    data_dir, estimators, train_labels, test_labels, params;
    frame=FRAME, feature_type=FEATURE_TYPE, corrected_channels=output_channels,
    noise_spec=noise,
    train_tr_ratio=1.0,
    test_tr_ratio=0.1
)

fig = HybridZuptInsJl.plot_multi_track_training_quality(df_results)

##
using Dates
base_dir = "out/4OnlineCorrection/6_HypOpt"
combo_dir = joinpath(base_dir, data_key, string(FRAME) * "-" * string(FEATURE_TYPE))
mkpath(combo_dir)


time = string(Dates.now())
filename = "$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).json"
HybridZuptInsJl.to_json(joinpath(combo_dir, filename), params;
    metadata=Dict(
        "data_key" => data_key,
        # "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        # "normalize_input" => normalization_params["normalize_x"],
        # "normalize_output" => normalization_params["normalize_y"],
        # "train_ids" => train_ids,
        # "outlier_removal" => outlier_removal_params,
        # "normalization" => normalization_params,
        # "optimization_parameters" => optim_params
    )
)