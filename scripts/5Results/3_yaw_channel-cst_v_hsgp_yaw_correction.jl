include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames

data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)

# 1. Define datasets / trials to process
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    "Angermann" => (data_dir["ANG2"], [1, 2, 3, 4, 8, 9, 13, 15, 16]),
    "TuDCSC" => (data_dir["DCSC"], [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]),
)

# 2. Align INS / GT trajectories for every trial
aligned = HybridZuptInsJl.collect_aligned_trajectories(data_dict)

## 3. Load HSGP hyperparameters / Input feature type
m=200
hsgp_p_key = 42
hsgp_p_path = Dict{Int,String}(
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no ouptut norm; DCSC
)[hsgp_p_key]
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
hsgp_p = HybridZuptInsJl.basecopy(hsgp_p; new_m=m)
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

## 4. Define correction methods to compare
estimators = OrderedDict(
    "Base (no correction)" => HybridZuptInsJl.BaseEstimator,
    "Decoupled Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)

output_channels = [:yaw]

train_ratios = [0.5]

## 5. Run the sweep
results_df = HybridZuptInsJl.run_online_correction_sweep(
    aligned,
    FRAME,
    FEATURE_TYPE,
    hsgp_p,
    train_ratios,
    estimators,
    output_channels,
)
## 6. Plot
import CairoMakie, Dates
base_dir = "out/Results/3_yaw_channel/Const_v_Hsgp_yaw_correction"

time = string(Dates.now())
filename = "$(time)_dataset_comparison_yaw_only_correction.svg"
path = joinpath(base_dir, filename)
CairoMakie.with_theme(CairoMakie.theme_ggplot2()) do
    fig_rmse = HybridZuptInsJl.boxplot_dataset_comparison(
        results_df;
        metric=:rmse_rate,
        train_ratio=0.5,
        save_path=path,
    )
end