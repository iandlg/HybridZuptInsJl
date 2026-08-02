include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections, Dates
import CSV

# Choose Parameters file
hsgp_p_key = 30

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
data_key = "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

hsgp_p = HybridZuptInsJl.HsgpParameters(
    hsgp_p.hp, hsgp_p.d, 300, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_ids = HybridZuptInsJl.list_trial_ids(data_dir)
train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
correctors = OrderedDict{String,HybridZuptInsJl.AbstractEstimator}(
    "Default" => HybridZuptInsJl.DefaultCorrector(300),
    "Static" => HybridZuptInsJl.StaticCorrectorV2(300),
    "Split" => HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, 300), hsgp_p),
    "Slam" => HybridZuptInsJl.JointHsgpEstimator(round(Int, 300), hsgp_p)
)

dataset = HybridZuptInsJl.collect_dataset(data_dir, trial_ids, train_ratios, correctors; frame=FRAME, feature_type=FEATURE_TYPE)

df = HybridZuptInsJl.performance_dataframe(dataset)

dirname = "out/4OnlineCorrection/4PerformanceResults"
time = string(Dates.now())
CSV.write(joinpath(dirname, "results_$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).csv"), df)
##
include("../../src/HybridZuptInsJl.jl");

using .HybridZuptInsJl;
using GLMakie, OrderedCollections, Dates
using Dates, CSV, DataFrames
dirname = "out/4OnlineCorrection/4PerformanceResults"
df = CSV.read("out/4OnlineCorrection/4PerformanceResults/results_ANG2_HEADING_TWOD_STEP_DT_2026-06-26T16:02:48.787.csv", DataFrame)
time = string(Dates.now())

with_theme(theme_ggplot2()) do
    fig = HybridZuptInsJl.plot_corrector_boxplots(df, :rmse; save_path=joinpath(dirname, "RMSE_$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).svg"), corrector_names=["Default", "Static", "Split", "Slam"])
end

with_theme(theme_ggplot2()) do
    fig = HybridZuptInsJl.plot_corrector_boxplots(df, :rmse_rate; save_path=joinpath(dirname, "RMSE_RATE_$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).svg"), corrector_names=["Default", "Static", "Split", "Slam"])
end

with_theme(theme_ggplot2()) do
    fig = HybridZuptInsJl.plot_corrector_boxplots(df, :rmse; save_path=joinpath(dirname, "RMSE_noouliers_$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).svg"), show_outliers=false, corrector_names=["Default", "Static", "Split", "Slam"])
end

with_theme(theme_ggplot2()) do
    fig = HybridZuptInsJl.plot_corrector_boxplots(df, :rmse_rate; save_path=joinpath(dirname, "RMSE_RATE_noouliers_$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).svg"), show_outliers=false, corrector_names=["Default", "Static", "Split", "Slam"])
end