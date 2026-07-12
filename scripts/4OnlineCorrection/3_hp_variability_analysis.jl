include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

# Choose Parameters file
hsgp_p_key = 40
hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    12 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-26T13:06:11.411.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    40 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_DT/ANG15_HEADING_TWOD_STEP_DT_2026-07-10T14:24:26.288.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
m = 200
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
hsgp_p = HybridZuptInsJl.HsgpParameters(
    hsgp_p.hp, hsgp_p.d, m, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)

data_key = "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = 15
train_ratio = 0.45

rmse_fun = HybridZuptInsJl.make_rmse_evaluator(data_dir, trial_id, train_ratio, FEATURE_TYPE, FRAME, m)

# Experiment variables
groups = [:yaw]
log_range = (-1.0, 1.0)
n_steps = 3
baseline_included = true

specs = HybridZuptInsJl.make_hp_param_grid(hsgp_p.hp, groups; log_range=log_range, n_steps=n_steps)
# specs = HybridZuptInsJl.make_stats_param_grid(hsgp_p; log_range=(-1.0, 1.0), n_steps=n_steps)

##
# Now vary hyperparameters
df = HybridZuptInsJl.vary_hsgp_hyperparameters(
    hsgp_p,
    rmse_fun,                      # <-- uses only the HSGP hyperparams
    specs[1];
    include_baseline=baseline_included
)

## Save things 


using JSON, Dates, CSV
outdir = "out/4OnlineCorrection/3HpVariabilityAnalysis"

# Construct filenames
time = string(Dates.now())

base_name = "$(meta["data_key"])$(trial_id)_$(FRAME)_$(FEATURE_TYPE)_$(time)"
csv_path = joinpath(outdir, "$base_name.csv")
json_path = joinpath(outdir, "$base_name.json")

# Save DataFrame to CSV
CSV.write(csv_path, df)

# Save metadata to JSON
metadata = Dict(
    "data_key" => meta["data_key"],
    "trial_id" => trial_id,
    "frame" => string(FRAME),
    "feature_type" => string(FEATURE_TYPE),
    "log10_range" => log_range,
    "n_steps" => n_steps,
    "baseline_included" => baseline_included,
    "grid" => HybridZuptInsJl.grid_to_dict(specs[2]),
    "timestamp" => time
)

open(json_path, "w") do f
    JSON.print(f, metadata, 4)
end

println("Saved CSV: $csv_path")
println("Saved JSON: $json_path")

## Plot Hp sensitivity
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using CairoMakie
outdir = "out/4OnlineCorrection/3HpVariabilityAnalysis"
basename_key = 32
basename = Dict(
    11 => "ANG15_BODY_TWOD_STEP_DT_2026-05-29T13:47:42.748",
    16 => "ANG15_BODY_TWOD_STEP_DT_2026-06-01T10:20:18.078",
    30 => "ANG15_HEADING_TWOD_STEP_DT_2026-06-26T16:36:18.693",
    31 => "ANG15_HEADING_TWOD_STEP_DT_2026-06-26T17:59:58.069",
    32 => "ANG15_HEADING_TWOD_STEP_DT_2026-06-26T18:47:22.051"
)[basename_key]

csv_file = joinpath(outdir, "$basename.csv")
json_file = joinpath(outdir, "$basename.json")

df, metadata = HybridZuptInsJl.load_hp_variation_results(csv_file, json_file)

grid = HybridZuptInsJl.grid_from_dict(metadata["grid"])
log_range = (
    float(metadata["log10_range"][1]), float(metadata["log10_range"][2]))
with_theme(theme_ggplot2()) do
    HybridZuptInsJl.plot_hp_sensitivity(df, grid, log_range; save_path="out/4OnlineCorrection/3HpVariabilityAnalysis/plots/$basename.svg")
end
