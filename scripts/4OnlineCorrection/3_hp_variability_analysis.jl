include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

# Choose Parameters file
hsgp_p_key = 21
hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    12 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-26T13:06:11.411.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)

data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision"
)[meta["data_key"]]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = meta["trial_id"]
m = 200
margin = meta["margin"]
train_ratio = 0.4


rmse_fun = HybridZuptInsJl.make_rmse_evaluator(data_dir, trial_id, train_ratio, FEATURE_TYPE, FRAME, m)

# Experiment variables
groups = [:pos_1, :pos_2, :pos_3, :yaw]
log_range = (-1.0, 1.0)
n_steps = 3
baseline_included = true
specs = HybridZuptInsJl.make_hp_param_grid(hsgp_p.hp, groups; log_range=log_range, n_steps=n_steps)
stat_specs = HybridZuptInsJl.make_stats_param_grid(hsgp_p; log_range=(-1.0, 1.0), n_steps=3)
##
# Now vary hyperparameters
df = HybridZuptInsJl.vary_hsgp_hyperparameters(
    hsgp_p,
    rmse_fun,                      # <-- uses only the HSGP hyperparams
    stat_specs
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
    "groups" => groups,    # convert Symbols to String
    "log10_range" => log_range,
    "n_steps" => n_steps,
    "baseline_included" => baseline_included,
    "timestamp" => time
)

open(json_path, "w") do f
    JSON.print(f, metadata, 4)
end

println("Saved CSV: $csv_path")
println("Saved JSON: $json_path")

## Plot Hp sensitivity
outdir = "out/4OnlineCorrection/3HpVariabilityAnalysis"
basename_key = 12
basename = Dict(
    10 => "ANG15_BODY_TWOD_STEP_DT_2026-05-29T12:38:13.301",
    11 => "ANG15_BODY_TWOD_STEP_DT_2026-05-29T13:47:42.748",
    12 => "ANG15_BODY_TWOD_STEP_DT_2026-05-29T14:27:17.328"
)[basename_key]

csv_file = joinpath(outdir, "$basename.csv")
json_file = joinpath(outdir, "$basename.json")

df, meta = HybridZuptInsJl.load_hp_variation_results(csv_file, json_file)
HybridZuptInsJl.plot_hp_sensitivity(df, meta; save_path="out/4OnlineCorrection/3HpVariabilityAnalysis/sensitivity.png")