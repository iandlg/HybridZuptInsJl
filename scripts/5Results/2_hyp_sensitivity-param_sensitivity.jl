include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

# Choose Parameters file
hsgp_p_key = 42
hsgp_p_path = Dict{Int,String}(
    41 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG23_HEADING_TWOD_STEP_YAW_2026-07-13T12:28:30.952.json",
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no ouptut norm; DCSC
)[hsgp_p_key]

# Load parameters with corresponding metatdata
m = 200
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
hsgp_p = HybridZuptInsJl.basecopy(hsgp_p; new_m=m)

data_key = "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = 15
train_ratio = 0.45

output_channel_idxs = [1, 2, 4]

rmse_fun = HybridZuptInsJl.make_rmse_evaluator(
    data_dir, trial_id, train_ratio, FEATURE_TYPE, FRAME;
    m=m, output_channel_idxs=output_channel_idxs, hsgp_estimator_factory=HybridZuptInsJl.DecoupledHsgpEstimator)

# Experiment variables
groups = [Symbol(HybridZuptInsJl._OUTPUT_NAMES[idx]) for idx in output_channel_idxs]

log_range = (-0.5, 0.5)
n_steps = 5
baseline_included = true

specs = HybridZuptInsJl.make_hp_param_grid(hsgp_p.hp, output_channel_idxs; log_range=log_range, n_steps=n_steps)
specs = HybridZuptInsJl.make_stats_param_grid(hsgp_p;
    log_range=log_range, n_steps=n_steps, output_channel_idxs=output_channel_idxs,
    include_output_params=false)

##
# Now vary hyperparameters
df = HybridZuptInsJl.vary_hsgp_parameters(
    hsgp_p,
    rmse_fun,                      # <-- uses only the HSGP hyperparams
    specs[1];
    include_baseline=baseline_included
)

## Save things 


using JSON, Dates, CSV
outdir = "out/Results/2_HypSensitivity/SensitivityAnalysis/data"

# Construct filenames
time = string(Dates.now())

base_name = "$(data_key)$(trial_id)_$(FRAME)_$(FEATURE_TYPE)_$(time)"
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
    "timestamp" => time,
    "base_parameters_metadata" => meta
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
outdir = "out/Results/2_HypSensitivity/SensitivityAnalysis/data"
basename_key = 61
basename = Dict(
    59 => "ANG215_HEADING_TWOD_STEP_YAW_2026-08-14T14:50:07.279",
    60 => "ANG215_HEADING_TWOD_STEP_YAW_2026-08-14T15:10:36.451",
    61 => "ANG215_HEADING_TWOD_STEP_YAW_2026-08-14T15:19:22.230",
    62 => "ANG215_HEADING_TWOD_STEP_YAW_2026-08-14T15:26:20.692"
)[basename_key]

csv_file = joinpath(outdir, "$basename.csv")
json_file = joinpath(outdir, "$basename.json")

df, metadata = HybridZuptInsJl.load_hp_variation_results(csv_file, json_file)

grid = HybridZuptInsJl.grid_from_dict(metadata["grid"])
log_range = (
    float(metadata["log10_range"][1]), float(metadata["log10_range"][2]))
with_theme(theme_ggplot2()) do
    HybridZuptInsJl.plot_hp_sensitivity(df, grid, log_range; save_path="out/Results/2_HypSensitivity/SensitivityAnalysis/$(basename)_param_var.svg")
end
with_theme(theme_ggplot2()) do
    HybridZuptInsJl.plot_max_relative_change(df; color_offset=2, save_path="out/Results/2_HypSensitivity/SensitivityAnalysis/$(basename)_max_rel_change.svg")
end
