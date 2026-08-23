# Section 2 (sensitivity): how sensitive is performance to the GP
# hyperparameters, one at a time?
#
# LIMITS OF THIS DESIGN, worth stating in the thesis rather than discovering in
# the viva: n = 1 trajectory, one noise realisation, one-at-a-time perturbation
# (so no interactions), and the pipeline is deterministic given the fixed seed,
# which means the y-axis has no noise floor to compare the curves against. A
# 5% wiggle here is not known to be larger than run-to-run variation, because
# run-to-run variation is zero by construction. Treat it as a screening
# experiment that says which parameters are worth studying properly.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")

# Choose Parameters file
hsgp_p_key = 42
m = 200
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

data_key = "ANG2"
data_dir_path = data_dir(data_key)

trial_id = 15
train_ratio = 0.45

output_channel_idxs = [1, 2, 4]

noise_spec = HybridZuptInsJl.NoiseSpec() # ; pos_std=0.05, att_std=5*pi/180, tag="Position & Heading Noise (0.05m, ±5°)"

# NOTE: `pred_includes_noise` controls whether the GP `noise` hyperparameter
# reaches the estimator at all. With the default `false`, DecoupledHsgpEstimator
# loads sigma_n and never reads it, so every `noise` row of this sweep returns a
# bit-identical RMSE and the figure shows three flat lines that look like an
# insensitivity result but are a dead knob. vary_hsgp_parameters now warns when
# that happens. Set it to `true` to sweep the noise hyperparameters meaningfully.
pred_includes_noise = false

rmse_fun = HybridZuptInsJl.make_rmse_evaluator(
    data_dir_path, trial_id, train_ratio, FEATURE_TYPE, FRAME;
    m=m, output_channel_idxs=output_channel_idxs,
    hsgp_estimator_factory=HybridZuptInsJl.DecoupledHsgpEstimator,
    noise_spec=noise_spec,
    pred_includes_noise=pred_includes_noise,
)

# Experiment variables
groups = [Symbol(HybridZuptInsJl._OUTPUT_NAMES[idx]) for idx in output_channel_idxs]

log_range = (-0.5, 0.5)
n_steps = 5
baseline_included = true

specs = HybridZuptInsJl.make_hp_param_grid(hsgp_p.hp, output_channel_idxs; log_range=log_range, n_steps=n_steps)
# specs = HybridZuptInsJl.make_stats_param_grid(hsgp_p;
#     log_range=log_range, n_steps=n_steps, output_channel_idxs=output_channel_idxs,
#     include_output_params=false)

##
# Now vary hyperparameters
df = HybridZuptInsJl.vary_hsgp_parameters(
    hsgp_p,
    rmse_fun,                      # <-- uses only the HSGP hyperparams
    specs[1];
    include_baseline=baseline_included
)

## Save things 


using JSON, CSV
const SECTION = "2_HypSensitivity/SensitivityAnalysis"
outdir = joinpath("out/Results", SECTION, "data")
mkpath(outdir)

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
    "trial_id" => trial_id,
    "train_ratio" => train_ratio,
    "noise_spec_tag" => noise_spec.tag,
    "pred_includes_noise" => pred_includes_noise,
    "hsgp_p_key" => hsgp_p_key,
    "base_parameters_metadata" => meta
)

open(json_path, "w") do f
    JSON.print(f, metadata, 4)
end

println("Saved CSV: $csv_path")
println("Saved JSON: $json_path")
## Plot Hp sensitivity
# Set `replot_basename` to re-plot a previously saved sweep, or leave it
# `nothing` to plot the sweep just computed above. The old version always
# re-read a hard-coded basename from a dict of timestamps, so editing the
# compute cell above had no effect on the figures unless you also remembered to
# add a key down here.
replot_basename = nothing

# Both branches load from disk, so the freshly computed sweep goes through the
# exact same JSON round-trip as a replot -- grid_from_dict then sees identically
# typed input either way.
plot_name = isnothing(replot_basename) ? base_name : replot_basename
plot_df, plot_meta = HybridZuptInsJl.load_hp_variation_results(
    joinpath(outdir, "$plot_name.csv"),
    joinpath(outdir, "$plot_name.json"))

grid = HybridZuptInsJl.grid_from_dict(plot_meta["grid"])
plot_log_range = (float(plot_meta["log10_range"][1]), float(plot_meta["log10_range"][2]))

results_figure() do
    HybridZuptInsJl.plot_hp_sensitivity(plot_df, grid, plot_log_range;
        save_path=results_path(SECTION, "$(plot_name)_param_var.svg"))
end

# Signed range: which parameters can be improved, and in which direction.
# This is the one to read first. `plot_max_relative_change` below reduces each
# parameter to maximum(relative_change), which for the yaw length scale is 0.0 --
# it renders as a zero-height bar indistinguishable from a hyperparameter that
# never reaches the code, while its true range is [-0.60, 0.0].
results_figure() do
    HybridZuptInsJl.plot_signed_relative_change(plot_df;
        save_path=results_path(SECTION, "$(plot_name)_signed_rel_change.svg"))
end

results_figure() do
    HybridZuptInsJl.plot_max_relative_change(plot_df; color_offset=2,
        save_path=results_path(SECTION, "$(plot_name)_max_rel_change.svg"))
end
