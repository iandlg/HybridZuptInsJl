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

# σ_n is therefore excluded from the sweep below (`include_noise`), leaving the
# length scale ℓ_s and the signal variance σ_f. Flip both together if you want
# the noise term back.
sweep_noise = pred_includes_noise

rmse_fun = HybridZuptInsJl.make_rmse_evaluator(
    data_dir_path, trial_id, train_ratio, FEATURE_TYPE, FRAME;
    m=m, output_channel_idxs=output_channel_idxs,
    hsgp_estimator_factory=HybridZuptInsJl.DecoupledHsgpEstimator,
    noise_spec=noise_spec,
    pred_includes_noise=pred_includes_noise,
)

# Experiment variables
groups = [Symbol(HybridZuptInsJl._OUTPUT_NAMES[idx]) for idx in output_channel_idxs]

log_range = (-1.0, 1.0)
n_steps = 21
baseline_included = true

# Two families of parameters, swept in one pass against one baseline:
#
#   * the GP hyperparameters -- ℓ_s, σ_f, and σ_n when `sweep_noise` connects it;
#   * the input normalisation statistics -- mean, std and centering. These are as
#     much a fitted input to the estimator as the kernel hyperparameters, and a
#     wrong input std is not visible anywhere else in the chapter.
#
# WAS: `specs` was assigned twice and only the second assignment survived, so the
# hyperparameter grid was built and thrown away and every figure in this section
# showed the statistics *instead of* the hyperparameters -- with no way to rank
# one family against the other, which is what the signed-range figure is for.
# See notes/007-input-normalisation-sensitivity.md.
include_stats_params = true
include_output_stats = false   # output mean/std: fitted per channel, off by default

hp_specs, hp_grid = HybridZuptInsJl.make_hp_param_grid(hsgp_p.hp, output_channel_idxs;
    log_range=log_range, n_steps=n_steps, include_noise=sweep_noise)

stats_specs, stats_grid = include_stats_params ?
                          HybridZuptInsJl.make_stats_param_grid(hsgp_p;
    log_range=log_range, n_steps=n_steps, output_channel_idxs=output_channel_idxs,
    include_output_params=include_output_stats) :
                          (HybridZuptInsJl.ParamSpec[], nothing)

# One vector, so `vary_hsgp_parameters` evaluates the baseline once and every row
# of the frame is a relative change against that same number.
specs = vcat(hp_specs, stats_specs)

##
# Now vary the parameters
df = HybridZuptInsJl.vary_hsgp_parameters(
    hsgp_p,
    rmse_fun,
    specs;
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
    # One grid per family: the grid is only the panel layout for
    # plot_hp_sensitivity, which draws one row per parameter group.
    "grid" => HybridZuptInsJl.grid_to_dict(hp_grid),
    "stats_grid" => isnothing(stats_grid) ? nothing : HybridZuptInsJl.grid_to_dict(stats_grid),
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
# `nothing` for a run that swept hyperparameters only, and missing entirely from
# runs saved before the statistics were added to this script.
stats_grid_meta = get(plot_meta, "stats_grid", nothing)
plot_log_range = (float(plot_meta["log10_range"][1]), float(plot_meta["log10_range"][2]))

results_figure() do
    HybridZuptInsJl.plot_hp_sensitivity(plot_df, grid, plot_log_range;
        save_path=results_path(SECTION, "$(plot_name)_param_var.pdf"))
end

# Same per-parameter curves for the normalisation statistics: one row per family
# (mean / std / centering), one column per feature dimension.
if !isnothing(stats_grid_meta)
    results_figure() do
        HybridZuptInsJl.plot_hp_sensitivity(plot_df,
            HybridZuptInsJl.grid_from_dict(stats_grid_meta), plot_log_range;
            save_path=results_path(SECTION, "$(plot_name)_stats_var.pdf"))
    end
end

# Signed range: which parameters can be improved, and in which direction.
# This is the one to read first, and it is the one figure that puts both families
# on the same axis -- a hyperparameter and an input statistic are directly
# comparable there because both are quoted as percent RMSE change against the
# same baseline. `plot_max_relative_change` below reduces each parameter to
# maximum(relative_change), which for the yaw length scale is 0.0 -- it renders
# as a zero-height bar indistinguishable from a hyperparameter that never reaches
# the code, while its true range is [-0.60, 0.0].
results_figure() do
    HybridZuptInsJl.plot_signed_relative_change(plot_df;
        save_path=results_path(SECTION, "$(plot_name)_signed_rel_change.pdf"))
end

# Close-ups of the most consequential parameters, one figure each. The
# signed-range plot above compresses each parameter to the interval [min, max],
# which says how far RMSE moved but not *how* it got there -- and the shape is
# often the result: for the yaw length scale the baseline sits on a local
# optimum, so both directions move RMSE the same way, which a bar cannot show.
#
# Both families are addressable here. `hp_param_name` names a GP hyperparameter
# by channel and kind; `stat_param_name` names a normalisation statistic by
# family and feature dimension. Spelling both out beats writing "yaw[2]" and
# "input_std[2]" as literals, where the bracketed number means a hyperparameter
# kind in the first and a feature dimension in the second.
#
# The defaults are the three widest rows of notes/007 §5 that are not each
# other's family-mates: the yaw length scale, the input std of feature dim 2
# (the widest row in that sweep, wider than any hyperparameter), and the
# centering of the same dimension.
focus_params = [
    HybridZuptInsJl.hp_param_name(:yaw, :length_scale),
    HybridZuptInsJl.stat_param_name(:input_std, 2),
    HybridZuptInsJl.stat_param_name(:input_center, 2),
]

# A sweep run with `include_stats_params=false` (or an older CSV predating the
# statistics) has no rows for the stats entries, and `plot_hp_param_sensitivity`
# throws for a parameter it cannot find. Skipping with a warning keeps the rest
# of the figures being written; a typo'd name still shows up, as a warning
# naming a parameter that is not in the frame.
swept_params = Set(plot_df.parameter)
for focus_param in focus_params
    if !(focus_param in swept_params)
        @warn "Skipping close-up: \"$focus_param\" was not swept in $plot_name"
        continue
    end
    # "yaw[2]" -> "yaw_2", "input_std[2]" -> "input_std_2": filename-safe.
    focus_slug = HybridZuptInsJl.param_slug(focus_param)
    results_figure() do
        HybridZuptInsJl.plot_hp_param_sensitivity(plot_df, focus_param;
            log_range=plot_log_range,
            save_path=results_path(SECTION, "$(plot_name)_$(focus_slug)_sensitivity.pdf"))
    end
end


results_figure() do
    HybridZuptInsJl.plot_max_relative_change(plot_df;
        save_path=results_path(SECTION, "$(plot_name)_max_rel_change.pdf"))
end
