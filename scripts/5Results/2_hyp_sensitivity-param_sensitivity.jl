# Section 2 (sensitivity): how sensitive is performance to the GP
# hyperparameters and to the input normalisation, one at a time?
#
# WHAT THIS DESIGN DOES AND DOES NOT SUPPORT.
#
# Each parameter is perturbed in the group that matches its type. Scales -- the
# length scales, the signal variances, the input std -- get a multiplicative
# probe over decades, which is the right orbit for a positive, unit-equivariant
# quantity with no fixed point. Locations -- the input mean and the centering
# offset -- get an additive probe in units of the matching input std, because a
# multiplier on a location is sized by its own base value, cannot cross zero,
# reverses direction on a negative base, and fixes zero. Mixing the two, which
# is what this script used to do, made the location rows un-rankable against the
# scale rows: their span was partly just a reading of mu/sigma.
#
# The sweep is repeated over several trials and each trial is scored against its
# OWN baseline, so the design is paired and every curve carries an across-trial
# band. That band is the reference distribution -- a parameter whose band
# contains zero at every probe has not been shown to matter. It is NOT a noise
# floor: the pipeline is deterministic under a fixed seed, so the unperturbed
# point is exactly zero in every trial by construction. Interactions are still
# untested; this is still one-at-a-time.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")

using JSON, CSV, DataFrames

## ----- Configuration -------------------------------------------------------

# `true` runs one trial at 5 steps -- enough to check the probes and the wiring
# before committing to the full sweep. Cost is
# `n_trials * n_params * n_steps + n_trials` filter runs; n_params is 15.
smoke_test = false

hsgp_p_key = 42
m = 200
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

data_key = "ANG2"
data_dir_path = data_dir(data_key)

# Trials to repeat the sweep over. Held-out by default: key 42 was trained on
# `train_ids(data_key)`, so sweeping all of `trial_ids(data_key)` would run
# mostly on the hyperparameters' own training walks and the sensitivity would be
# partly a statement about fit rather than about transfer. Swap in
# `trial_ids(data_key)` to sweep everything, knowing that.
sweep_trial_ids = trial_ids(data_key)

train_ratio = 0.5
output_channel_idxs = [1, 2, 4]

noise_spec = HybridZuptInsJl.NoiseSpec() # ; pos_std=0.05, att_std=5*pi/180, tag="Position & Heading Noise (0.05m, ±5°)"

# `pred_includes_noise` controls whether the GP `noise` hyperparameter reaches
# the estimator at all. With the default `false`, DecoupledHsgpEstimator loads
# sigma_n and never reads it, so every `noise` row would return a bit-identical
# RMSE -- three flat lines that look like an insensitivity result but are a dead
# knob. `sweep_noise` therefore tracks it, and `vary_hsgp_parameters` warns if
# anything else comes back inert.
pred_includes_noise = false
sweep_noise = pred_includes_noise

# Probe ranges. `n_steps` must be ODD so both identities -- multiplier 1 and
# offset 0 -- are hit exactly and the baseline sits on every curve.
n_steps = smoke_test ? 5 : 7
log_range = (-1.0, 1.0)     # scale families: decades
delta_range = (-2.0, 2.0)   # location families: units of sigma_x (mu_x) or z (c_x)

# The box diagnostic is closed-form in (mu_x, sigma_x, c_x) and costs no filter
# runs, so it is swept wider and finer than the RMSE sweep. That is what lets the
# figure answer "where would a location perturbation leave the box" even though
# no physically sensible mean shift gets close.
box_n_steps = 121
box_log_range = (-1.5, 1.5)
box_delta_range = (-12.0, 12.0)

# Parameters that get their own single-panel figure. `hp_param_name` names a GP
# hyperparameter by channel and kind; `stat_param_name` names a normalisation
# statistic by family and feature dimension -- spelling both out beats writing
# "yaw[2]" and "input_std[2]" as literals, where the bracketed number means a
# hyperparameter kind in the first and a feature dimension in the second.
# Normalisation parameters additionally get domain-containment shading.
focus_params = [
    HybridZuptInsJl.hp_param_name(:yaw, :length_scale),
    HybridZuptInsJl.stat_param_name(:input_std, 2),
    HybridZuptInsJl.stat_param_name(:input_center, 2),
]

if smoke_test
    sweep_trial_ids = sweep_trial_ids[1:1]
end
@info "Sweeping trials $sweep_trial_ids at $n_steps steps ($(smoke_test ? "SMOKE TEST" : "full run"))"

## ----- Specs ---------------------------------------------------------------

# Two families, concatenated into one vector so `vary_hsgp_parameters` evaluates
# each trial's baseline once and every row is a relative change against that same
# number. The grids stay separate because a grid is only a panel layout.
include_stats_params = true
include_output_stats = false   # output_mean is fitted to exactly 0; see make_stats_param_grid

hp_specs, hp_grid = HybridZuptInsJl.make_hp_param_grid(hsgp_p.hp, output_channel_idxs;
    log_range=log_range, n_steps=n_steps, include_noise=sweep_noise)

stats_specs, stats_grid = include_stats_params ?
                          HybridZuptInsJl.make_stats_param_grid(hsgp_p;
    log_range=log_range, delta_range=delta_range, n_steps=n_steps,
    output_channel_idxs=output_channel_idxs,
    include_output_params=include_output_stats) :
                          (HybridZuptInsJl.ParamSpec[], nothing)

specs = vcat(hp_specs, stats_specs)

# Probe sanity, before spending any time on filter runs. These are the four
# properties the typed probe was introduced to guarantee, so a regression in
# `make_stats_param_grid` should stop the run rather than produce a plausible
# figure.
for spec in specs
    base_val = spec.get_current(hsgp_p)
    probes = [HybridZuptInsJl.probe_coordinate(spec, hsgp_p, base_val, v)
              for v in spec.value_generator(base_val)]
    identity_probe = spec.probe_kind === :multiplicative ? 1.0 : 0.0
    @assert any(isapprox.(probes, identity_probe; atol=1e-12)) "$(spec.name): probe grid misses the unperturbed point $identity_probe"
    @assert all(diff(probes) .> 0) "$(spec.name): probe axis is not increasing"
    @assert issorted(probes) "$(spec.name): probe axis is not sorted"
end
@info "Probe grids OK for $(length(specs)) parameters"

## ----- Sweep ---------------------------------------------------------------

const SECTION = "2_HypSensitivity/SensitivityAnalysis"
outdir = joinpath("out/Results", SECTION, "data")
mkpath(outdir)

time = string(Dates.now())
base_name = "$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time)"

make_evaluator(tid) = HybridZuptInsJl.make_rmse_evaluator(
    data_dir_path, tid, train_ratio, FEATURE_TYPE, FRAME;
    m=m, output_channel_idxs=output_channel_idxs,
    hsgp_estimator_factory=HybridZuptInsJl.DecoupledHsgpEstimator,
    noise_spec=noise_spec,
    pred_includes_noise=pred_includes_noise,
)

df = HybridZuptInsJl.sweep_over_trials(
    hsgp_p, specs, make_evaluator, sweep_trial_ids;
    include_baseline=true,
    checkpoint_dir=joinpath(outdir, base_name),
)

## ----- Box occupancy -------------------------------------------------------

# Computed for every swept trial, not just one: the boundary is not a constant
# (×0.13-×0.47 across the ANG2 trials), so a single trial's exit point would
# misstate it. Closed-form in the normalisation statistics, so this is one
# feature extraction per trial and no filter runs.
box_specs, _ = HybridZuptInsJl.make_stats_param_grid(hsgp_p;
    log_range=box_log_range, delta_range=box_delta_range, n_steps=box_n_steps,
    output_channel_idxs=output_channel_idxs, include_output_params=false)

box_df = HybridZuptInsJl.box_exit_over_trials(data_dir_path, sweep_trial_ids,
    hsgp_p, box_specs, FEATURE_TYPE; ref_frame=FRAME)
exit_df = HybridZuptInsJl.box_exit_points(box_df)

# LL was built from the training features with a margin, so the trained
# parameters should place every stride inside the domain. A trial that is
# already outside is not a reason to stop -- it is a finding (notes/009 §6) --
# but it must be reported rather than absorbed into the figures silently.
for tid in sweep_trial_ids
    X = HybridZuptInsJl.raw_features(data_dir_path, tid;
        ref_frame=FRAME, feature_type=FEATURE_TYPE)
    occ = HybridZuptInsJl.box_occupancy(X, hsgp_p, FEATURE_TYPE)
    occ.frac_outside == 0.0 ||
        @warn "trial $tid is already outside ±LL at the trained parameters" occ.frac_outside occ.max_z_ratio
end

## ----- Save ----------------------------------------------------------------

csv_path = joinpath(outdir, "$base_name.csv")
json_path = joinpath(outdir, "$base_name.json")
box_path = joinpath(outdir, "$(base_name)_box.csv")
exit_path = joinpath(outdir, "$(base_name)_box_exit.csv")
agree_path = joinpath(outdir, "$(base_name)_agreement.csv")
rank_path = joinpath(outdir, "$(base_name)_ranking.csv")

CSV.write(csv_path, df)
CSV.write(box_path, box_df)
CSV.write(exit_path, exit_df)
# One row per parameter: span, worst probe, across-trial IQR and the sign test.
# This is the table notes/009 §4 quotes, so it is generated rather than
# recomputed by hand whenever the sweep is rerun.
CSV.write(agree_path, HybridZuptInsJl.probe_agreement(df))
# Box geometry behind the ranking figure, including the values that figure clips
# at its frame -- the input std rows run to roughly +100% where the axis stops
# near +49%, so the untruncated numbers have to live somewhere quotable.
CSV.write(rank_path, HybridZuptInsJl.probe_extremes_summary(df))

metadata = Dict(
    "data_key" => meta["data_key"],
    "sweep_trial_ids" => sweep_trial_ids,
    "box_trial_ids" => sweep_trial_ids,
    "frame" => string(FRAME),
    "feature_type" => string(FEATURE_TYPE),
    "log10_range" => log_range,
    "delta_range" => delta_range,
    "box_log10_range" => box_log_range,
    "box_delta_range" => box_delta_range,
    "box_n_steps" => box_n_steps,
    "n_steps" => n_steps,
    "smoke_test" => smoke_test,
    # One grid per family: a grid is only the panel layout for
    # plot_probe_sensitivity, which draws one row per parameter group.
    "grid" => HybridZuptInsJl.grid_to_dict(hp_grid),
    "stats_grid" => isnothing(stats_grid) ? nothing : HybridZuptInsJl.grid_to_dict(stats_grid),
    "timestamp" => time,
    "train_ratio" => train_ratio,
    "noise_spec_tag" => noise_spec.tag,
    "pred_includes_noise" => pred_includes_noise,
    "hsgp_p_key" => hsgp_p_key,
    "base_parameters_metadata" => meta
)

open(json_path, "w") do f
    JSON.print(f, metadata, 4)
end

println("Saved CSV:  $csv_path")
println("Saved JSON: $json_path")
println("Saved box:  $box_path")
println("Saved exit: $exit_path")
println("Saved agree: $agree_path")
println("Saved rank:  $rank_path")
println()
println("Box exit points:")
show(stdout, MIME("text/plain"), exit_df)
println()

## ----- Plot ----------------------------------------------------------------
# Set `replot_basename` to re-plot a previously saved sweep, or leave it
# `nothing` to plot the sweep just computed above.
replot_basename = nothing

# Both branches load from disk, so the freshly computed sweep goes through the
# exact same JSON round-trip as a replot -- grid_from_dict then sees identically
# typed input either way.
plot_name = isnothing(replot_basename) ? base_name : replot_basename
plot_df, plot_meta = HybridZuptInsJl.load_hp_variation_results(
    joinpath(outdir, "$plot_name.csv"),
    joinpath(outdir, "$plot_name.json"))
plot_box = CSV.read(joinpath(outdir, "$(plot_name)_box.csv"), DataFrame)

grid = HybridZuptInsJl.grid_from_dict(plot_meta["grid"])
stats_grid_meta = get(plot_meta, "stats_grid", nothing)

# GP hyperparameters: one panel per channel and kind, multiplier axis.
results_figure() do
    HybridZuptInsJl.plot_probe_sensitivity(plot_df, grid;
        save_path=results_path(SECTION, "$(plot_name)_param_var.pdf"))
end

# Normalisation statistics: one row per family, one column per feature
# dimension. The two location rows are now on a linear, zero-centred offset axis
# and the std row stays on the multiplier axis.
if !isnothing(stats_grid_meta)
    results_figure() do
        HybridZuptInsJl.plot_probe_sensitivity(plot_df,
            HybridZuptInsJl.grid_from_dict(stats_grid_meta);
            save_path=results_path(SECTION, "$(plot_name)_stats_var.pdf"))
    end
end

# Where the features leave the fixed domain, against what that does to RMSE.
#
# A consistency check this figure makes visible: `mu_x[d] += delta*sigma_x[d]`
# and `c_x[d] += delta` both send z -> z - delta, so for a non-angle dimension
# the two families must produce bit-identical RMSE -- and they do, to 1e-15, for
# dims 1 and 2. Dim 3 is the yaw feature, where `normalize_feature!` wraps to
# +-pi *between* the mean subtraction and the division, so shifting the mean
# changes what gets wrapped and shifting the centering does not. The two curves
# coincide there too until the shift is large enough to carry strides across the
# wrap (delta = -2 in this artifact). That the two location families now agree at
# all is itself the fix: under the old multiplicative probe they were sized by
# mu/sigma and by c respectively, and were reported as separate findings with
# spans differing by an order of magnitude.
results_figure() do
    HybridZuptInsJl.plot_box_exit(plot_df, plot_box;
        save_path=results_path(SECTION, "$(plot_name)_box_exit.pdf"))
end

# Ranking: which parameters move RMSE, by how much, and whether the trials
# agree. This is the one to read first. Its x limits come from the bars, not the
# data -- see plot_probe_ranking on why the previous version was unreadable.
results_figure() do
    HybridZuptInsJl.plot_probe_ranking(plot_df;
        xlims=(-25.0, 70.0),
        save_path=results_path(SECTION, "$(plot_name)_ranking.pdf"))
end

# Close-ups, one figure per parameter, sized for the write-up. The ranking
# compresses each parameter to [min, max], which says how far RMSE moved but not
# how it got there -- and the shape is often the result: the yaw length scale
# saturates above x2, which is a bar of the same height as a curve that rises
# steadily. Normalisation parameters also get domain-containment shading, so the
# box story travels with the parameter instead of needing the 3x3 grid figure.
swept_params = Set(plot_df.parameter)
for focus_param in focus_params
    if !(focus_param in swept_params)
        @warn "Skipping close-up: \"$focus_param\" was not swept in $plot_name"
        continue
    end
    focus_slug = HybridZuptInsJl.param_slug(focus_param)
    results_figure() do
        HybridZuptInsJl.plot_param_closeup(plot_df, focus_param;
            box_df=plot_box,
            save_path=results_path(SECTION, "$(plot_name)_$(focus_slug)_sensitivity.pdf"))
    end
end
