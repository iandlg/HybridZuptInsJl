# Does propagating the feature's uncertainty through the HSGP (`J Σ_z J'`, with
# `J = ∂y/∂z`; `propagate_input` on `JointStrideHsgpEstimator`) help the V4 corrector?
#
# Two sweeps over the same cells, one per setting (the sweep takes one
# `estimator_kwargs` per call), paired per trial against plain "HSGP" and against
# "ZUPT only"; then test-half NEES consistency of both HSGP arms at one ratio, since
# the term acts on the covariance first.
#
# Run from the repository root:
#     julialauncher --project=. -t 1 scripts/5Results/7_input_uncertainty.jl

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics
import CSV

const H = HybridZuptInsJl

data_keys = ["ANG2", "DCSC"]
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    k => (data_dir(k), trial_ids(k)) for k in data_keys)

const SECTION = "7_InputUncertainty"
const DATA_SECTION = "$(SECTION)/data"

# File names under out/Results/7_InputUncertainty/data/ to re-plot a finished run
# instead of recomputing it; `nothing` runs it.
results_csv = nothing
nees_csv = nothing

filter_tag = "V4"
hsgp_p_key = 42
m = 200
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

output_channels = [:pos_1, :pos_2, :yaw]
train_ratios = [0.3, 0.5, 0.7]
nees_train_ratio = 0.5
alloc = 300

const BASE = "ZUPT only"
const HSGP = "HSGP"
const HSGP_IN = "HSGP + input unc."
arm_kwargs = OrderedDict(HSGP => (;), HSGP_IN => (propagate_input=true,))

tag = "$(filter_tag)_key$(hsgp_p_key)_$(join(data_keys, "-"))_$(FRAME)_$(FEATURE_TYPE)"
aligned = isnothing(results_csv) || isnothing(nees_csv) ? H.collect_aligned_trajectories(data_dict) : nothing

## 1. RMSE sweep
score_cols = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :noise_spec_tag, :noise_spec_order, :seed,
    :rmse, :rmse_rate, :rmse_yaw]

sweep(estimators; kwargs...) = H.run_online_correction_sweep(aligned, FRAME, FEATURE_TYPE, hsgp_p,
    train_ratios, estimators, output_channels;
    estimator_alloc=alloc, correction_filter=CORRECTION_FILTERS[filter_tag],
    keep_artifacts=false, kwargs...)

if isnothing(results_csv)
    df_off = sweep(OrderedDict(BASE => H.BaseEstimator, HSGP => CORRECTORS[filter_tag].hsgp))
    df_on = sweep(OrderedDict(BASE => H.BaseEstimator, HSGP_IN => CORRECTORS[filter_tag].hsgp);
        estimator_kwargs=arm_kwargs[HSGP_IN])
    df_on.estimator_order[df_on.estimator .== HSGP_IN] .= 3

    # "ZUPT only" ignores the setting, so both sweeps must agree on it cell by cell:
    # otherwise the two halves are not pairable.
    base_off = sort(df_off[df_off.estimator .== BASE, :], [:dataset_order, :trial_id, :train_ratio])
    base_on = sort(df_on[df_on.estimator .== BASE, :], [:dataset_order, :trial_id, :train_ratio])
    base_off.rmse == base_on.rmse || error("ZUPT-only rmse differs between the two sweeps")

    results_df = vcat(df_off, df_on[df_on.estimator .== HSGP_IN, :])
    csv_path = stamped(DATA_SECTION, "results_$(tag)"; ext="csv")
    CSV.write(csv_path, results_df[:, score_cols])
    @info "Saved results table: $csv_path"
else
    results_df = CSV.read(results_path(DATA_SECTION, results_csv), DataFrame)
end

## 2. Paired contrasts
for metric in (:rmse, :rmse_yaw), ref in (HSGP, BASE)
    paired = H.paired_estimator_contrast(results_df; metric=metric, reference_estimator=ref,
        train_ratios=train_ratios)
    summary = combine(groupby(paired, [:dataset_name, :estimator, :train_ratio]),
        :rel_change_pct => median => :median_pct,
        :delta => (d -> count(<(0), d)) => :wins,
        nrow => :n)
    @info "$metric vs \"$ref\" (rel_change_pct < 0: better than $ref)" summary

    for ds in unique(paired.dataset_name)
        results_figure() do
            H.plot_train_ratio_paired_relative_change(paired, ds;
                metric=metric, show_outliers=true, show_points=true,
                save_path=stamped(SECTION, "paired_$(metric)_vs_$(ref == BASE ? "zupt" : "hsgp")_$(tag)_$(ds)"))
        end
    end
end

## 3. Test-half consistency at one ratio
if isnothing(nees_csv)
    nees_df = DataFrame(dataset_name=String[], trial_id=Int[], estimator=String[],
        nees_pos_ratio=Float64[], nees_yaw_ratio=Float64[],
        mean_nees_pos=Float64[], mean_nees_yaw=Float64[],
        pred_std_pos=Float64[], pred_std_yaw=Float64[])
    for (ds, trials) in aligned, (id, res) in trials, (name, kw) in arm_kwargs
        N = length(res.inertial_updated)
        k0 = floor(Int, nees_train_ratio * N) + 1
        d = H.CorrectorDiagnostics()
        c = CORRECTORS[filter_tag].hsgp(alloc; params=hsgp_p, corrected_channels=output_channels, kw...)
        _, _, _, io, _ = CORRECTION_FILTERS[filter_tag](res.inertial_updated, res.sim_config_updated,
            res.gt_traj_aligned, c;
            x_init=res.x_init, gt_available=[n < k0 for n in 1:N],
            ref_frame=FRAME, feature_type=FEATURE_TYPE, diagnostics=d)
        test = d.k .>= k0
        nees = H.corrector_nees_series(d, res.gt_traj_aligned)
        nyaw = H.nees_yaw_series(d, res.gt_traj_aligned)
        σ = io["prediction"].data_std
        push!(nees_df, (ds, id, name,
            H.consistency_ratio(nees.pos[test], nees.lower, nees.upper),
            H.consistency_ratio(nyaw.yaw[test], nyaw.lower, nyaw.upper),
            mean(nees.pos[test]), mean(nyaw.yaw[test]),
            mean(σ[1:2, :]), mean(σ[4, :])))
    end
    nees_path = stamped(DATA_SECTION, "nees_r$(nees_train_ratio)_$(tag)"; ext="csv")
    CSV.write(nees_path, nees_df)
    @info "Saved consistency table: $nees_path"
else
    nees_df = CSV.read(results_path(DATA_SECTION, nees_csv), DataFrame)
end

# In-envelope ratio: 0.95 is consistent. Mean NEES: 3 (pos) / 1 (yaw) is consistent.
@info "Test-half consistency at train_ratio=$(nees_train_ratio)" combine(
    groupby(nees_df, [:dataset_name, :estimator]),
    [:nees_pos_ratio, :nees_yaw_ratio, :mean_nees_pos, :mean_nees_yaw, :pred_std_pos, :pred_std_yaw] .=> median)
