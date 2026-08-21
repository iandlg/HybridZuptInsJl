### Why `cov_update=true` costs position accuracy in the single filter.
###
### Produces the three-panel figure behind notes/002-why-cov-update-false-wins.md:
###   (a) the position covariance collapse            -- the cause
###   (b) the ZUPT -> position gain being throttled   -- the mechanism
###   (c) a counterfactual run                        -- the causal proof
###
### Panel (c) carries the argument. It keeps the collapsed P everywhere *except*
### the ZUPT gain (`p_split=:downstream_only, zupt_gain_source=:P_alt`); if that
### curve lands on `cov_update=false`, the ZUPT gain is the whole mechanism and
### (b) is causal rather than merely correlated.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, Statistics, LinearAlgebra, Printf, GLMakie

## 1. HSGP hyperparameters
m = 200
hsgp_p_key = 42
hsgp_p_path = Dict{Int,String}(
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no output norm; DCSC
    44 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-19T12:56:03.443.json", # same; ANG2 higher lower noise bound
)[hsgp_p_key]

params, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
params = HybridZuptInsJl.basecopy(params; new_m=m)
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

## 2. Track
data_key = "DCSC"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]
trial_id = 5

ins_traj_aligned, gt_traj, zupt, segs, inertial, simdata =
    HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir, trial_id)

x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
)

# The correction is only active where ground truth is withheld, so every number
# and every panel below is restricted to the test half.
train_ratio = 0.5
N = length(inertial)
n_train_cutoff = floor(Int, train_ratio * N)
gta = [n <= n_train_cutoff for n in 1:N]
k0 = n_train_cutoff + 1
test_ks = k0:N

## 3. Run the three configurations
configs = OrderedDict{String,NamedTuple}(
    "cov_update=true" => (cov_update=true,),
    "cov_update=false" => (cov_update=false,),
    # collapsed P everywhere except the ZUPT gain
    "counterfactual: collapsed P, healthy ZUPT gain" =>
        (cov_update=true, p_split=:downstream_only, zupt_gain_source=:P_alt),
)

zupt_runs = OrderedDict{String,NamedTuple}()   # panels (a) and (b)
pos_errors = OrderedDict{String,Any}()          # panel (c)

for (name, kw) in configs
    _, _, _, _, _, _, _, _, _, diag, quat, x, P = HybridZuptInsJl.hybrid_zupt_aided_ins(
        inertial, simdata, gt_traj, params;
        gt_available=gta, x_init=x_init,
        feature_type=FEATURE_TYPE, ref_frame=FRAME, kw...)

    rmse = HybridZuptInsJl.rmse_summary(x, quat, gt_traj; ks=test_ks)
    nees = HybridZuptInsJl.nees_series(x, P, quat, gt_traj; ks=test_ks)

    # (a)/(b) compare the two real configurations; the counterfactual only needs
    # to appear in (c), where the claim is made.
    if !occursin("counterfactual", name)
        zupt_runs[name] = HybridZuptInsJl.zupt_gain_series(diag; from_k=k0)
    end
    pos_errors["$name  (RMSE $(round(rmse.pos, digits=3)) m)"] =
        (collect(test_ks), [norm(x[1:3, k] .- gt_traj.pos[:, k]) for k in test_ks])

    @printf("%-48s RMSE pos = %.4f m   yaw = %.4f rad   mean NEES pos = %10.2f   in 95%% = %5.1f%%\n",
        name, rmse.pos, rmse.yaw, mean(nees.pos),
        100 * HybridZuptInsJl.consistency_ratio(nees.pos, nees.lower, nees.upper))
end

for (name, r) in zupt_runs
    @printf("%-20s mean tr(P[1:3,1:3]) = %.4e m²   mean ||K[1:3,:]|| = %.4e   (%d ZUPT epochs)\n",
        name, r.mean_P_pos, r.mean_K_pos, r.n)
end

## 4. Figure
fig = HybridZuptInsJl.plot_zupt_starvation(
    zupt_runs; poserr=pos_errors,
    title="cov_update=true starves the ZUPT position correction  ($data_key trial $trial_id, test half)")
display(fig)

## 5. Save
import CairoMakie, Dates
base_dir = "out/Results/6_SingleFilter"
mkpath(base_dir)

time = string(Dates.now())
path = joinpath(base_dir, "$(time)_zupt_starvation_$(data_key)$(trial_id).svg")

CairoMakie.with_theme(CairoMakie.theme_ggplot2()) do
    HybridZuptInsJl.plot_zupt_starvation(
        zupt_runs; poserr=pos_errors, save_path=path,
        title="cov_update=true starves the ZUPT position correction  ($data_key trial $trial_id, test half)")
end
println("saved -> ", path)
