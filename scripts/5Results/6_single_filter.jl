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
### NOTE ON SCOPE: this script runs ONE trial. The 14-trial evidence quoted in
### notes/002 (paired win count 10/14, mean delta 0.283 m) is not reproduced by
### anything in this repository -- the script that produced it no longer exists.
### Panel (c) is a strong causal argument on this trial; the generality claim
### needs the multi-trial loop restored.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, Statistics, LinearAlgebra, Printf

## 1. HSGP hyperparameters
m = 200
hsgp_p_key = 42
params, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 2. Track
data_key = "DCSC"
trial_id = 5

ins_traj_aligned, gt_traj, zupt, segs, inertial, simdata =
    HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir(data_key), trial_id)

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
nees_runs = OrderedDict{String,NamedTuple}()    # NEES consistency figure

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
    nees_runs[name] = nees
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
const SECTION = "6_SingleFilter"
fig_title = "cov_update=true starves the ZUPT position correction  ($data_key trial $trial_id, test half)"

results_figure() do
    HybridZuptInsJl.plot_zupt_starvation(
        zupt_runs; poserr=pos_errors,
        save_path=stamped(SECTION, "zupt_starvation_$(data_key)$(trial_id)"),
        title=fig_title)
end

## 5. NEES consistency figure.
# The NEES numbers and consistency ratios were previously printed to stdout and
# appeared in no figure, even though plot_nees_comparison exists and takes
# exactly the NamedTuples already built above. NEES is the diagnostic that
# discriminates cleanly here (1597 vs 3.2 in notes/002), so it belongs in the
# chapter rather than in a terminal scrollback.
results_figure() do
    HybridZuptInsJl.plot_nees_comparison(
        nees_runs; block=:pos,
        title="Position NEES, $data_key trial $trial_id (test half)",
        save_path=stamped(SECTION, "nees_pos_$(data_key)$(trial_id)"))
end
