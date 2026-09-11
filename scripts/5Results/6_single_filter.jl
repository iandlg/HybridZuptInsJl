### Why `cov_update=true` costs position accuracy in the single filter -- and why
### the mocap updates in the train half do not cost the same thing.
###
### Three panels, over the FULL run with the train/test boundary marked:
###   (a) the position covariance collapse            -- the cause
###   (b) the ZUPT -> position gain being throttled   -- the mechanism
###   (c) position error, incl. a counterfactual run  -- the cost, and the proof
###
### In the test half, panel (c) carries the argument. It keeps the collapsed P
### everywhere *except* the ZUPT gain (`p_split=:downstream_only,
### zupt_gain_source=:P_alt`); if that curve lands on `cov_update=false`, the
### ZUPT gain is the whole mechanism and (b) is causal rather than merely
### correlated.
###
### The train half answers the question that argument raises: the mocap update
### (`HybridZuptIns.jl` ~L250-270) is *also* an absolute 4-dof update on P, and a
### tighter one than the GP ever supplies (R_gt = sigma_groundtruth^2 = 1e-4 m²,
### InsConfig.jl:101), so it should starve the ZUPT gain at least as hard. If it
### does, and costs nothing, then the shrink itself is not the damage -- what
### matters is whether anything replaces the absolute-position channel it closes.
### Mocap does (it observes absolute position); the GP does not (it predicts a
### stride *increment*, whose covariance is ΔP, not P).
###
### Two consequences of the code that the figure has to be read with:
###   * `cov_update`, `p_split` and `correct` all gate the GP branch only, so the
###     configurations coincide left of the boundary by construction -- §4 below
###     asserts that as a number rather than leaving it hidden under overlapping
###     lines;
###   * P_alt takes the mocap update too, so the fixed-gain counterfactual can
###     only differ from `cov_update=true` in the test half.
###
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
data_key = "ANG2"
trial_id = 14

ins_traj_aligned, gt_traj, zupt, segs, inertial, simdata =
    HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir(data_key), trial_id)

x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
)

# Ground truth is withheld after the cutoff: the mocap update runs over `train_ks`
# and the GP correction over `test_ks`. Both phases are now reported and plotted,
# because the comparison between them is the point.
train_ratio = 0.5
N = length(inertial)
n_train_cutoff = floor(Int, train_ratio * N)
k0 = n_train_cutoff + 1
train_ks = 1:n_train_cutoff
test_ks = k0:N
full_ks = 1:N

## 3. Run the configurations
# Order is draw order: the uncorrected baseline goes first so it sits at the back
# of every panel, behind the configurations the figure is actually comparing.
const BASELINE = "ZUPT-aided INS"
const FIXED_GAIN = "cov_update=true + fixed ZUPT gain"

configs = OrderedDict{String,NamedTuple}(
    # no GP correction at all: the notes/002 baseline row
    BASELINE => (correct=false,),
    "cov_update=true" => (cov_update=true,),
    "cov_update=false" => (cov_update=false,),
    # collapsed P everywhere except the ZUPT gain
    FIXED_GAIN =>
        (cov_update=true, p_split=:downstream_only, zupt_gain_source=:P_alt),
)

# Pin each configuration to its draw-order colour from FILTER_CONFIG_COLORS. The
# palette is positional, but the panels hold different subsets of the runs, so
# only a name => colour map keeps one configuration the same colour throughout.
config_colors = Dict(name => HybridZuptInsJl.FILTER_CONFIG_COLORS[i]
                     for (i, name) in enumerate(keys(configs)))

# (a)/(b) compare the two real configurations; the fixed-gain counterfactual
# belongs in (c), where its claim is made, and the baseline only in the table.
in_gain_panels(name) = name in ("cov_update=true", "cov_update=false")

zupt_runs = OrderedDict{String,NamedTuple}()   # panels (a) and (b), full run
zupt_train = OrderedDict{String,NamedTuple}()  # train-half window, for §4
pos_errors = OrderedDict{String,Any}()         # panel (c)
nees_runs = OrderedDict{String,NamedTuple}()   # NEES consistency figure

# Mean NEES is reported alongside the median because the first stride of the run
# carries a ~1e7 spike (P is initialised at sigma_initial_pos^2 = 1e-10 m² while
# the aligned INS start position is not exactly the mocap one), which dominates
# the train-half mean. The median and the in-95% ratio are the robust readings.
@printf("%-46s %-6s %9s %9s %11s %10s %7s %11s %11s %7s\n",
    "config", "phase", "RMSE[m]", "yaw[rad]", "mean NEES", "med NEES",
    "in 95%", "tr(P_pp)", "||K_pos||", "n_zupt")

for (name, kw) in configs
    _, _, _, _, _, _, _, _, _, diag, quat, x, P = HybridZuptInsJl.hybrid_zupt_aided_ins(
        inertial, simdata, gt_traj, params;
        gt_available=[n <= n_train_cutoff for n in 1:N], x_init=x_init,
        feature_type=FEATURE_TYPE, ref_frame=FRAME, kw...)

    nees = HybridZuptInsJl.nees_series(x, P, quat, gt_traj; ks=full_ks)
    nees_runs[name] = nees

    in_gain_panels(name) &&
        (zupt_runs[name] = HybridZuptInsJl.zupt_gain_series(diag; from_k=1))
    zupt_train[name] =
        HybridZuptInsJl.zupt_gain_series(diag; from_k=1, to_k=n_train_cutoff)

    for (phase, ks, from_k, to_k) in
        (("train", train_ks, 1, n_train_cutoff), ("test", test_ks, k0, N))

        rmse = HybridZuptInsJl.rmse_summary(x, quat, gt_traj; ks=ks)
        nees_phase = view(nees.pos, ks)          # nees was computed over 1:N
        zg = HybridZuptInsJl.zupt_gain_series(diag; from_k=from_k, to_k=to_k)

        @printf("%-46s %-6s %9.4f %9.4f %11.2f %10.2f %6.1f%% %11.3e %11.3e %7d\n",
            phase == "train" ? name : "", phase,
            rmse.pos, rmse.yaw, mean(nees_phase), median(nees_phase),
            100 * HybridZuptInsJl.consistency_ratio(nees_phase, nees.lower, nees.upper),
            zg.mean_P_pos, zg.mean_K_pos, zg.n)
    end

    pos_errors[name] =
        (collect(full_ks), [norm(x[1:3, k] .- gt_traj.pos[:, k]) for k in full_ks])
end

## 4. The train half is identical across configurations, by construction.
# `cov_update`, `p_split` and `correct` all gate the GP branch only, so nothing
# they change can reach the train half. In panels (a)/(b) that shows up as
# perfectly overlapping lines, i.e. as nothing; state it as a number instead.
# Exactly 0.0 is expected, with one exception: `zupt_gain_source=:P_alt` takes
# the ZUPT gain from the parallel covariance in *both* halves, and P_alt is
# maintained by a mathematically equivalent but separately evaluated sequence, so
# it agrees to round-off (~1e-5 relative) rather than bitwise. Anything larger
# means a knob is reaching the mocap branch and this figure cannot be read the
# way its caption says.
ref_name, ref = first(zupt_train)
println("\nTrain-half identity check (reference: $ref_name)")
for (name, r) in zupt_train
    name == ref_name && continue
    if length(r.k) != length(ref.k)
        @printf("  %-46s ZUPT epoch counts differ: %d vs %d\n",
            name, length(r.k), length(ref.k))
    else
        @printf("  %-46s max|ΔK_pos| = %.3e   max|ΔP_pos| = %.3e   (expect 0)\n",
            name, maximum(abs, r.K_pos .- ref.K_pos),
            maximum(abs, r.P_pos .- ref.P_pos))
    end
end

## 5. Figure
const SECTION = "6_SingleFilter"
run_label = "$data_key trial $trial_id, full run — train: mocap updates, test: GP correction"

results_figure() do
    HybridZuptInsJl.plot_zupt_starvation(
        zupt_runs; poserr=pos_errors, split_k=k0,
        colors=config_colors, dashed=[FIXED_GAIN],
        save_path=stamped(SECTION, "zupt_starvation_$(data_key)$(trial_id)"),
        title="Both absolute updates starve the ZUPT position gain  ($run_label)")
end

## 6. NEES consistency figure.
# NEES is what separates the two shrinks: the mocap update shrinks P and the
# state error shrinks with it (earned), while the GP shrink is not matched by any
# reduction in true error (mean NEES 1597 vs 3.2 in notes/002). Log axis because
# the two phases live three decades apart.
results_figure() do
    HybridZuptInsJl.plot_nees_comparison(
        nees_runs; block=:pos, yscale=log10, split_k=k0,
        colors=config_colors, dashed=[FIXED_GAIN],
        title="Position NEES, $run_label",
        save_path=stamped(SECTION, "nees_pos_$(data_key)$(trial_id)"))
end
