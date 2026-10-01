### Is the decoupled HSGP corrector's own uncertainty honest?
###
### Two figures over the full run, split at the train/test boundary:
###   (a) position NEES against the chi-square(3) 95% envelope
###   (b) tr(Σ[1:3,1:3]), the position uncertainty the corrector reports
###
### Both come from the SAME matrix -- `DecoupledHsgpEstimator.Σ`, the 6×6
### [pos(1:3); att(4:6)] covariance the corrector carries from footfall to
### footfall -- which is what makes the pair readable as one statement: (b) is
### the shrink, (a) asks whether it was earned. A trace that falls while NEES
### rises is over-confidence; both falling together is a corrector that knows
### what it knows.
###
### The boundary is where the shrink changes hands. Left of it the corrector
### takes absolute mocap updates (`posyaw_measurement_update!`); right of it it
### takes the GP's stride prediction (`learned_measurement_update!`). notes/002
### established for the V1 *single* filter that the first shrink is earned and
### the second is not; nothing had asked the same question of the V2 decoupled
### design, which is the estimator every current §5b result actually uses.
###
### Note this is the corrector's own covariance, not the inner ZUPT-INS `P`.
### The decoupled design never lets the GP touch that one -- that is what
### "decoupled" means -- so it is `6_single_filter.jl` that has the story about
### `P`, and this script has the story about `Σ`.
###
### NOTE ON SCOPE: one trial, one estimator. Read (a) as a property of this walk.
### The dict plumbing below takes more entries unchanged, so adding
### `DecoupledStaticEstimator` or a second trial is a small edit, not a rewrite.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, Statistics, Printf

## 1. HSGP hyperparameters
m = 200
hsgp_p_key = 42
params, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 2. Track
data_key = "ANG2"
trial_id = 14                                   # = TEST_IDS["ANG2"], the held-out walk
output_channels = [:pos_1, :pos_2, :yaw]

ins_traj_aligned, gt_traj, zupt, segs, inertial, simdata =
    HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir(data_key), trial_id)

x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
)

# Ground truth is withheld after the cutoff: the mocap update runs over the first
# `train_ratio` of the walk and the GP correction over the rest.
train_ratio = 0.3
N = length(inertial)
n_train_cutoff = floor(Int, train_ratio * N)
k0 = n_train_cutoff + 1

## 3. Run the estimator, recording the corrector at every footfall
# `n_alloc` is the corrector's preallocation, and the estimators index it
# unguarded -- a walk with more footfalls than this overruns silently rather than
# erroring, so §4 checks it after the run.
n_alloc = 300

estimators = OrderedDict{String,Any}(
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
)

nees_runs = OrderedDict{String,NamedTuple}()
cov_runs = OrderedDict{String,NamedTuple}()
diags = OrderedDict{String,HybridZuptInsJl.CorrectorDiagnostics}()

@printf("%-18s %-6s %9s %9s %11s %10s %8s %12s %7s\n",
    "estimator", "phase", "RMSE[m]", "yaw[rad]", "mean NEES", "med NEES",
    "in 95%", "tr(Σ_pp)", "n_foot")

for (name, factory) in estimators
    corrector = factory(n_alloc; params=params, corrected_channels=output_channels)
    diag = HybridZuptInsJl.CorrectorDiagnostics()

    HybridZuptInsJl.hybrid_zupt_aided_insv2(
        inertial, simdata, gt_traj, corrector;
        x_init=x_init,
        gt_available=[n <= n_train_cutoff for n in 1:N],
        ref_frame=FRAME, feature_type=FEATURE_TYPE,
        diagnostics=diag)

    diags[name] = diag
    nees_runs[name] = HybridZuptInsJl.corrector_nees_series(diag, gt_traj)
    cov_runs[name] = HybridZuptInsJl.pos_cov_trace(diag)

    # The corrector state lives on the footfall axis, so the phase split is a
    # partition of the footfalls by the sample index each one landed on.
    nees = nees_runs[name]
    gt_foot = gt_traj[diag.k]
    x_foot = hcat(diag.pos...)
    quat_foot = hcat(diag.quat...)
    trace = cov_runs[name].trace

    for (phase, idx) in (("train", findall(<(k0), diag.k)),
        ("test", findall(>=(k0), diag.k)))
        isempty(idx) && continue

        # rmse_summary wants the 9-state layout; only rows 1:3 are read for
        # position, and yaw comes from `quat_foot`.
        x9 = zeros(9, length(diag.k))
        x9[1:3, :] = x_foot
        rmse = HybridZuptInsJl.rmse_summary(x9, quat_foot, gt_foot; ks=idx)

        @printf("%-18s %-6s %9.4f %9.4f %11.2f %10.2f %7.1f%% %12.3e %7d\n",
            phase == "train" ? name : "", phase,
            rmse.pos, rmse.yaw,
            mean(view(nees.pos, idx)), median(view(nees.pos, idx)),
            100 * HybridZuptInsJl.consistency_ratio(
                view(nees.pos, idx), nees.lower, nees.upper),
            mean(view(trace, idx)), length(idx))
    end
end

## 4. Checks the figures cannot make for you
# The preallocation overrun is silent, and a cutoff that lands outside the
# footfall range would give one empty phase and a figure with no boundary in it.
for (name, d) in diags
    n_foot = length(d)
    n_foot < n_alloc || error("$name: $n_foot footfalls against a preallocation of \
                               $n_alloc -- raise `n_alloc`, the run overran.")
    n_train = count(<(k0), d.k)
    n_test = n_foot - n_train
    (n_train > 0 && n_test > 0) || error("$name: train_ratio=$train_ratio puts \
                                          $n_train/$n_test footfalls either side of k=$k0.")
    @printf("\n%s: %d footfalls, %d train / %d test, split at k=%d\n",
        name, n_foot, n_train, n_test, k0)
end

## 5. Figures
const SECTION = "7_DecoupledConsistency"
run_label = "$data_key trial $trial_id, train_ratio=$train_ratio — train: mocap updates, test: GP correction"

# Log axis: the mocap-anchored half and the GP-corrected half live decades apart,
# so on a linear axis the first is flattened onto the x-axis by the second.
results_figure() do
    HybridZuptInsJl.plot_nees_comparison(
        nees_runs; block=:pos, yscale=log10, split_k=k0,
        legend_position=:lt,   # the NEES curve climbs into the default top-right corner
        title="Corrector position NEES, $run_label",
        save_path=stamped(SECTION, "nees_pos_$(data_key)$(trial_id)"))
end

results_figure() do
    HybridZuptInsJl.plot_position_covariance(
        cov_runs; split_k=k0,
        title="Corrector position uncertainty, $run_label",
        save_path=stamped(SECTION, "pos_cov_trace_$(data_key)$(trial_id)"))
end
