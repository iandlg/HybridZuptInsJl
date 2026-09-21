### Does splitting the stride residual actually buy anything?
###
### V4 splits the stride residual into a per-stride process noise `σ_w` and a
### per-footfall jitter `σ_j` (`StrideNoise`, notes/015 §2.5; notes/017 shows it is
### the textbook local level model). This script measures that split against the
### model it replaced -- the GP likelihood noise `σ_n` taken as process noise, with
### nothing on the mocap fix -- paired on the same trial at the same train_ratio.
###
### It exists because the evidence in 015 §2.5 is a hand-assembled table whose
### columns were run on DIFFERENT trial sets ("across the table read the direction,
### not the digits"), reproduced by no script. This is that table, run properly.
###
###   arm "split"     σ_w = √(γ₀−2σ_j²) online, σ_j = √(−γ₁) floored at Σ_gt
###   arm "σ_n as w"  σ_w = σ_n fixed,          σ_j = 0, so the fix noise is Σ_gt
###   arm "√γ₀ as w"  σ_w = √γ₀ online,         σ_j = 0, so the fix noise is Σ_gt
###
### Arm B disables both halves of `StrideNoise` (no split, no online estimate),
### because `σ_n` is by construction one fixed unsplit number. That confounds the two,
### which is why arm C exists: it keeps the online estimate and drops only the split, so
### A−C is the split alone and C−B is the online re-sizing alone (notes/018 §5).
###
### ARM B IS NOT A REVERT TO 5e213d6. That commit also put the process noise on the
### *corrected* channels only; extending it to all four rode along in the split's
### commit (47ee8e5) but is an independent fix (015 §2.3, the uncorrected-z NEES
### defect). Arm B keeps the all-channel process noise, so the split is the only
### factor that moves. Numbers here will not match a `git checkout 5e213d6` run.
###
### What 015 §2.5 predicts, so that this run can falsify it:
###   * yaw NEES ≪ 1 under "σ_n as w" -- the yaw covariance grows as a random walk
###     while the true jitter telescopes, so the filter is under-confident;
###   * the RMSE cost concentrated on yaw, worst at low train_ratio.
### If the run disagrees, that is a finding. 015's table mixed trial sets; this one
### does not. Record the disagreement rather than tuning until it matches.
###
### Two parts:
###   1. paired RMSE / RMSE_yaw against ZUPT only, every trial x train_ratio;
###   2. test-half consistency (position and yaw NEES) per trial, plus the (σ_w, σ_j)
###      arm A actually identified against the σ_n arm B was handed.
### Part 2 runs every trial, so unlike 7_decoupled_consistency.jl its NEES claim
### comes with a distribution rather than one walk.
###
### Clean data only: with `sigma_groundtruth` correct, nothing here is confounded by
### the filter discovering mocap noise it was not told about. That regime is
### 5_noise_robustness.jl's, and notes/016 §7 is the warning about reading it.
###
### Run twice, editing `data_key` only -- the key stays 42 for both datasets.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics, Printf
import CSV

const SECTION = "8_NoiseSplit"
const DATA_SECTION = "$(SECTION)/data"

# 1. Dataset / trials.
data_key = "ANG2"       # or "DCSC"
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    data_key => (data_dir(data_key), trial_ids(data_key)),
)

# Set to a scores CSV under out/Results/8_NoiseSplit/data/ to re-plot part 1 without
# paying for the sweep. Part 2 is 30 s and always runs fresh, so it is skipped on a
# re-plot rather than half-restored from a file it did not write.
results_csv = nothing

aligned = isnothing(results_csv) ? HybridZuptInsJl.collect_aligned_trajectories(data_dict) : nothing

## 2. Hyperparameters
# Key 42 for BOTH datasets: it is the project default, and the ablation reads the yaw
# channel, which is exactly what a key change moves (notes/014 -- key 47's DCSC yaw
# prior underflows to 0). 5_noise_robustness.jl's key 47 is that script's exception.
m = 200
hsgp_p_key = 42
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 3. Arms
# The single definition of what the arms are. Everything downstream -- estimator
# names, file names, the tables -- indexes this, so an arm cannot be renamed in one
# place and not another.
const ARMS = OrderedDict{String,Symbol}(
    "split" => :split,
    "σ_n as w" => :process_only,
    "√γ₀ as w" => :online_total,
)

# Ordered, indexable arm names. `keys(::OrderedDict)` iterates in order but does not
# support `first`/`last`/`[i]`, and every use below is positional.
const ARM_LABELS = collect(keys(ARMS))

## 4. Filter and correctors
filter_tag = "V4"
# `noise_mode` reaches the corrector through the constructor, and every estimator
# swallows unknown keywords via `kwargs...`. Under V2/V3 it would be swallowed
# silently and the two arms would be the same run twice.
@assert CORRECTION_FILTERS[filter_tag] === HybridZuptInsJl.hybrid_zupt_aided_insv4 "\
    the noise split exists only in V4; filter_tag=$filter_tag has no `noise_mode`."

const BASE_ESTIMATOR = "ZUPT only"
estimators = OrderedDict(
    BASE_ESTIMATOR => HybridZuptInsJl.BaseEstimator,
    "Static" => CORRECTORS[filter_tag].static,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
)

# Two arms of one method are two shades of that method's colour, built from the same
# `ARMS` that names them. `method_color` knows bare method names only, so every arm
# would otherwise land on its one fallback grey and the figure could not be read.
const ARM_COLORS = Dict(
    "$(name) ($(arm))" => shade
    for (name, shades) in (
        n => HybridZuptInsJl.color_shades(HybridZuptInsJl.method_color(n), length(ARMS))
        for n in keys(estimators) if n != BASE_ESTIMATOR)
    for (arm, shade) in zip(ARM_LABELS, shades))

output_channels = [:pos_1, :pos_2, :yaw]
train_ratios = [0.3, 0.5]

score_cols = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :noise_spec_tag, :noise_spec_order, :seed,
    :rmse, :rmse_rate, :rmse_yaw]

const CSV_PREFIX = "split_ablation"

## 5. Part 1 -- one sweep per arm, merged into one paired frame
if isnothing(results_csv)
    arm_frames = DataFrame[]
    base_frames = OrderedDict{String,DataFrame}()

    for (arm_order, (arm_label, mode)) in enumerate(ARMS)
        @info "Sweeping arm: $arm_label (noise_mode=:$mode)"
        df = HybridZuptInsJl.run_online_correction_sweep(
            aligned, FRAME, FEATURE_TYPE, hsgp_p, train_ratios, estimators, output_channels;
            estimator_alloc=300,
            keep_artifacts=false,
            correction_filter=CORRECTION_FILTERS[filter_tag],
            estimator_kwargs=(noise_mode=mode,),
        )

        base_frames[arm_label] = sort(df[df.estimator .== BASE_ESTIMATOR, :],
            [:trial_id, :train_ratio])

        # The corrector rows carry the arm in their name, which is what makes the arms
        # ordinary estimators to `paired_estimator_contrast` and the boxplot below.
        corr = df[df.estimator .!= BASE_ESTIMATOR, :]
        corr.estimator = ["$(e) ($(arm_label))" for e in corr.estimator]
        # ZUPT only = 1, then the correctors interleaved arm-by-arm: Static(A),
        # Static(B), HSGP(A), HSGP(B), so the figure puts the two arms side by side.
        corr.estimator_order = 1 .+ (corr.estimator_order .- 2) .* length(ARMS) .+ arm_order
        push!(arm_frames, corr)
    end

    # The baseline never sees `noise_mode`: `BaseEstimator` has no `StrideNoise`. If the
    # two arms' ZUPT-only rows differ at all, the arm switch escaped the corrector and
    # nothing below is a paired comparison. Cheapest possible check, so it is not optional.
    let a = base_frames[ARM_LABELS[1]]
        for other in ARM_LABELS[2:end]
            b = base_frames[other]
            @assert a.trial_id == b.trial_id && a.train_ratio == b.train_ratio "\
                arms $(ARM_LABELS[1]) and $other did not run the same (trial, train_ratio) cells."
            for col in (:rmse, :rmse_yaw)
                d = maximum(abs.(a[!, col] .- b[!, col]))
                @assert d == 0.0 "$BASE_ESTIMATOR differs between arms \
                    $(ARM_LABELS[1]) and $other by $d on $col -- \
                    `noise_mode` leaked outside the corrector."
            end
        end
        @info "Baseline identical across all $(length(ARMS)) arms on $(nrow(a)) cells."
    end

    # One reference row per cell, from the first arm. Both arms produced it identically,
    # so which one is kept is immaterial -- keeping both would double every pairing.
    results_df = vcat(base_frames[ARM_LABELS[1]], arm_frames...)

    run_stem = "$(filter_tag)_$(data_key)_key$(hsgp_p_key)_$(FRAME)_$(FEATURE_TYPE)_$(Dates.now())"
    csv_path = results_path(DATA_SECTION, "$(CSV_PREFIX)_$(run_stem).csv")
    CSV.write(csv_path, results_df[:, score_cols])
    @info "Saved scores table: $csv_path" nrow(results_df)
else
    csv_path = results_path(DATA_SECTION, results_csv)
    results_df = CSV.read(csv_path, DataFrame)
    run_stem = chopprefix(file_stem(csv_path), "$(CSV_PREFIX)_")
    @info "Loaded scores table: $csv_path" nrow(results_df)
end

const DATASET = results_df[1, :dataset_name]

## 6. Part 1 figures and the table that replaces notes/015 §2.5
paired = OrderedDict{Symbol,DataFrame}()
for metric in (:rmse, :rmse_yaw)
    paired[metric] = HybridZuptInsJl.paired_estimator_contrast(
        results_df; metric=metric, reference_estimator=BASE_ESTIMATOR,
        train_ratios=sort(unique(results_df.train_ratio)))
    results_figure() do
        HybridZuptInsJl.plot_train_ratio_paired_relative_change(
            paired[metric], DATASET;
            metric=metric,
            show_outliers=true,
            show_points=false,
            series_colors=ARM_COLORS,
            save_path=results_path(SECTION, "split_paired_$(metric)_$(run_stem).pdf"),
        )
    end
end

# Median relative change vs ZUPT only, per corrector x arm x train_ratio, for both
# metrics side by side -- the numbers 015 §2.5 tried to tabulate. `n` is the paired
# cell count behind each median, because a median of 11 trials and a median of 5 are
# not the same claim.
println("\n=== Paired median change vs $BASE_ESTIMATOR (%), $DATASET, key $hsgp_p_key ===")
@printf("%-24s %6s %12s %12s %5s\n", "estimator", "tr", "RMSE_p", "RMSE_psi", "n")
for tr in sort(unique(paired[:rmse].train_ratio))
    for est in unique(paired[:rmse].estimator)
        rows = [paired[m][(paired[m].estimator .== est) .& (paired[m].train_ratio .== tr), :]
                for m in (:rmse, :rmse_yaw)]
        isempty(rows[1]) && continue
        @printf("%-24s %6.2f %11.1f%% %11.1f%% %5d\n", est, tr,
            median(rows[1].rel_change_pct), median(rows[2].rel_change_pct), nrow(rows[1]))
    end
end

## 6b. Arm against arm, on the same trial -- the comparison this script exists to make
# NOT derivable from the table above. Those are medians of each arm's change against
# ZUPT only, and the median of a difference is not the difference of medians: on ANG2
# the two views disagree in sign on the position channel. Here each point is one
# trial's split-minus-alternative difference at one train_ratio, so the baseline
# cancels exactly and the win count is over paired trials.
const CORRECTOR_NAMES = [n for n in keys(estimators) if n != BASE_ESTIMATOR]
arm_name(corrector, arm) = "$(corrector) ($(arm))"

println("\n=== \"$(ARM_LABELS[1])\" against each other arm, paired per trial, $DATASET ===")
println("    negative = \"$(ARM_LABELS[1])\" is better; wins = trials it won")
@printf("%-10s %-12s %5s %13s %7s %14s %7s\n",
    "corrector", "vs arm", "tr", "Δ RMSE_p", "wins", "Δ RMSE_psi", "wins")
for corrector in CORRECTOR_NAMES, other in ARM_LABELS[2:end]
    for tr in sort(unique(results_df.train_ratio))
        cell(arm) = select(results_df[(results_df.estimator .== arm_name(corrector, arm)) .&
                                      (results_df.train_ratio .== tr), :],
            :trial_id, :rmse, :rmse_yaw)
        j = innerjoin(cell(ARM_LABELS[1]), cell(other), on=:trial_id, makeunique=true)
        isempty(j) && continue
        Δp = 100 .* (j.rmse .- j.rmse_1) ./ j.rmse_1
        Δy = 100 .* (j.rmse_yaw .- j.rmse_yaw_1) ./ j.rmse_yaw_1
        @printf("%-10s %-12s %5.2f %12.1f%% %7s %13.1f%% %7s\n",
            corrector, other, tr,
            median(Δp), "$(count(<(0), Δp))/$(nrow(j))",
            median(Δy), "$(count(<(0), Δy))/$(nrow(j))")
    end
end

## 7. Part 2 -- test-half consistency, and what each arm believed its noise was
# RMSE cannot show a misspecified covariance, and the claim under test is exactly a
# covariance claim. Run at the low-mocap end, where 015 says the failure shows.
if isnothing(results_csv)
    nees_train_ratio = 0.3
    n_alloc = 300
    nees_rows = DataFrame()

    for (trial_id, res) in aligned[data_key]
        N = length(res.inertial_updated)
        n_cut = floor(Int, nees_train_ratio * N)
        k0 = n_cut + 1

        for (arm_label, mode) in ARMS
            corrector = CORRECTORS[filter_tag].hsgp(n_alloc;
                params=hsgp_p, corrected_channels=output_channels, noise_mode=mode)
            diag = HybridZuptInsJl.CorrectorDiagnostics()

            HybridZuptInsJl.hybrid_zupt_aided_insv4(
                res.inertial_updated, res.sim_config_updated, res.gt_traj_aligned, corrector;
                x_init=res.x_init,
                gt_available=[n <= n_cut for n in 1:N],
                ref_frame=FRAME, feature_type=FEATURE_TYPE,
                diagnostics=diag)

            # Silent overruns and empty phases are the two ways this loop lies.
            n_foot = length(diag)
            n_foot < n_alloc || error("trial $trial_id/$arm_label: $n_foot footfalls \
                                       against a preallocation of $n_alloc -- it overran.")
            test = findall(>=(k0), diag.k)
            (!isempty(test) && any(<(k0), diag.k)) ||
                error("trial $trial_id/$arm_label: train_ratio=$nees_train_ratio leaves \
                       a phase empty at k=$k0.")

            nees = HybridZuptInsJl.corrector_nees_series(diag, res.gt_traj_aligned)
            nyaw = HybridZuptInsJl.nees_yaw_series(diag, res.gt_traj_aligned)
            # Frozen at the end of the train half: `observe_stride_error!` only fires
            # while mocap is available, so this is what the test half ran on.
            σ_w, σ_j = HybridZuptInsJl.noise_split(corrector.noise)

            push!(nees_rows, (
                dataset_name=data_key, trial_id=trial_id, arm=arm_label,
                train_ratio=nees_train_ratio, n_test=length(test),
                nees_pos=median(view(nees.pos, test)),
                in95_pos=HybridZuptInsJl.consistency_ratio(
                    view(nees.pos, test), nees.lower, nees.upper),
                nees_yaw=median(view(nyaw.yaw, test)),
                in95_yaw=HybridZuptInsJl.consistency_ratio(
                    view(nyaw.yaw, test), nyaw.lower, nyaw.upper),
                sigma_w_yaw=σ_w[4], sigma_j_yaw=σ_j[4],
                sigma_w_h=(σ_w[1] + σ_w[2]) / 2, sigma_j_h=(σ_j[1] + σ_j[2]) / 2,
            ); promote=true)
        end
    end

    nees_csv = results_path(DATA_SECTION, "nees_$(run_stem).csv")
    CSV.write(nees_csv, nees_rows)
    @info "Saved consistency table: $nees_csv" nrow(nees_rows)

    # Medians across trials. An arm whose NEES sits far below its dof is
    # under-confident: covariance growing along a path the error does not take.
    #
    # Read the yaw column on NEES, not on in95. Chisq(1)'s lower bound is 9.8e-4, so
    # an arm can be an order of magnitude under-confident and still put ~95% of its
    # footfalls "inside the envelope" -- in95 catches over-confidence, and nothing here
    # is over-confident. The 3-dof position bound (0.216) is loose in the same
    # direction, just less extremely.
    println("\n=== HSGP test-half consistency, $DATASET, tr=$nees_train_ratio ",
        "(pos NEES ideal ≈ 3 dof, yaw ≈ 1 dof, in95 ideal 0.95) ===")
    @printf("%-12s %10s %8s %10s %8s %11s %11s %6s\n",
        "arm", "NEES_pos", "in95", "NEES_yaw", "in95", "σ_w,ψ[rad]", "σ_j,ψ[rad]", "n")
    for arm_label in ARM_LABELS
        g = nees_rows[nees_rows.arm .== arm_label, :]
        @printf("%-12s %10.2f %7.0f%% %10.2f %7.0f%% %11.4f %11.4f %6d\n",
            arm_label, median(g.nees_pos), 100 * median(g.in95_pos),
            median(g.nees_yaw), 100 * median(g.in95_yaw),
            median(g.sigma_w_yaw), median(g.sigma_j_yaw), nrow(g))
    end
else
    @info "Re-plot from CSV: part 2 (consistency) skipped -- it runs fresh, in ~30 s."
end
