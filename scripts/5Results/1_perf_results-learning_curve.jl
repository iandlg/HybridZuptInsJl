### How much online mocap does the correction need before it helps on strides it has
### never seen?
###
### This is 1_perf_results.jl's question with its confound removed. There the x axis is
### `train_ratio`, which also decides where the evaluation window starts
### (`run_online_correction_sweep` scores `corr_traj[floor(r*N_s):end]`), so more
### training also buys a shorter, later, easier test — and a ZUPT-INS drifts with
### open-loop distance, so the curve falls whether or not anything was learned. Its
### columns are therefore not comparable with each other; the paired contrast inside one
### column still is. See notes/011.
###
### Here the test window is a FIXED number of strides at the end of the walk and the
### budget is the window of ground-truth strides immediately before it:
###
###   stride:  1 ............ ks-b .... ks | ks+1 ...... N
###   mocap:   .  no mocap  . [==== b ====] |   none (test)
###   score:                                 [=== n_test ==]
###
### V4 only, and that is not a packaging choice: V2 could separate the two roles of
### ground truth (`gp_train_available` fed the GP, `gt_available` anchored the state) so
### the prefix could stay anchored while only the training budget moved. V4 cannot — `β`
### lives in the corrector's error state and the mocap pose update is what learns it
### (JointStrideEstimators.jl) — so the budget IS the mocap, and the prefix before the
### window runs open loop. What that costs: the pose entering the test window is pinned
### by the last fix whatever the budget is, but the covariance is not. Hence the
### "ZUPT only" baseline is re-run at every budget and the pairing is within a budget;
### `learning_curve_contrast` prints how much that baseline actually moved.
###
### What it does NOT answer: what happens when ground truth is lost mid-walk and the
### walk continues for a variable distance. That is 1_perf_results.jl's.
###
### One run is one `estimator_kwargs` setting, named in the output stem. Two settings in
### one figure is 8_noise_split_ablation.jl's job, on the train_ratio axis.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics, Printf
import CSV

const SECTION = "1_Performance/LearningCurve"
const DATA_SECTION = "$(SECTION)/data"

## 1. Dataset / trials / budgets — the knobs.
# `DATA_KEY` in the environment overrides the default, which is how one unattended run
# covers both datasets without editing the file; a bare REPL include behaves as before.
data_key = get(ENV, "DATA_KEY", "DCSC")
# Every trial of the dataset, as in 1_perf_results.jl. Replace with a literal list to
# subset — but keep it a list from `trial_ids`' dataset (the lists are not interchangeable).
ids = trial_ids(data_key)

# DCSC's ten walks are 90-224 strides, i.e. 50-184 strides of prefix once a 40-stride
# test window is taken off the end, which is the range where "how many footfalls does it
# need" is worth asking. Eight of ANG2's eleven walks are ~30 strides long, so there the
# test window and the budgets both have to shrink; the axis is much shorter.
N_TEST_STRIDES = Dict("DCSC" => 40, "ANG2" => 10)[data_key]
BUDGETS = Dict("DCSC" => [20, 30, 40, 50], "ANG2" => [3, 8, 16])[data_key]

# Constructor keywords for the correctors. `noise_mode` is :split, :process_only
# (σ_n as w) or :online_total (√γ₀ as w) — see StrideNoise in JointStrideEstimators.jl
# and the arms of 8_noise_split_ablation.jl.
estimator_kwargs = (noise_mode=:process_only,)

# Set to a scores CSV under out/Results/1_Performance/LearningCurve/data/ to re-plot a
# finished sweep instead of paying for it again.
# DCSC : learning_curve_V4_DCSC_key42_ntest10_process_only_2026-09-24T10:17:18.182.csv
# ANG2 : learning_curve_V4_ANG2_key42_ntest10_process_only_2026-09-23T12:22:07.749.csv
results_csv = "learning_curve_V4_DCSC_key42_ntest60_process_only_2026-09-23T17:48:50.574.csv"


## 2. Filter and correctors
filter_tag = "V4"
# `noise_mode` reaches the corrector through the constructor, and every estimator
# swallows unknown keywords via `kwargs...`. Under V2/V3 it would be swallowed silently
# and the setting above would be a no-op.
@assert CORRECTION_FILTERS[filter_tag] === HybridZuptInsJl.hybrid_zupt_aided_insv4 "\
    this script is V4 only; filter_tag=$filter_tag has no `noise_mode`."

const BASE_ESTIMATOR = "ZUPT only"
estimators = OrderedDict(
    BASE_ESTIMATOR => HybridZuptInsJl.BaseEstimator,
    "Static" => CORRECTORS[filter_tag].static,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
)
@assert estimators["Static"] === CORRECTORS[filter_tag].static
@assert estimators["HSGP"] === CORRECTORS[filter_tag].hsgp

## 3. Hyperparameters. Key 42 is the project default, for DCSC as well as ANG2.
m = 200
hsgp_p_key = 42
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

output_channels = [:pos_1, :pos_2, :yaw]

score_cols = [:dataset_name, :dataset_order, :trial_id, :train_strides, :train_strides_order,
    :estimator, :estimator_order, :n_strides, :k_split, :n_test_strides, :test_distance_m,
    :rmse, :rmse_rate, :rmse_yaw, :final_pos_err]

const CSV_PREFIX = "learning_curve"

## 4. Run the sweep and save the scores, or read a finished run back
if isnothing(results_csv)
    aligned = HybridZuptInsJl.collect_aligned_trajectories(
        OrderedDict{String,Tuple{String,Vector{Int}}}(data_key => (data_dir(data_key), ids)))

    results_df = HybridZuptInsJl.run_online_learning_curve(
        aligned, FRAME, FEATURE_TYPE, hsgp_p, BUDGETS, estimators, output_channels;
        n_test_strides=N_TEST_STRIDES,
        estimator_alloc=300,
        estimator_kwargs=estimator_kwargs,
        correction_filter=CORRECTION_FILTERS[filter_tag],
    )

    # The setting is part of a run's identity: two `noise_mode` values differ only by
    # timestamp otherwise, and the re-plot branch picks a file by name.
    run_stem = "$(filter_tag)_$(data_key)_key$(hsgp_p_key)_ntest$(N_TEST_STRIDES)_" *
        "$(estimator_kwargs.noise_mode)_$(Dates.now())"
    csv_path = results_path(DATA_SECTION, "$(CSV_PREFIX)_$(run_stem).csv")
    CSV.write(csv_path, results_df[:, score_cols])
    @info "Saved scores table: $csv_path" nrow(results_df)
else
    csv_path = results_path(DATA_SECTION, results_csv)
    results_df = CSV.read(csv_path, DataFrame)
    run_stem = chopprefix(file_stem(csv_path), "$(CSV_PREFIX)_")
    @info "Loaded scores table: $csv_path" nrow(results_df)
end

const DATASET = first(unique(results_df.dataset_name))

## 5. What the sweep actually covered.
# The evaluation window is the claim this figure rests on, so print it rather than
# trusting it: one window per trial, the same for every budget and estimator.
windows = combine(groupby(results_df, :trial_id),
    :n_strides => first => :strides,
    :k_split => first => :split,
    :n_test_strides => first => :test_strides,
    :test_distance_m => first => :test_m,
    :n_test_strides => (v -> maximum(v) - minimum(v)) => :window_spread)
@info "Evaluation windows (window_spread must be 0 everywhere)"
show(stdout, windows; allrows=true)
println()

## 6. Paired figures: per-trial change against that trial's own no-correction baseline
# at the same budget.

# Trials that reach every budget. The wide end of the axis drops the short walks, so the
# all-trials medians change the trial mix along with the budget; these are the rows to
# quote a trend from.
budget_levels = sort(unique(results_df.train_strides))
full_trials = [t for t in sort(unique(results_df.trial_id))
                     if length(unique(results_df[results_df.trial_id .== t, :train_strides])) == length(budget_levels)]
@info "Trials reaching every budget: $full_trials of $(length(unique(results_df.trial_id)))"

"""Median change per budget, and how many of the trials present at EVERY budget improve
at every step of it. The medians are taken over a different trial at each budget, so a
falling median is not evidence that any one walk got better with more mocap — these two
counts are. `diff(v) .< 0` is strict, so a curve that flattens is not monotone; the
endpoint count beside it is the weaker question of whether the widest budget beat the
narrowest at all, which a walk that improves early and then wanders can still pass."""
function print_change(paired::DataFrame, metric::Symbol, label::AbstractString)
    level_order = Dict(r.train_strides => r.train_strides_order for r in eachrow(paired))
    levels = sort(unique(paired.train_strides), by=b -> level_order[b])

    println("\n── $metric: change against $BASE_ESTIMATOR ($label) ──")
    for est in unique(sort(paired, :estimator_order).estimator)
        esub = paired[paired.estimator .== est, :]

        for b in levels
            v = esub[esub.train_strides .== b, :rel_change_pct]
            isempty(v) && continue
            @printf("%-8s %4d strides: better on %2d/%2d trials, median %+7.1f%%\n",
                est, b, count(<(0), v), length(v), median(v))
        end

        # One curve per trial that carries every budget; a trial missing one is not
        # monotone or non-monotone, it is absent, and counting it either way would be a lie.
        curves = Vector{Float64}[]
        for t in unique(esub.trial_id)
            tsub = sort(esub[esub.trial_id .== t, :], :train_strides_order)
            nrow(tsub) == length(levels) || continue
            push!(curves, tsub.rel_change_pct)
        end
        @printf("%-8s %13s monotone on %2d/%2d trials present at every budget\n",
            est, "", count(v -> all(diff(v) .< 0), curves), length(curves))
        @printf("%-8s %13s better at %d than at %d strides on %2d/%2d of them\n",
            est, "", last(levels), first(levels),
            count(v -> last(v) < first(v), curves), length(curves))
    end
end

"""LaTeX table of the median change against the baseline: one row per corrector, one
column per training budget."""
function print_latex_median_table(paired::DataFrame, metric::Symbol)
    levels = sort(unique(paired.train_strides))
    println("\n% $metric: median change against $BASE_ESTIMATOR (%)")
    println("\\begin{tabular}{l", "r"^length(levels), "}")
    println("\\toprule")
    println(" & ", join(["\$n_\\text{train} = $b\$" for b in levels], " & "), " \\\\")
    println("\\midrule")
    for est in unique(sort(paired, :estimator_order).estimator)
        cells = [@sprintf("%+.1f\\%%", median(paired[(paired.estimator .== est) .& (paired.train_strides .== b), :rel_change_pct]))
                 for b in levels]
        println(est, " & ", join(cells, " & "), " \\\\")
    end
    println("\\bottomrule\n\\end{tabular}")
end

for metric in (:rmse, :rmse_yaw)
    paired = HybridZuptInsJl.learning_curve_contrast(
        results_df; metric=metric, reference_estimator=BASE_ESTIMATOR)
    print_latex_median_table(paired, metric)

    base = results_df[results_df.estimator .== BASE_ESTIMATOR, :]
    for b in budget_levels
        v = base[base.train_strides .== b, metric]
        @printf("%s %s %4d strides: median %.3f over %d trials\n", BASE_ESTIMATOR, metric, b, median(v), length(v))
    end

    print_change(paired, metric, "all trials")
    print_change(paired[in.(paired.trial_id, Ref(full_trials)), :], metric,
        "$(length(full_trials)) trials at every budget")

    results_figure() do
        HybridZuptInsJl.plot_learning_curve_relative_change(
            paired, DATASET;
            metric=metric,
            show_outliers=true,
            show_points=true,
            save_path=results_path(SECTION, "$(CSV_PREFIX)_$(metric)_$(run_stem).pdf"),
        )
    end
end
