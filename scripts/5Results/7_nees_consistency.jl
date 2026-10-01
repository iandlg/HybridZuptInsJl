### Is each corrector's covariance consistent with its actual error, and how does that
### change with how much of the walk has mocap online?
###
### 1_perf_results.jl's train-ratio design (mocap for n <= floor(r*N), open loop after),
### scored on NEES instead of RMSE: per footfall, position (χ²(3)) and yaw (χ²(1)) NEES
### of the corrector's own state and Σ against the clean ground truth. See notes/022.
###
### Knobs from the environment, so an unattended run needs no edit:
###   DATA_KEY=DCSC TRAIN_RATIOS=0.3,0.6 julialauncher --project=. -t 1 scripts/5Results/7_nees_consistency.jl
###   RESULTS_CSV=nees_footfalls_<stem>.csv ...   re-plots a finished run
using StrideGP
include("_common.jl")
using OrderedCollections, DataFrames, Statistics, Printf
import CSV

const SECTION = "7_NeesConsistency"
const DATA_SECTION = "$(SECTION)/data"
const CSV_PREFIX = "nees_footfalls"

## 1. Dataset / train ratios
data_key = get(ENV, "DATA_KEY", "ANG2")
ids = trial_ids(data_key)
train_ratios = parse.(Float64, split(get(ENV, "TRAIN_RATIOS", "0.2,0.3,0.4,0.5,0.6,0.7,0.8"), ","))
# A per-footfall CSV under out/Results/7_NeesConsistency/data/, or `nothing` to run.
results_csv = get(ENV, "RESULTS_CSV", nothing)

## 2. Filter, correctors, hyperparameters
filter_tag = "V4"
estimators = OrderedDict(
    "ZUPT only" => StrideGP.BaseEstimator,
    "Static" => CORRECTORS[filter_tag].static,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
)
m = 200
hsgp_p_key = 42
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)
output_channels = [:pos_1, :pos_2, :yaw]

## 3. Run the sweep and save the per-footfall NEES, or read a finished run back
if isnothing(results_csv)
    aligned = StrideGP.collect_aligned_trajectories(
        OrderedDict{String,Tuple{String,Vector{Int}}}(data_key => (data_dir(data_key), ids)))
    footfalls = StrideGP.run_online_nees_sweep(
        aligned, FRAME, FEATURE_TYPE, hsgp_p, train_ratios, estimators, output_channels;
        estimator_alloc=300,
        correction_filter=CORRECTION_FILTERS[filter_tag])
    run_stem = "$(filter_tag)_key$(hsgp_p_key)_$(data_key)_tr$(join(train_ratios, "-"))_$(Dates.now())"
    csv_path = results_path(DATA_SECTION, "$(CSV_PREFIX)_$(run_stem).csv")
    CSV.write(csv_path, footfalls)
    @info "Saved per-footfall NEES: $csv_path" nrow(footfalls)
else
    csv_path = results_path(DATA_SECTION, results_csv)
    footfalls = CSV.read(csv_path, DataFrame)
    run_stem = chopprefix(file_stem(csv_path), "$(CSV_PREFIX)_")
    @info "Loaded per-footfall NEES: $csv_path" nrow(footfalls)
end

summary = StrideGP.nees_summary(footfalls)
CSV.write(results_path(DATA_SECTION, "nees_summary_$(run_stem).csv"), summary)

## 4. Median over trials of each run's ANEES and fraction inside the 95% envelope.
# Consistent: ANEES ≈ 3 (position) / 1 (yaw), ~95% inside.
for phase in ("train", "test")
    println("\n── $phase phase: median over trials (ANEES / % inside) ──")
    @printf("%-6s %-10s %16s %16s %4s\n", "ratio", "estimator", "pos (dof 3)", "yaw (dof 1)", "n")
    sub = sort(summary[summary.phase .== phase, :], [:train_ratio_order, :estimator_order])
    for g in groupby(sub, [:train_ratio, :estimator], sort=false)
        @printf("%-6.2f %-10s %8.2f / %4.0f%% %8.2f / %4.0f%% %4d\n",
            g.train_ratio[1], g.estimator[1],
            median(g.anees_pos), 100median(g.inside_pos),
            median(g.anees_yaw), 100median(g.inside_yaw), nrow(g))
    end
end

## 5. Figure, named after the CSV it was plotted from
results_figure() do
    StrideGP.plot_nees_train_ratio(summary, first(unique(summary.dataset_name));
        phase="test",
        save_path=results_path(SECTION, "nees_$(run_stem).pdf"))
end
