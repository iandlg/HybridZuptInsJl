# THESIS SECTION 1 (headline): performance vs how much ground truth is available
# online (train_ratio), for every corrector, across all trials of a dataset.
#
# Despite the "6_" filename this is the *first* results section -- it is the
# figure an examiner reads before any other. Consider renaming to 1_ so the
# directory order matches the chapter order.
#
# Structurally this is the strongest sweep in 5Results: a genuinely swept
# continuous x-axis with per-trial replication at every level. It was also the
# least maintained -- see the notes below.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames
import CSV

# Choose Parameters file
hsgp_p_key = 42
m = 200   # WAS 300 here and 200 in every other script, so the headline figure
          # was not directly comparable with the rest of the chapter.

hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

data_key = "ANG2"
data_dir_path = data_dir(data_key)

ids = HybridZuptInsJl.list_trial_ids(data_dir_path)
train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# WAS: "Static" => HybridZuptInsJl.StaticCorrectorV2(300)
# StaticCorrectorV2 does not exist anywhere in src/ -- this cell could not run
# as written, which is why the committed figure dates from 26 June while the
# rest of the chapter is from August. Names also updated from the old
# Default/Static/Split/Slam vocabulary to the one every other script uses, so
# the same estimator is called the same thing in every figure.
correctors = OrderedDict{String,HybridZuptInsJl.AbstractEstimator}(
    "Base (no correction)" => HybridZuptInsJl.BaseEstimator(300),
    "Decoupled Static" => HybridZuptInsJl.DecoupledStaticEstimator(300),
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator(300; params=hsgp_p),
    "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator(300; params=hsgp_p),
)

dataset = HybridZuptInsJl.collect_dataset(data_dir_path, ids, train_ratios, correctors;
    frame=FRAME, feature_type=FEATURE_TYPE)

df = HybridZuptInsJl.performance_dataframe(dataset)

const SECTION = "1_Performance"
csv_path = stamped(SECTION, "results_$(data_key)_$(FRAME)_$(FEATURE_TYPE)"; ext="csv")
CSV.write(csv_path, df)
@info "Saved results table: $csv_path"

## Plot
# WAS: this cell re-read a hard-coded CSV path from June while stamping the
# output filenames with the data_key/FRAME/FEATURE of whatever the compute cell
# above had set -- so the figure legend could describe a different run than the
# data plotted. It now plots the DataFrame just computed. To re-plot an older
# run, set `df = CSV.read(<path>, DataFrame)` here deliberately.
corrector_names = collect(keys(correctors))

for metric in (:rmse, :rmse_rate), show_outliers in (true, false)
    suffix = show_outliers ? "" : "_nooutliers"
    results_figure() do
        HybridZuptInsJl.plot_corrector_boxplots(
            df, metric;
            show_outliers=show_outliers,
            corrector_names=corrector_names,
            save_path=stamped(SECTION, "$(uppercase(string(metric)))$(suffix)"),
        )
    end
end

## Paired view at the operating point used by the rest of the chapter.
# Same trials at every train_ratio, so per-trial differences are the honest
# summary at n ≈ 16.
results_figure() do
    HybridZuptInsJl.plot_paired_differences(
        df;
        metric=:rmse_rate,
        baseline="Base (no correction)",
        train_ratio=0.5,
        save_path=stamped(SECTION, "paired_vs_baseline_tr0.5"),
    )
end
