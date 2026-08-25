# Section 3 (yaw channel): can a constant yaw correction replace the HSGP one?
#
# Only the yaw channel is corrected here (output_channels = [:yaw]).
#
# NOTE ON THE METRIC. This experiment used to be scored with :rmse_rate alone,
# which is horizontal *position* error -- the claim was about yaw and the
# measurement was x/y. Both are now produced: :rmse_yaw is the direct evidence
# for "does the yaw correction work", :rmse_rate is the downstream consequence
# for position. Report the yaw figure as the primary one.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames

# 1. Datasets / trials
data_dict = OrderedDict{String,Tuple{String,Vector{Int}}}(
    "Angermann" => (data_dir("ANG2"), trial_ids("ANG2")),
    "TuDCSC" => (data_dir("DCSC"), trial_ids("DCSC")),
)

# 2. Align INS / GT trajectories for every trial
aligned = HybridZuptInsJl.collect_aligned_trajectories(data_dict)

## 3. Hyperparameters
m = 200
hsgp_p_key = 42
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 4. Correction methods to compare
estimators = OrderedDict(
    "Base (no correction)" => HybridZuptInsJl.BaseEstimator,
    "Decoupled Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    # "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "Decoupled HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    # "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)

output_channels = [:yaw]
train_ratios = [0.5]

## 5. Run the sweep
results_df = HybridZuptInsJl.run_online_correction_sweep(
    aligned,
    FRAME,
    FEATURE_TYPE,
    hsgp_p,
    train_ratios,
    estimators,
    output_channels,
)

## 6. Plot
const SECTION = "3_yaw_channel/Const_v_Hsgp_yaw_correction"

# Primary: the channel actually being corrected.
results_figure() do
    HybridZuptInsJl.boxplot_dataset_comparison(
        results_df;
        metric=:rmse_yaw,
        train_ratio=0.5,
        save_path=stamped(SECTION, "yaw_only_correction_YAW"),
    )
end

# Secondary: what it costs downstream in position.
results_figure() do
    HybridZuptInsJl.boxplot_dataset_comparison(
        results_df;
        metric=:rmse_rate,
        train_ratio=0.5,
        save_path=stamped(SECTION, "yaw_only_correction_POS"),
    )
end

# Paired view: five estimators x two datasets from n≈10 each is a lot of boxes
# to compare by eye, and they are all the same trials.
for metric in (:rmse_yaw, :rmse)
    results_figure() do
        HybridZuptInsJl.plot_paired_relative_change(
            results_df;
            metric=metric,
            baseline="Base (no correction)",
            group=:dataset_name,
            train_ratio=0.5,
            save_path=stamped(SECTION, "yaw_only_correction_paired_$(metric)"),
        )
    end
end
