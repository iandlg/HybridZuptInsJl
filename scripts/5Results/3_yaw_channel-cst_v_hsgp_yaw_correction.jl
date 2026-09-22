# Section 3 (yaw channel): can a constant yaw correction replace the HSGP one?
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
import CSV

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
# Correction filter (see CORRECTION_FILTERS in _common.jl). Its tag goes into every
# output file name, and picks the correctors below (CORRECTORS): V4 needs the
# JointStride ones, and running it with V2's Decoupled correctors silently corrects
# nothing. WAS: the Decoupled correctors under V2's default filter, which is why the
# figures in this section were not comparable with the rest of the chapter.
filter_tag = "V4"
# V4 stride-noise arm (`StrideNoise`): `:process_only` is σ_w = σ_n fixed from the
# hyperparameters, no split and no online estimate.
noise_mode = :process_only

estimators = OrderedDict(
    "ZUPT only" => HybridZuptInsJl.BaseEstimator,
    "Static" => CORRECTORS[filter_tag].static,
    "HSGP" => CORRECTORS[filter_tag].hsgp,
)

output_channels = [:pos_1, :pos_2, :yaw]
train_ratios = [0.5]

## 5. Run the sweep
results_df = HybridZuptInsJl.run_online_correction_sweep(
    aligned,
    FRAME,
    FEATURE_TYPE,
    hsgp_p,
    train_ratios,
    estimators,
    output_channels;
    correction_filter=CORRECTION_FILTERS[filter_tag],
    estimator_kwargs=(noise_mode=noise_mode,),
)

## 6. Plot
const SECTION = "3_yaw_channel/Const_v_Hsgp_yaw_correction"
# Figures in the section directory, scores table in its data/ subdirectory.
const DATA_SECTION = "$(SECTION)/data"
const RUN_STEM = "$(filter_tag)_$(noise_mode)_key$(hsgp_p_key)"

# Persist the numbers behind the figures. WAS: nothing was written at all, so every
# statement this section makes had to be re-run to be checked. Only the scalar columns:
# the sweep also carries the raw zupt/step_seg/corr_traj/io_data/model objects, which
# have no CSV representation.
score_cols = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order,
    :estimator, :estimator_order, :noise_spec_tag, :noise_spec_order, :seed,
    :rmse, :rmse_rate, :rmse_yaw]
CSV.write(stamped(DATA_SECTION, "yaw_only_correction_$(RUN_STEM)"; ext="csv"),
    results_df[:, score_cols])

# Primary: the channel actually being corrected.
results_figure() do
    HybridZuptInsJl.boxplot_dataset_comparison(
        results_df;
        metric=:rmse_yaw,
        train_ratio=0.5,
        save_path=stamped(SECTION, "yaw_only_correction_YAW_$(RUN_STEM)"),
    )
end

# Secondary: what it costs downstream in position.
results_figure() do
    HybridZuptInsJl.boxplot_dataset_comparison(
        results_df;
        metric=:rmse_rate,
        train_ratio=0.5,
        save_path=stamped(SECTION, "yaw_only_correction_POS_$(RUN_STEM)"),
    )
end

# Paired view: five estimators x two datasets from n≈10 each is a lot of boxes
# to compare by eye, and they are all the same trials. Same grouped-boxplot
# view as section 2's dataset comparison, so the two figures read alike.
for metric in (:rmse_yaw, :rmse)
    paired = HybridZuptInsJl.paired_estimator_contrast(
        results_df; metric=metric, reference_estimator="ZUPT only")

    results_figure() do
        HybridZuptInsJl.plot_dataset_paired_relative_change(
            paired;
            metric=metric,
            reference_label="ZUPT only",
            show_points=false,
            show_outliers=true,
            show_subtitle=false,
            save_path=stamped(SECTION, "yaw_only_correction_paired_$(metric)_$(RUN_STEM)"),
        )
    end
end
