# Section 5b: does MORE (noisy) training data buy back robustness?
#
# Training tracks are accumulated incrementally and the frozen model is re-tested
# after each addition.
#
# CAVEAT: the accumulation order below is one arbitrary permutation of the
# training tracks, so the curve confounds "more data" with "which track came
# next". Either randomise the order over repeats or state the order in the
# caption. Noise is applied to the training tracks only; the test track's GT
# window stays clean.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames

data_key = "DCSC"
data_dir_path = data_dir(data_key)

# estimators = OrderedDict(
#     "DecoupledStatic" => HybridZuptInsJl.JointStaticEstimator,
#     "DecoupledHsgp" => HybridZuptInsJl.DecoupledHsgpEstimator,
# )
estimators = OrderedDict(
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    # "Joint Static" => HybridZuptInsJl.JointStaticEstimator,
    "HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
    # "Joint HSGP" => HybridZuptInsJl.JointHsgpEstimator,
)
train_labels = Dict(
    "ANG2" => OrderedDict(
        4 => "Walk_8",
        13 => "Walk_withoutCarpetShape",
        14 => "Walk_Patrick_long",
        1 => "Walk_rectangles",
        2 => "Walk_rectangles_otherdir",
        3 => "Walk_straight",
    ),
    "DCSC" => OrderedDict(
        4 => "FigureEight_long",
        8 => "CW_Rect",
        12 => "Mixed",
        3 => "CCWRectangle_long_A",
        5 => "S_shape_long",
        6 => "CWRectangle_long",
        10 => "FigureEightLong"
    )
)[data_key]
test_labels = Dict(
    "ANG2" => OrderedDict(
        15 => "Walk_Patrick_mixed",
    ),
    "DCSC" => OrderedDict(
        1 => "CWRectangle_short",
        14 => "CCWRectangle_long_B",
        2 => "FigureEight_short",
    )
)[data_key]
# Choose Parameters file
hsgp_p_key = 46
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

params, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=200)

# Hand-tuned override of the loaded hyperparameters. Set `use_hand_tuned=false`
# to evaluate key $(hsgp_p_key) as trained. Keeping this explicit matters: the
# figure is otherwise labelled with a hyperparameter key whose values were not
# the ones used.
use_hand_tuned = false
if use_hand_tuned
    new_hp = HybridZuptInsJl.SeHyperparams(
        [5e-1, 2.0, 0.09],
        [5e-1, 2.0, 0.09],
        [5e-1, 2.0, 0.09],
        [0.146, 18.0, 2.8]
    )
    params = HybridZuptInsJl.basecopy(params; new_hp=new_hp)
end

noise = HybridZuptInsJl.NoiseSpec(; pos_std=0.1, att_std=10*pi/180, tag="Position & Heading Noise (0.1m, ±5°)")

df_results = HybridZuptInsJl.multi_track_training_analysis(
    data_dir_path, estimators, train_labels, test_labels, params;
    frame=FRAME, feature_type=FEATURE_TYPE, corrected_channels=output_channels,
    noise_spec=noise,
    train_tr_ratio=1.0,
    test_tr_ratio=0.1,
)

## Plot
# WAS: called without save_path, so this script wrote no figure either.
const SECTION = "5_NoiseRobustness/MoreData"

results_figure() do
    HybridZuptInsJl.plot_multi_track_training_quality(
        df_results;
        metric=:rmse,
        save_path=stamped(SECTION, "multi_track_training"),
    )
end

## Optionally persist the hand-tuned hyperparameters.
# WAS: this ran unconditionally and wrote hand-typed values into
# out/4OnlineCorrection/6_HypOpt/, the same store that holds *optimised*
# hyperparameters, with most metadata fields commented out. A results script
# silently mutating the hyperparameter store is how the provenance of keys 44/45
# became unrecoverable. Off by default; the metadata now records what it is.
save_hand_tuned_params = false
if save_hand_tuned_params && use_hand_tuned
    combo_dir = joinpath("out/4OnlineCorrection/6_HypOpt", data_key,
        string(FRAME) * "-" * string(FEATURE_TYPE))
    mkpath(combo_dir)
    filename = "$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(Dates.now()).json"
    HybridZuptInsJl.to_json(joinpath(combo_dir, filename), params;
        metadata=Dict(
            "data_key" => data_key,
            "ref_frame" => FRAME,
            "feature_type" => FEATURE_TYPE,
            "provenance" => "hand-tuned in scripts/5Results/5_noise_robustness-more_data.jl",
            "derived_from_key" => hsgp_p_key,
        )
    )
end