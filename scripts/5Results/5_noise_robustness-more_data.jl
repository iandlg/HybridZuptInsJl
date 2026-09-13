# Section 5b: does MORE (noisy) training data buy back robustness?
#
# The question: ground truth used for training is corrupted, which costs performance. Does
# accumulating more of it recover that loss? Training tracks are added one at a time and the
# frozen model is re-tested after each addition, against the no-correction ZUPT baseline
# (dashed line in the figure). Noise is applied to the training tracks only; the test
# tracks' GT stays clean.
#
# The accumulation order is now DRAWN, not chosen. WAS: tracks were fed in the declared
# order of `train_labels`, one arbitrary permutation, so every point on the curve confounded
# "more data" with "which track came next" -- and whichever permutation happened to be typed
# at the top of this file decided what the figure said. Each of the SEEDS repeats now walks
# its own random permutation and each box spans the repeats.
#
# Two things to keep straight when reading a box:
#
#   - up to the last group, a box spans both the order AND which subset of tracks that
#     permutation reached first, which is part of the "more data" question;
#   - in the LAST group every repeat has trained on the same set of tracks in a different
#     sequence, so the box there is order sensitivity in the fit alone. It should be
#     near-degenerate; that is the check that the shuffling measures what it should.
#
# Each run writes its per-repeat rows to a CSV beside the figure; set `replot_csv` below to
# one of those paths to redraw it without paying for the sweep again.
#
# The training-GT noise is NOT redrawn per repeat: a track's realisation is keyed on the
# track id alone (`Xoshiro(1000*train_id)` in `multi_track_training_analysis`), so it is the
# same in every repeat and for every estimator. That is what makes the last group a clean
# order-only check -- and it also means this figure marginalises over order but carries a
# single noise draw per track. See notes/005 and notes/006.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics, Printf
import CSV

data_key = "DCSC"
data_dir_path = data_dir(data_key)

estimators = OrderedDict(
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator,
    "HSGP" => HybridZuptInsJl.DecoupledHsgpEstimator,
)
# NOTE: the order of these entries no longer matters -- it defines the *set* of training
# tracks, and each repeat draws its own permutation of it.
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
hsgp_p_key = 47
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
        [0.146, 30.0, 127.0]
    )
    params = HybridZuptInsJl.basecopy(params; new_hp=new_hp)
end

# Set to the path of a CSV written by an earlier run to re-plot it and skip the sweep
# entirely -- the whole point of writing one CSV per run. `nothing` runs the sweep over
# every entry in `noise_specs` below. The path is taken as given (relative to the repo
# root, or absolute); it is not resolved against the section directory.
replot_csv = nothing
# replot_csv = "out/Results/5_NoiseRobustness/MoreData/multi_track_training_pos1.0_att10_2026-09-13T15:59:57.326.csv"

# One repeat per seed, each a random accumulation order. Cost is
# n_seeds x estimators x train_tracks x (1 train + n_test_tracks) filter runs:
# 5 x 2 x 7 x 4 = 280 per noise spec, ~12 min.
N_REPEATS = 5
SEEDS = collect(1:N_REPEATS)

noise_specs = OrderedDict(
    "pos1.0_att10" => HybridZuptInsJl.NoiseSpec(; pos_std=1.0, att_std=10*pi/180,
        tag="Position & Heading Noise (1.m, ±10°)"),
    "pos0.1_att10" => HybridZuptInsJl.NoiseSpec(; pos_std=0.1, att_std=10*pi/180,
        tag="Position & Heading Noise (0.1m, ±10°)"),
)

const SECTION = "5_NoiseRobustness/MoreData"
const METRIC = :rmse

"""Median metric at every training-set size, with the no-correction baseline beside it.
This is the headline read: along a row is "more data", against `base` is "buys back"."""
function summarise_more_data(df::DataFrame, label::AbstractString)
    trained = df[df.train_set .!= "Base", :]
    base = df[df.train_set .== "Base", :]
    steps = sort(unique(skipmissing(trained.train_set_order)))

    @printf("\n=== %s : median %s over %d random orders ===\n",
        label, METRIC, length(unique(skipmissing(trained.seed))))
    print(rpad("test track", 22), rpad("estimator", 10), rpad("base", 9))
    println(join([rpad("n=$n", 9) for n in steps]))
    for test_id in sort(unique(trained.test_id), by=t -> first(trained[trained.test_id.==t, :test_order]))
        tsub = trained[trained.test_id.==test_id, :]
        bval = first(base[base.test_id.==test_id, METRIC])
        for est in unique(tsub.estimator)
            esub = tsub[tsub.estimator.==est, :]
            meds = map(steps) do n
                v = esub[esub.train_set_order.==n, METRIC]
                isempty(v) ? NaN : median(v)
            end
            print(rpad(first(tsub.test_name), 22), rpad(est, 10), rpad(round(bval; digits=3), 9))
            println(join([rpad(round(m; digits=3), 9) for m in meds]))
        end
    end
end

"""Final step only: every repeat has trained on the same set, so the spread here is the
order sensitivity of the incremental fit. Static is exactly linear-Gaussian and should sit
at roundoff; HSGP linearises its measurement noise about the current β (notes/006) and is
expected small but nonzero."""
function summarise_order_invariance(df::DataFrame, label::AbstractString)
    trained = df[df.train_set .!= "Base", :]
    n_max = maximum(skipmissing(trained.train_set_order))
    fin = trained[trained.train_set_order.==n_max, :]

    @printf("\n=== %s : final step, n=%d tracks, %d orders ===\n",
        label, n_max, length(unique(skipmissing(fin.seed))))
    println(rpad("test track", 22), rpad("estimator", 10),
        rpad("min", 11), rpad("median", 11), rpad("max", 11), "rel spread")
    for test_id in sort(unique(fin.test_id), by=t -> first(fin[fin.test_id.==t, :test_order]))
        tsub = fin[fin.test_id.==test_id, :]
        for est in unique(tsub.estimator)
            v = tsub[tsub.estimator.==est, METRIC]
            isempty(v) && continue
            med = median(v)
            print(rpad(first(tsub.test_name), 22), rpad(est, 10),
                rpad(round(minimum(v); sigdigits=5), 11),
                rpad(round(med; sigdigits=5), 11),
                rpad(round(maximum(v); sigdigits=5), 11))
            @printf("%.2e  (%d draws)\n", (maximum(v) - minimum(v)) / med, length(v))
        end
    end
end

if isnothing(replot_csv)
    results = OrderedDict{String,DataFrame}()
    for (noise_label, noise) in noise_specs
        @info "##### Noise spec $(noise_label): $(noise.tag) #####"

        df_spec = HybridZuptInsJl.multi_track_training_analysis(
            data_dir_path, estimators, train_labels, test_labels, params;
            frame=FRAME, feature_type=FEATURE_TYPE, corrected_channels=output_channels,
            noise_spec=noise,
            order_seeds=SEEDS,
            train_tr_ratio=1.0,
            test_tr_ratio=0.1,
        )
        results[noise_label] = df_spec

        results_figure() do
            HybridZuptInsJl.plot_multi_track_training_quality(
                df_spec;
                metric=METRIC,
                save_path=stamped(SECTION, "multi_track_training_$(noise_label)"),
            )
        end

        # The per-repeat rows, beside the figure: a box of 5 points is worth being able to
        # look at, the `train_set` column is the only record of which permutation each
        # repeat drew, and `replot_csv` above turns this file back into the figure.
        CSV.write(stamped(SECTION, "multi_track_training_$(noise_label)"; ext="csv"), df_spec)

        summarise_more_data(df_spec, noise.tag)
        summarise_order_invariance(df_spec, noise.tag)
    end
else
    df_results = CSV.read(replot_csv, DataFrame)
    @info "Re-plotting from $replot_csv" nrow(df_results)

    # The figure takes the CSV's own stem rather than a fresh timestamp: it is not new
    # evidence, it is the same run drawn again, and pairing the names is what lets you tell
    # which table a figure came from. Re-plotting the same CSV overwrites its figure, which
    # is what you want while iterating on the styling.
    fig_path = results_path(SECTION, replace(basename(replot_csv), r"\.csv$" => ".pdf"))
    results_figure() do
        HybridZuptInsJl.plot_multi_track_training_quality(
            df_results; metric=METRIC, save_path=fig_path)
    end
    @info "Wrote $fig_path"

    summarise_more_data(df_results, basename(replot_csv))
    summarise_order_invariance(df_results, basename(replot_csv))
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
