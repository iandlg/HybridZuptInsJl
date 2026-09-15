# Section 5b, the cross-noise read: one test track, one panel per training-GT noise level.
#
# `5_noise_robustness-more_data.jl` runs ONE noise spec per invocation and draws one panel
# per test track, so the comparison this experiment is actually about -- how the same
# track's recovery curve changes as the training ground truth gets worse -- ends up spread
# across separate PDFs. This script stitches those runs back together: it reads the CSV
# each of them wrote and draws a single row of panels, one noise spec each, all of the
# same test track.
#
# It is plot-only. Nothing here re-runs a sweep; to add a panel, run the sibling script
# with that noise spec and add the CSV it writes to `panel_csvs` below.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics, Printf
import CSV

const SECTION = "5_NoiseRobustness/MoreData"
const METRIC = :rmse   # the per-spec CSVs carry rmse / rmse_rate only, no rmse_yaw

# Which test track every panel shows. Ids are the keys of `test_labels` in the sibling
# script; a track that is not in the tables below is an error naming the ones that are.
TEST_ID = 2

# Panel label => the CSV that run wrote, in panel order. The noise spec is not a column in
# those files -- one run is one spec -- so the label here is the only record of which
# spec a table came from. Paths are taken as given (relative to the repo root, or
# absolute), like `replot_csv` in the sibling script.
panel_csvs = OrderedDict{String,String}(
    "No Noise" => "out/Results/5_NoiseRobustness/MoreData/multi_track_training_no_noise_2026-09-15T11:10:37.055.csv",
    "Position & Heading Noise (0.1m, ±10°)" => "out/Results/5_NoiseRobustness/MoreData/multi_track_training_pos0.1_att10_2026-09-15T11:29:30.728.csv",
    "Position & Heading Noise (1.0m, ±10°)" => "out/Results/5_NoiseRobustness/MoreData/multi_track_training_pos1.0_att10_2026-09-15T11:48:25.205.csv",
)

"""Relative standard deviation (std/mean) of the metric at the final accumulation step,
one row per panel. Every repeat has trained on the same set of tracks there, in a
different sequence, so what is left is order sensitivity alone (notes/006)."""
function summarise_order_rsd(df::DataFrame, test_id::Int)
    sub = df[(df.test_id .== test_id) .& (df.train_set .!= "Base"), :]
    isempty(sub) && error("no trained rows for test_id $test_id; have $(sort(unique(df.test_id)))")

    @printf("\n=== %s : final step RSD over accumulation orders ===\n", first(sub.test_name))
    println(rpad("noise spec", 40), rpad("estimator", 10), rpad("n", 4),
        rpad("mean", 11), rpad("std", 13), "rsd")
    for tag in unique(sort(sub, :noise_spec_order).noise_spec_tag)
        psub = sub[sub.noise_spec_tag .== tag, :]
        n_max = maximum(skipmissing(psub.train_set_order))
        fin = psub[psub.train_set_order .== n_max, :]
        for est in unique(fin.estimator)
            v = fin[fin.estimator .== est, METRIC]
            length(v) < 2 && continue
            mu, sd = mean(v), std(v)
            print(rpad(tag, 40), rpad(est, 10), rpad(n_max, 4),
                rpad(round(mu; sigdigits=5), 11), rpad(round(sd; sigdigits=4), 13))
            @printf("%.2e  (%d draws)\n", sd / mu, length(v))
        end
    end
end

# `noise_spec_tag`/`noise_spec_order` are the group/order column pair the plotting code
# facets on everywhere else in this section; they are added here because the sweep that
# wrote these tables had no second spec to distinguish them from.
df = reduce(vcat, map(enumerate(collect(panel_csvs))) do (i, (tag, path))
    d = CSV.read(path, DataFrame)
    d.noise_spec_tag .= tag
    d.noise_spec_order .= i
    d
end)

@info "Panels stitched from $(length(panel_csvs)) runs" collect(values(panel_csvs)) nrow(df)

test_name = first(df[df.test_id .== TEST_ID, :test_name])
results_figure() do
    HybridZuptInsJl.plot_multi_track_training_noise_panels(
        df, TEST_ID;
        metric=METRIC,
        save_path=stamped(SECTION, "multi_track_training_noise_panels_$(test_name)"),
    )
end

summarise_order_rsd(df, TEST_ID)
