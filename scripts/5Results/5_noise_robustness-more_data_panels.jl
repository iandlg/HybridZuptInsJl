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
const DATA_SECTION = "$(SECTION)/data"
const METRIC = :rmse   # the per-spec CSVs carry rmse / rmse_rate only, no rmse_yaw

# The run whose tables are stitched. These five must match the sibling script's settings:
# they are what its file stems are built from, and so what the panels are looked up by.
filter_tag = "V4"
noise_mode = :process_only
hsgp_p_key = 42
data_key = get(ENV, "DATA_KEY", "DCSC")
test_tr_ratio = 0.0

# Panel label => the `noise_label` the sibling wrote into its file stem, in panel order.
# The noise spec is not a column in those tables -- one run is one spec -- so this is the
# only record of which spec a table came from. WAS a list of pasted absolute paths, which
# went stale the moment the sweep was re-run.
panel_specs = OrderedDict{String,String}(
    "No Noise" => "no_noise",
    "Position & Heading Noise (0.1m, ±10°)" => "pos0.1_att10",
    "Position & Heading Noise (1.0m, ±10°)" => "pos1.0_att10",
)

"""Newest table this configuration wrote for `noise_label`. The sibling stamps every file
with the moment it ran, so the name cannot be known in advance; the stem is otherwise
identical, which makes the lexicographic maximum the most recent run."""
function latest_run_csv(noise_label::AbstractString)::String
    dir = joinpath(RESULTS_ROOT, DATA_SECTION)
    prefix = "multi_track_training_$(filter_tag)_$(noise_mode)_matchedR_key$(hsgp_p_key)_$(data_key)_testgt$(test_tr_ratio)_$(noise_label)_"
    matches = isdir(dir) ? filter(f -> startswith(f, prefix) && endswith(f, ".csv"), readdir(dir)) : String[]
    isempty(matches) && error("no $(prefix)*.csv in $dir -- run 5_noise_robustness-more_data.jl \
                               with DATA_KEY=$data_key first")
    return joinpath(dir, last(sort(matches)))
end

panel_csvs = OrderedDict{String,String}(
    label => latest_run_csv(noise_label) for (label, noise_label) in panel_specs)

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

# Which test track the panels show. The sibling runs one test track per dataset, so the
# frame names it; `TEST_ID` in the environment picks one when a run had several.
TEST_ID = haskey(ENV, "TEST_ID") ? parse(Int, ENV["TEST_ID"]) : only(unique(df.test_id))

test_name = first(df[df.test_id .== TEST_ID, :test_name])
results_figure() do
    HybridZuptInsJl.plot_multi_track_training_noise_panels(
        df, TEST_ID;
        metric=METRIC,
        save_path=stamped(SECTION,
            "multi_track_training_noise_panels_$(filter_tag)_$(noise_mode)_key$(hsgp_p_key)_$(data_key)_testgt$(test_tr_ratio)_$(test_name)"),
    )
end

summarise_order_rsd(df, TEST_ID)
