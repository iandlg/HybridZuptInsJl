# Section 3 (yaw channel): how much of the correction target do the input
# features actually explain? Low input/output correlation on the yaw channel is
# the justification for treating yaw differently from x/y.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, DataFrames, Statistics

# WAS: trial_ids = [1, 2, 3, 4, 5, 6, 8, 10, 12] under data_key = "ANG2".
# That is the DCSC list. Because trial ids are just integers, running it against
# ANG2 silently selected a different set of walks instead of erroring, so the
# published heatmap does not describe the trials its caption claims. Now taken
# from the shared table, which is keyed by dataset.
data_key = "DCSC"
data_dir_path = data_dir(data_key)
ids = trial_ids(data_key)


frames = [
    # HybridZuptInsJl.BODY,
    HybridZuptInsJl.HEADING
]
feature_types = [
    # HybridZuptInsJl.THREED_STEP,
    # HybridZuptInsJl.TWOD_STEP_DT,
    # HybridZuptInsJl.THREED_STEP_DT,
    HybridZuptInsJl.TWOD_STEP_YAW,
    # HybridZuptInsJl.THREED_STEP_DT_YAW
]

# Channels kept by the row selection below, in the same order. Named once so the
# heatmaps cannot drift out of step with the slice they label.
const OUTPUT_CHANNELS = [1, 2, 4]
const OUTPUT_LABELS = ["Δx", "Δy", "Δψ"]

results = []
for frame in frames
    for ft in feature_types
        # Load data
        dataset = HybridZuptInsJl.collect_dataset(
            data_dir_path, ids;
            frame=frame,
            feature_type=ft,
        )
        data_vec = [dataset[id] for id in keys(dataset)]

        input_io = HybridZuptInsJl.concatenate_io([res[2] for res in data_vec])
        output_io = HybridZuptInsJl.concatenate_io([res[1] for res in data_vec])

        # Consider only x, y yaw corrections
        output_io = HybridZuptInsJl.CorrectionIO(
            output_io.t, output_io.data[OUTPUT_CHANNELS, :], output_io.data_std[OUTPUT_CHANNELS, :]
        )

        # Remove outliers. Was alpha=0.975 (chi-squared); keep_fraction states the same
        # 2.5% trim directly and actually delivers it -- the chi-squared cut assumed D²
        # was χ²_d, which these residuals are not, so it removed fewer points than the
        # 0.975 suggested. This figure therefore changes slightly on a re-run.
        input_io, output_io = HybridZuptInsJl.remove_outliers(input_io, output_io;
            method="mahalanobis", threshold=3.0, keep_fraction=0.85, dims=:output)


        # Compute training IO and CCA (reuse run_correlation_analysis but only return CCA results)
        fig, canonical_corrs, _ = HybridZuptInsJl.run_correlation_analysis(input_io, output_io;
            feature_type=ft, output_labels=OUTPUT_LABELS)
        k = length(canonical_corrs)
        score1 = canonical_corrs[1]                     # first canonical correlation
        score_sum_sq = sqrt(mean(canonical_corrs .^ 2))   # RMS of all
        push!(results, (frame, ft, score1, score_sum_sq, canonical_corrs, fig, output_io))
    end
end

# Convert to DataFrame for easy sorting
df = DataFrame(
    frame=[r[1] for r in results],
    feature_type=[r[2] for r in results],
    first_cc=[r[3] for r in results],
    rms_cc=[r[4] for r in results],
    all_corrs=[r[5] for r in results]
)
sort!(df, :first_cc, rev=true)
println("Combinations ranked by first canonical correlation:")
display(df)

## Output-channel correlation
# The heatmap above asks how much of each correction the input features explain.
# This one asks a different question about the same targets: how correlated the
# correction channels are with *each other*. It matters for the yaw argument
# because a Δψ already largely predictable from Δx/Δy is redundant rather
# than a genuinely separate channel, and because the correction is fitted per
# output channel, i.e. it assumes these are independent.
#
# Same trials, same outlier trim and same channel slice as the input/output
# heatmap: this reads the output_io each loop iteration pushed rather than
# reloading, so the two figures cannot describe different data.
output_corr = [(r[1], r[2], HybridZuptInsJl.compute_correlation_matrix(r[7].data, r[7].data))
               for r in results]

for (frame, ft, corr_mat) in output_corr
    println("\nOutput-output correlation ($frame, $ft):")
    println("Channels: ", join(OUTPUT_LABELS, ", "))
    display(round.(corr_mat, digits=3))
end

## Plot
# CairoMakie, not GLMakie: this writes an SVG for the thesis and must run
# headless. The previous version opened a GLMakie Screen per result, which
# fails without a display and produced raster-backed output when it did run.
const SECTION = "3_yaw_channel/Correlation_Analysis"
results_figure() do
    path = stamped(SECTION, "correlation_analysis")
    CairoMakie.save(path, results[end][6])
    @info "Saved figure: $path"

    # Built inside the theme block, unlike the figure above, which
    # run_correlation_analysis constructs back in the loop: Makie resolves a theme
    # when a figure is created, not when it is saved.
    fig = HybridZuptInsJl.plot_correlation_heatmap(
        output_corr[end][3], OUTPUT_LABELS, OUTPUT_LABELS;
        figsize=(500, 450),
        xlabel="Output corrections",
        ylabel="Output corrections")
    path = stamped(SECTION, "output_channel_correlation")
    CairoMakie.save(path, fig)
    @info "Saved figure: $path"
end