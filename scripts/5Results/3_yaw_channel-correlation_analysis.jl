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
data_key = "ANG2"
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
            output_io.t, output_io.data[[1, 2, 4], :], output_io.data_std[[1, 2, 4], :]
        )

        # Remove outliers
        input_io, output_io = HybridZuptInsJl.remove_outliers(input_io, output_io;
            method="mahalanobis", threshold=3.0, alpha=0.975, dims=:output)


        # Compute training IO and CCA (reuse run_correlation_analysis but only return CCA results)
        fig, canonical_corrs, _ = HybridZuptInsJl.run_correlation_analysis(input_io, output_io;
            feature_type=ft, output_labels=["Δx", "Δy", "Δψ"])
        k = length(canonical_corrs)
        score1 = canonical_corrs[1]                     # first canonical correlation
        score_sum_sq = sqrt(mean(canonical_corrs .^ 2))   # RMS of all
        push!(results, (frame, ft, score1, score_sum_sq, canonical_corrs, fig))
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

## Plot
# CairoMakie, not GLMakie: this writes an SVG for the thesis and must run
# headless. The previous version opened a GLMakie Screen per result, which
# fails without a display and produced raster-backed output when it did run.
const SECTION = "3_yaw_channel/Correlation_Analysis"
results_figure() do
    path = stamped(SECTION, "correlation_analysis")
    CairoMakie.save(path, results[end][6])
    @info "Saved figure: $path"
end