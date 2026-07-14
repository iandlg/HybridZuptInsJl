include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames, Statistics, GLMakie


# Parameters (same as in 2_batch_correction.jl)
data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

trial_ids = [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16]


frames = [
    # HybridZuptInsJl.BODY,
    HybridZuptInsJl.HEADING
]
feature_types = [
    # HybridZuptInsJl.THREED_STEP,
    # HybridZuptInsJl.TWOD_STEP_DT,
    # HybridZuptInsJl.THREED_STEP_DT,
    # HybridZuptInsJl.TWOD_STEP_YAW,
    HybridZuptInsJl.THREED_STEP_DT_YAW
]

results = []
for frame in frames
    for ft in feature_types
        # Load data
        dataset = HybridZuptInsJl.collect_dataset(
            data_dir, trial_ids;
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

with_theme(theme_ggplot2()) do
    for r in results
        display(GLMakie.Screen(), r[6])
    end
end