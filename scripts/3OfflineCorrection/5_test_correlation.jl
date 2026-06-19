include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames, Statistics


# Parameters (same as in 2_batch_correction.jl)
data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]
trial_id = 5

frames = [
    HybridZuptInsJl.BODY,
    HybridZuptInsJl.HEADING
]
feature_types = [
    HybridZuptInsJl.THREED_STEP,
    HybridZuptInsJl.TWOD_STEP_DT,
    HybridZuptInsJl.THREED_STEP_DT
]

results = []
for frame in frames
    for ft in feature_types
        # Compute training IO and CCA (reuse run_correlation_analysis but only return CCA results)
        _, canonical_corrs, _ = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, frame, ft)
        k = length(canonical_corrs)
        score1 = canonical_corrs[1]                     # first canonical correlation
        score_sum_sq = sqrt(mean(canonical_corrs .^ 2))   # RMS of all
        push!(results, (frame, ft, score1, score_sum_sq, canonical_corrs))
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