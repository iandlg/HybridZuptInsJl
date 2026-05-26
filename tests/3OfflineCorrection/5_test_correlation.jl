include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections


# Parameters (same as in 2_batch_correction.jl)
data_key = "ANG"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision"
)[data_key]
trial_id = 15

# Run for each feature type
fig_bod, _, fig_corr_bod = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, HybridZuptInsJl.BODY, HybridZuptInsJl.THREED_STEP_DT)
fig_head, _, fig_corr_head = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, HybridZuptInsJl.HEADING, HybridZuptInsJl.THREED_STEP_DT)

fig_head, _, fig_corr_head = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, HybridZuptInsJl.BODY, HybridZuptInsJl.THREED_STEP)
fig_head, _, fig_corr_head = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, HybridZuptInsJl.HEADING, HybridZuptInsJl.THREED_STEP)

fig_head, _, fig_corr_head = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, HybridZuptInsJl.BODY, HybridZuptInsJl.TWOD_STEP_DT)
fig_head, _, fig_corr_head = HybridZuptInsJl.run_correlation_analysis(trial_id, data_dir, HybridZuptInsJl.HEADING, HybridZuptInsJl.TWOD_STEP_DT)
