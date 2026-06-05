include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames, Statistics

data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]
FRAME = HybridZuptInsJl.BODY
FEATURE_TYPE = HybridZuptInsJl.THREED_STEP

# All trials in the directory
trial_ids = HybridZuptInsJl.list_trial_ids(data_dir)

# Or hand-pick / exclude specific ones
trial_ids = filter(!=(3), HybridZuptInsJl.list_trial_ids(data_dir))

outputs, features = HybridZuptInsJl.collect_dataset(
    data_dir, trial_ids;
    frame=FRAME,
    feature_type=FEATURE_TYPE
)
