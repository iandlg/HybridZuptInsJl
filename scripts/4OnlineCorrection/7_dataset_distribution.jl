include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

data_key = "DCSC" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

train_ids = Dict{String,Vector{Int}}(
    "ANG2" => [1, 2, 3, 4, 8, 9, 13, 14, 15, 16],
    "DCSC" => [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
)[data_key]
FRAME = HybridZuptInsJl.HEADING
FEATURE_TYPE = HybridZuptInsJl.TWOD_STEP_YAW

# Compute IO data
valid_results = HybridZuptInsJl.collect_dataset(
    data_dir, train_ids;
    frame=FRAME,
    feature_type=FEATURE_TYPE,
)
# ===================================================
# === Boxplot of dataset input output data  =========
# ===================================================
df = HybridZuptInsJl.io_dataframe(valid_results)
fig = HybridZuptInsJl.plot_channel_boxplots(df)

# ===================================================
# === Normalisation and Bounding Box constants  =====
# ===================================================
# Each result tuple: (target::CorrectionIO, input::CorrectionIO, corr_traj, gt_traj, step_seg)
train_out = HybridZuptInsJl.concatenate_io([res[1] for (key, res) in valid_results])
train_in = HybridZuptInsJl.concatenate_io([res[2] for (key, res) in valid_results])

# Remove outliers
outlier_removal_params = OrderedDict(
    "dims" => :output, # nothing or :input or :both
    "method" => "mahalanobis",
    "threshold" => 3.0,
    "alpha" => 0.975,
)
if !isnothing(outlier_removal_params["dims"])
    train_in, train_out = HybridZuptInsJl.remove_outliers(train_in, train_out;
        method=outlier_removal_params["method"],
        threshold=outlier_removal_params["threshold"],
        alpha=outlier_removal_params["alpha"],
        dims=outlier_removal_params["dims"]
    )
end

inp = HybridZuptInsJl.compute_input_preprocessing(train_in.data; normalize_x=true, margin=0.5)
outp = HybridZuptInsJl.compute_output_normalisation(train_out.data; normalize_y=true)

HybridZuptInsJl.display_preprocessing(inp, outp)