include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
using GLMakie, OrderedCollections

data_key = "ANG2" # meta["data_key"]
data_dir_path = data_dir(data_key)

# Train split lives in _common.jl (TRAIN_IDS) so the normalisation constants
# derived here are computed over exactly the walks the optimiser fits on.
train_trial_ids = train_ids(data_key)
FRAME = HybridZuptInsJl.HEADING
FEATURE_TYPE = HybridZuptInsJl.TWOD_STEP_YAW

# Compute IO data
valid_results = HybridZuptInsJl.collect_dataset(
    data_dir_path, train_trial_ids;
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
    # Was alpha=0.975 (chi-squared). keep_fraction states the same intent -- trim the
    # extreme 2.5% -- and now actually delivers it: the chi-squared cut assumed D² was
    # χ²_d, which these residuals are not, so it trimmed far less than it claimed.
    "keep_fraction" => 0.847,
)
if !isnothing(outlier_removal_params["dims"])
    train_in, train_out = HybridZuptInsJl.remove_outliers(train_in, train_out;
        method=outlier_removal_params["method"],
        threshold=outlier_removal_params["threshold"],
        keep_fraction=outlier_removal_params["keep_fraction"],
        dims=outlier_removal_params["dims"]
    )
end

inp = HybridZuptInsJl.compute_input_preprocessing(train_in.data; normalize_x=true, margin=0.5)
outp = HybridZuptInsJl.compute_output_normalisation(train_out.data; normalize_y=true)

HybridZuptInsJl.display_preprocessing(inp, outp)