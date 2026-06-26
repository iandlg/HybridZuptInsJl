include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 30

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
data_key = "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = 15 # meta["trial_id"]
m = 300
margin = meta["margin"]
train_ratio = 0.45

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)

# Extract the aligned initial state from the trajectory
x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(
        ins_traj_aligned.R_nb[:, :, 1]
    )
)
N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]
gt_available[1:min(500, N)] .= true

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

# Run online correction
hsgp_p = HybridZuptInsJl.HsgpParameters(
    hsgp_p.hp, hsgp_p.d, m, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)

io_data = OrderedDict()

default_corr = HybridZuptInsJl.DefaultCorrector(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Default"] = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

static_corr = HybridZuptInsJl.StaticCorrector(round(Int, N / 60))
zupt, step_seg, stat_corr_traj, io_data["Static"] = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, static_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

# splitHsgp_corr = HybridZuptInsJl.SplitHybridCorrector(round(Int, N / 60), hsgp_p)
# zupt, step_seg, hsgp1_corr_traj, io_data["SplitHsgp"] = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


# slamHsgp_corr = HybridZuptInsJl.SlamCorrector(round(Int, N / 60), hsgp_p)
# zupt, step_seg, slamHsgp_corr_traj, io_data["SlamHsgp"] = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "zupt ins" => def_corr_traj,
    "static correction" => stat_corr_traj,
    # "split hsgp correction" => hsgp1_corr_traj,
    # "slam hsgp" => slamHsgp_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=18, stop=25)
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg], train_ratio; show_index_ticks=false)
end
with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])
end

fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])
fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Default"]["target"])
fig_in_def = HybridZuptInsJl.plot_input_features(io_data["Default"]["input"])
fig_in_hsgp = HybridZuptInsJl.plot_input_features(io_data["SplitHsgp"]["input"])
