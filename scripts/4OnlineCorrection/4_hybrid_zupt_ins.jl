include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 40

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    40 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_DT/ANG15_HEADING_TWOD_STEP_DT_2026-07-10T15:06:17.927.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
data_key = "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])
m = 200
var_pos = 1e-2
var_yaw = 1e-3

trial_id = 15 # meta["trial_id"]
train_ratio = 0.45

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)
# sim_config_updated.sigma_groundtruth = (sqrt(var_pos), sqrt(var_pos), sqrt(var_pos), sqrt(var_yaw))

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
# n_train_phases = 5
# block_size = N ÷ (2 * n_train_phases)

# gt_available = [((i ÷ block_size) % 2 == 0) for i in 0:(N-1)]

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
zupt, step_seg, def_corr_traj, io_data["Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

tatic_corr = HybridZuptInsJl.JointStaticEstimator(round(Int, N / 60))
zupt, step_seg, stat_corr_traj, io_data["Joint Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, tatic_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

decoup_static_est = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60))
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


splitHsgp_corr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60), hsgp_p)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], split_β_Σβ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


slamHsgp_corr = HybridZuptInsJl.JointHsgpEstimator(round(Int, N / 60), hsgp_p)
zupt, step_seg, slamHsgp_corr_traj, io_data["Joint HSGP"], slam_β_Σβ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "Base" => def_corr_traj,
    "Decoupled Static" => decoupled_stat_traj,
    "Joint Static" => stat_corr_traj,
    "Decoupled HSGP" => hsgp1_corr_traj,
    "Joint HSGP" => slamHsgp_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=1, stop=10, show_heading=true, heading_stride=1)

with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg], train_ratio; show_index_ticks=true)
end
# fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])
fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Base"]["target"])
# fig_in_def = HybridZuptInsJl.plot_input_features(io_data["Default"]["input"])
# fig_in_hsgp = HybridZuptInsJl.plot_input_features(io_data["SplitHsgp"]["input"])

## Test on second ack
trial_id = 2 # meta["trial_id"]
train_ratio = 0.45

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)
sim_config_updated.sigma_groundtruth = (sqrt(var_pos), sqrt(var_pos), sqrt(var_pos), sqrt(var_yaw))

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
# n_train_phases = 5
# block_size = N ÷ (2 * n_train_phases)

# gt_available = [((i ÷ block_size) % 2 == 0) for i in 0:(N-1)]

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
zupt, step_seg, def_corr_traj, io_data["Default"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

tatic_corr2 = HybridZuptInsJl.StaticCorrectorV2(round(Int, N / 60))
zupt, step_seg, stat_corr_traj2, io_data["Static2"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, tatic_corr2;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

splitHsgp_corr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60), hsgp_p)
zupt, step_seg, hsgp1_corr_traj, io_data["SplitHsgp"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, β_Σβ_0=split_β_Σβ)


slamHsgp_corr = HybridZuptInsJl.JointHsgpEstimator(round(Int, N / 60), hsgp_p)
zupt, step_seg, slamHsgp_corr_traj, io_data["SlamHsgp"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, β_Σβ_0=slam_β_Σβ)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "zupt ins" => def_corr_traj,
    "static correction" => stat_corr_traj2,
    "split hsgp correction" => hsgp1_corr_traj,
    "slam hsgp" => slamHsgp_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=1, stop=10, show_heading=true, heading_stride=1)
fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Default"]["target"])

with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg], train_ratio; show_index_ticks=true)
end