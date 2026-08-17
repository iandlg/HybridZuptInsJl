include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 42

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    40 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_DT/ANG15_HEADING_TWOD_STEP_DT_2026-07-10T15:06:17.927.json",
    41 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG23_HEADING_TWOD_STEP_YAW_2026-07-13T12:28:30.952.json",
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no ouptut norm; DCSC
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
pos_std = 0.05
att_std = 0.01
sigma_groundtruth = (
    sqrt(var_pos)+pos_std,
    sqrt(var_pos)+pos_std,
    sqrt(var_pos)+pos_std,
    sqrt(var_yaw)+att_std
)

trial_id = 15 # meta["trial_id"]
train_ratio = 0.45
output_channels = [:yaw] # [:pos_1, :pos_2, :pos_3, :yaw]
sim_config = HybridZuptInsJl.InsConfig(sigma_groundtruth=sigma_groundtruth)
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; sim_config=sim_config
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
# gt_available = fill(true, N)
# n_train_phases = 5
# block_size = N ÷ (2 * n_train_phases)

# gt_available = [((i ÷ block_size) % 2 == 0) for i in 0:(N-1)]

# Add noise to GT 
noisy_gt_traj = HybridZuptInsJl.add_gaussian_noise(gt_traj_aligned; pos_std=pos_std)

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

# Run online correction
hsgp_p = HybridZuptInsJl.basecopy(hsgp_p; new_m=m)

io_data = OrderedDict()

default_corr = HybridZuptInsJl.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

decoup_static_est = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60); corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], decoup_stat_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

static_corr = HybridZuptInsJl.JointStaticEstimator(round(Int, N / 60); corrected_channels=output_channels)
zupt, step_seg, stat_corr_traj, io_data["Joint Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, static_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

decoup_hsgp_estmtr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], hsgp_decoup_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_hsgp_estmtr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


slamHsgp_corr = HybridZuptInsJl.JointHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, slamHsgp_corr_traj, io_data["Joint HSGP"], hsgp_joint_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, slamHsgp_corr;
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
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=90, stop=105, show_heading=false, heading_stride=1)

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

## Test on second track
trial_id = 1 # meta["trial_id"]
train_ratio = 0.1

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
# gt_available = fill(false, N)
# n_train_phases = 5
# block_size = N ÷ (2 * n_train_phases)

# gt_available = [((i ÷ block_size) % 2 == 0) for i in 0:(N-1)]

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

io_data = OrderedDict()

default_corr = HybridZuptInsJl.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

decoup_static_est = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60); corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=decoup_stat_model)

decoup_hsgp_estmtr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, decoup_hsgp_estmtr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=hsgp_decoup_model)

slamHsgp_corr = HybridZuptInsJl.JointHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, slamHsgp_corr_traj, io_data["Joint HSGP"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=hsgp_joint_model)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "Base" => def_corr_traj,
    "Decoupled Static" => decoupled_stat_traj,
    "Decoupled HSGP" => hsgp1_corr_traj,
    # "Joint HSGP" => slamHsgp_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=1, stop=30, show_heading=false, heading_stride=1)
fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Base"]["target"])

with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg], train_ratio; show_index_ticks=true)
end