include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
using GLMakie, OrderedCollections

# Choose Parameters file (see HSGP_PARAM_PATHS in scripts/5Results/_common.jl)
hsgp_p_key = 42
m = 200

# Load parameters with corresponding metatdata
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)
use_hand_tuned = false
if use_hand_tuned
    new_hp = HybridZuptInsJl.SeHyperparams(
        [0.06063349111094665, 1.9009449161575966, 0.06392436373774413],
        [0.06063340044438242, 1.755542511682359, 0.09133912436420788],
        [0.06063313723928771, 1.4138158563148968, 0.018763841300751767],
        [0.01721875922611514, 14.684512247403717, 127.94772453083775]
    )
    # original 42
    # new_hp = HybridZuptInsJl.SeHyperparams(
    #     [0.06063349111094665, 1.9009449161575966, 0.06392436373774413],
    #     [0.06063340044438242, 1.755542511682359, 0.09133912436420788],
    #     [0.06063313723928771, 1.4138158563148968, 0.018763841300751767],
    #     [0.01721875922611514, 14.684512247403717, 127.94772453083775]
    # )
    hsgp_p = HybridZuptInsJl.basecopy(hsgp_p; new_hp=new_hp)
end

data_key = "ANG2" # meta["data_key"]
data_dir_path = data_dir(data_key)
# Saved figures go to out/Results/<section>/, the same tree scripts/5Results/ writes to.
# Plain variable, not `const`: this script gets re-included in a live REPL.
section = "1_Performance/Trajectory2D"

var_pos = 1e-3
var_yaw = 1e-3
pos_std = 0.0 # 0.5
att_std = 0.0 # 5*pi/180
sigma_groundtruth = (
    sqrt(var_pos)+pos_std,
    sqrt(var_pos)+pos_std,
    sqrt(var_pos)+pos_std,
    sqrt(var_yaw)+att_std
)
posyaw_measurement_update=true

trial_id = 4 # meta["trial_id"]
train_ratio = 0.5
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]
# sim_config = HybridZuptInsJl.InsConfig(sigma_groundtruth=sigma_groundtruth)
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir_path, trial_id;# sim_config=sim_config
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

# Add noise to GT
noisy_gt_traj = HybridZuptInsJl.add_gaussian_noise(gt_traj_aligned; pos_std=pos_std, att_std=att_std)

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

# Run online correction
io_data = OrderedDict()

default_corr = HybridZuptInsJl.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

decoup_static_est = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60); corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], decoup_stat_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

# static_corr = HybridZuptInsJl.JointStaticEstimator(round(Int, N / 60); corrected_channels=output_channels)
# zupt, step_seg, stat_corr_traj, io_data["Joint Static"], joint_stat_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, noisy_gt_traj, static_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

decoup_hsgp_estmtr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], hsgp_decoup_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_hsgp_estmtr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)


# slamHsgp_corr = HybridZuptInsJl.JointHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
# zupt, step_seg, slamHsgp_corr_traj, io_data["Joint HSGP"], hsgp_joint_model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, noisy_gt_traj, slamHsgp_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "ZUPT only" => def_corr_traj,
    "Static" => decoupled_stat_traj,
    # "Joint Static" => stat_corr_traj,
    "HSGP" => hsgp1_corr_traj,
    # "Joint HSGP" => slamHsgp_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])

# results_figure() == CairoMakie + theme_ggplot2(), the theme every saved results figure
# uses. It has to be CairoMakie: saving SVG under GLMakie silently rasterises the figure.
fig = results_figure() do
    HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg];
        segment=:test, train_ratio=train_ratio, show_heading=false, heading_stride=1,
        save_path=stamped(section, "trajectory2d_$(data_key)_trial$(trial_id)"))
end

# Same test window as above, with the absolute distance error panel alongside it and a
# single shared legend. Saved next to the map-only figure, under its own name.
fig_traj_err = results_figure() do
    HybridZuptInsJl.plot_trajectory_and_distance_error(trajs, gt_traj_aligned[step_seg];
        segment=:test, train_ratio=train_ratio,
        save_path=stamped(section, "trajectory2d_disterr_$(data_key)_trial$(trial_id)"))
end
# results_figure leaves CairoMakie active; restore GLMakie so later plots still open windows.
GLMakie.activate!()

with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
end
# fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])
fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Base"]["target"])
# fig_in_def = HybridZuptInsJl.plot_input_features(io_data["Default"]["input"])
# fig_in_hsgp = HybridZuptInsJl.plot_input_features(io_data["SplitHsgp"]["input"])

## Test on second track
trial_id = 14
train_ratio = 0.1
posyaw_measurement_update = true
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir_path, trial_id; sim_config=sim_config
)
# sim_config_updated.sigma_groundtruth = (sqrt(var_pos), sqrt(var_pos), sqrt(var_pos), sqrt(var_yaw))
noisy_gt_traj = HybridZuptInsJl.add_gaussian_noise(gt_traj_aligned; pos_std=pos_std, att_std=att_std)

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

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

io_data = OrderedDict()

default_corr = HybridZuptInsJl.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

decoup_static_est = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60); corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=decoup_stat_model, posyaw_measurement_update=posyaw_measurement_update)

joint_static_est = HybridZuptInsJl.JointStaticEstimator(round(Int, N / 60); corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, joint_stat_traj, io_data["Joint Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, joint_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=joint_stat_model, posyaw_measurement_update=posyaw_measurement_update)

decoup_hsgp_estmtr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_hsgp_estmtr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=hsgp_decoup_model, posyaw_measurement_update=posyaw_measurement_update)

slamHsgp_corr = HybridZuptInsJl.JointHsgpEstimator(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, slamHsgp_corr_traj, io_data["Joint HSGP"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, slamHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=hsgp_joint_model, posyaw_measurement_update=posyaw_measurement_update)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "ZUPT only" => def_corr_traj,
    "Static" => decoupled_stat_traj,
    "Joint Static" => joint_stat_traj,
    "HSGP" => hsgp1_corr_traj,
    "Joint HSGP" => slamHsgp_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
# results_figure() == CairoMakie + theme_ggplot2(), the theme every saved results figure
# uses. It has to be CairoMakie: saving SVG under GLMakie silently rasterises the figure.
fig = results_figure() do
    HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg];
        segment=:test, train_ratio=train_ratio, show_heading=false, heading_stride=1,
        save_path=stamped(section, "trajectory2d_$(data_key)_trial$(trial_id)"))
end
# results_figure leaves CairoMakie active; restore GLMakie so later plots still open windows.
GLMakie.activate!()
fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Base"]["target"])

with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
end