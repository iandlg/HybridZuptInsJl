using StrideGP
include("../5Results/_common.jl")
using GLMakie, OrderedCollections, Printf

# Choose Parameters file (see HSGP_PARAM_PATHS in scripts/5Results/_common.jl)
hsgp_p_key = 42
m = 200

# Load parameters with corresponding metatdata
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)
use_hand_tuned = false
if use_hand_tuned
    new_hp = StrideGP.SeHyperparams(
        [0.06063349111094665, 1.9009449161575966, 0.06392436373774413],
        [0.06063340044438242, 1.755542511682359, 0.09133912436420788],
        [0.06063313723928771, 1.4138158563148968, 0.018763841300751767],
        [0.01721875922611514, 14.684512247403717, 127.94772453083775]
    )
    hsgp_p = StrideGP.basecopy(hsgp_p; new_hp=new_hp)
end

data_key = "DCSC" # meta["data_key"]
data_dir_path = data_dir(data_key)
# Saved figures go to out/Results/<section>/, the same tree scripts/5Results/ writes to.
# Plain variable, not `const`: this script gets re-included in a live REPL.
section = "1_Performance/Trajectory2D"
regression_section = "1_Performance/Regression"

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

# Filter used for all three runs, with its correctors (see CORRECTION_FILTERS /
# CORRECTORS in scripts/5Results/_common.jl). "ZUPT only" is `BaseEstimator`, the
# 5Results baseline, run through the SAME filter: it has no learned model, but the
# filter still sets where the corrector starts (on the mocap pose when ground truth is
# available at k=1) and how the mocap fixes are weighted. Its per-stride process noise
# is `sim_config_updated.sigma_stride` (notes/022), not the HSGP σ_n.
filter_tag = "V4"
corr_filter = CORRECTION_FILTERS[filter_tag]

trial_id = 5 # meta["trial_id"]
train_ratio = 0.35
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = StrideGP.compute_aligned_ins_trajectory(
    data_dir_path, trial_id
)

# Extract the aligned initial state from the trajectory
x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    StrideGP.matrix_to_euler(
        ins_traj_aligned.R_nb[:, :, 1]
    )
)
N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]

# Add noise to GT
noisy_gt_traj = StrideGP.add_gaussian_noise(gt_traj_aligned; pos_std=pos_std, att_std=att_std)

true_outputs = Dict{String,StrideGP.CorrectionIO}()
pred_outputs = Dict{String,StrideGP.CorrectionIO}()

# Run online correction
io_data = OrderedDict()

default_corr = StrideGP.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Base"], _ = corr_filter(
    inertial_updated, sim_config_updated, noisy_gt_traj, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME,
    feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

decoup_static_est = CORRECTORS[filter_tag].static(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], decoup_stat_model = corr_filter(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

decoup_hsgp_estmtr = CORRECTORS[filter_tag].hsgp(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], hsgp_decoup_model = corr_filter(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_hsgp_estmtr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

input_data = OrderedDict{String,StrideGP.CorrectionIO}()
output_data = OrderedDict{String,StrideGP.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

# cutoff 
cutoff = 1e9 # s
mask = def_corr_traj.t .< cutoff

# Print the number of strides during training and testing
n_strides = length(def_corr_traj)
n_train_strides = floor(Int, train_ratio * n_strides)
n_test_strides = n_strides - n_train_strides

@info "Number of strides used for training : $n_train_strides"
@info "Training phase duration $(def_corr_traj.t[n_train_strides] - def_corr_traj.t[1])"
@info "Number of strides during testing : $n_test_strides"
@info "Training phase duration $(def_corr_traj.t[end] - def_corr_traj.t[n_train_strides])"


trajs = OrderedDict(
    "ZUPT only" => def_corr_traj[mask],
    "Static" => decoupled_stat_traj[mask],
    "HSGP" => hsgp1_corr_traj[mask],
)
final_rmse = OrderedDict(k => StrideGP.rmse(tr, gt_traj_aligned[step_seg][mask])[end] for (k, tr) in trajs)
foreach(((k, r),) -> @info(@sprintf("%-10s final RMSE %.3f m (%+.1f %% vs ZUPT only)", k, r, 100 * (r / final_rmse["ZUPT only"] - 1))), final_rmse)

fig_ori = StrideGP.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg][mask])
fig_xyz = StrideGP.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg][mask])

# Zoom on where the corrections diverge: the first strides after the model takes over
# (top row) and the end of the walk (bottom row), one column per correction, ground
# truth dashed underneath each.
n_first_strides = 20
n_last_strides = 20
fig_traj_panels = results_figure() do
    StrideGP.plot_trajectory_start_end_panels(
        trajs, gt_traj_aligned[step_seg][mask];
        train_ratio=train_ratio, n_first=n_first_strides, n_last=n_last_strides, markers=true,
        save_path=stamped(section, "trajectory2d_panels_$(data_key)_trial$(trial_id)"))
end

# These two take no save_path, so the figure is saved here instead — still inside
# results_figure(), so `save` is served by CairoMakie like every other results figure.
fig_dist = results_figure() do
    f = StrideGP.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])
    save(stamped(section, "distance_error_$(data_key)_trial$(trial_id)"), f)
    f
end
fig_rmse_hybrid = results_figure() do
    f = StrideGP.plot_position_rmse(trajs, gt_traj_aligned[step_seg], train_ratio; show_index_ticks=false)
    save(stamped(section, "rmse_$(data_key)_trial$(trial_id)"), f)
    f
end
# results_figure leaves CairoMakie active; restore GLMakie so later plots still open windows.
GLMakie.activate!()

fig_out = StrideGP.plot_regression_results(output_data, io_data["Decoupled HSGP"]["target"])

# The three channels the corrections actually estimate (Δz is left alone), on one stacked
# figure: target grey, Static wong yellow, HSGP wong green, a single shared legend. Keys
# have to be exactly "Static"/"HSGP" — that is what picks the colours.
fig_regr_panels = results_figure() do
    StrideGP.plot_regression_panels(
        OrderedDict(
            "Static" => io_data["Decoupled Static"]["prediction"],
            "HSGP" => io_data["Decoupled HSGP"]["prediction"],
        ),
        io_data["Decoupled HSGP"]["target"];
        save_path=stamped(regression_section, "regression_panels_$(data_key)_trial$(trial_id)"))
end
GLMakie.activate!()

## Test on second track
trial_id = 16
train_ratio = 0.1
posyaw_measurement_update = true
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = StrideGP.compute_aligned_ins_trajectory(
    data_dir_path, trial_id
)
noisy_gt_traj = StrideGP.add_gaussian_noise(gt_traj_aligned; pos_std=pos_std, att_std=att_std)

# Extract the aligned initial state from the trajectory
x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    StrideGP.matrix_to_euler(
        ins_traj_aligned.R_nb[:, :, 1]
    )
)
N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]

true_outputs = Dict{String,StrideGP.CorrectionIO}()
pred_outputs = Dict{String,StrideGP.CorrectionIO}()

io_data = OrderedDict()

default_corr = StrideGP.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Base"], _ = corr_filter(
    inertial_updated, sim_config_updated, noisy_gt_traj, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, posyaw_measurement_update=posyaw_measurement_update)

decoup_static_est = CORRECTORS[filter_tag].static(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels) # [:pos_1, :pos_2] ; corrected_channels=[:yaw]
zupt, step_seg, decoupled_stat_traj, io_data["Decoupled Static"], _ = corr_filter(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_static_est;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=decoup_stat_model, posyaw_measurement_update=posyaw_measurement_update)

decoup_hsgp_estmtr = CORRECTORS[filter_tag].hsgp(round(Int, N / 60); params=hsgp_p, corrected_channels=output_channels)
zupt, step_seg, hsgp1_corr_traj, io_data["Decoupled HSGP"], _ = corr_filter(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoup_hsgp_estmtr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE, init_model=hsgp_decoup_model, posyaw_measurement_update=posyaw_measurement_update)

input_data = OrderedDict{String,StrideGP.CorrectionIO}()
output_data = OrderedDict{String,StrideGP.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "ZUPT only" => def_corr_traj,
    "Static" => decoupled_stat_traj,
    "HSGP" => hsgp1_corr_traj,
)
final_rmse = OrderedDict(k => StrideGP.rmse(tr, gt_traj_aligned[step_seg])[end] for (k, tr) in trajs)
foreach(((k, r),) -> @info(@sprintf("%-10s final RMSE %.3f m (%+.1f %% vs ZUPT only)", k, r, 100 * (r / final_rmse["ZUPT only"] - 1))), final_rmse)

fig_ori = StrideGP.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = StrideGP.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
# results_figure() == CairoMakie + theme_ggplot2(), the theme every saved results figure
# uses. It has to be CairoMakie: saving SVG under GLMakie silently rasterises the figure.
fig = results_figure() do
    StrideGP.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg];
        segment=:test, train_ratio=train_ratio, show_heading=false, heading_stride=1,
        save_path=stamped(section, "trajectory2d_$(data_key)_trial$(trial_id)"))
end
fig_dist = results_figure() do
    f = StrideGP.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
    save(stamped(section, "distance_error_$(data_key)_trial$(trial_id)"), f)
    f
end
fig_rmse_hybrid = results_figure() do
    f = StrideGP.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
    save(stamped(section, "rmse_$(data_key)_trial$(trial_id)"), f)
    f
end
# results_figure leaves CairoMakie active; restore GLMakie so later plots still open windows.
GLMakie.activate!()
fig_out = StrideGP.plot_regression_results(output_data, io_data["Decoupled HSGP"]["target"])