include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections, Statistics

# Load GP parameters
hsgp_p_key = 42
hsgp_p_path = Dict{Int,String}(
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    100 => "out/3OfflineCorrection/6_HypOpt/ANG2/BODY-TWOD_STEP_DT/ANG2_BODY_TWOD_STEP_DT_best_fold8_2026-06-08T10:50:24.741.json",
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
)[hsgp_p_key]

params, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
params = HybridZuptInsJl.basecopy(params; new_m=200)
margin = meta["normalization"]["margin"]
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

# Load track
data_key = "DCSC" # "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]
trial_id = 3 # meta["trial_id"]
ins_traj_aligned, gt_traj, zupt, segs, inertial, simdata = HybridZuptInsJl.compute_aligned_ins_trajectory(
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
true_outputs = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()

# Run online correction
train_ratio = 0.5
N = length(inertial)
n_train_cutoff = floor(Int, train_ratio * N)
gta = [n <= n_train_cutoff for n in 1:N]

trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}()
# zupt, trajs["model"], step_seg, _, _, _ = HybridZuptInsJl.hybrid_zupt_aided_ins(
#     inertial, simdata, gt_traj, params;
#     x_init=x_init, gt_available=gta, feature_type=FEATURE_TYPE, correct=false
# )


# zupt, trajs["model + HSGP covupd"], step_seg, true_outputs["model + HSGP covupd"], pred_outputs["model + HSGP covupd"], betahist, tr_input, var_hist_covupd, pred_nis_covupd = HybridZuptInsJl.hybrid_zupt_aided_ins(
#     inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
#     x_init=x_init, gt_available=gt_available, feature_type=FEATURE_TYPE, cov_update=true
# )

# zupt, trajs["model + HSGP nocovupd"], step_seg, true_outputs["model + HSGP nocovupd"], pred_outputs["model + HSGP nocovupd"], betahist, tr_input, var_hist_nocovupd, pred_nis_nocovupd = HybridZuptInsJl.hybrid_zupt_aided_ins(
#     inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
#     x_init=x_init, gt_available=gt_available, feature_type=FEATURE_TYPE, cov_update=false
# )

# --- single vs two filter -----------------------------------------------
_, _, _, _, _, _, _, _, _, d_on, q_on, x_on, P_on = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial, simdata, gt_traj, params; cov_update=true, gt_available=gta, feature_type=FEATURE_TYPE, x_init=x_init, ref_frame=FRAME)
_, _, _, _, _, _, _, _, _, d_off, q_off, x_off, P_off = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial, simdata, gt_traj, params; cov_update=false, gt_available=gta, feature_type=FEATURE_TYPE, x_init=x_init, ref_frame=FRAME)

# 1. NEES envelope
n_on = HybridZuptInsJl.nees_series(x_on, P_on, q_on, gt_traj)
n_off = HybridZuptInsJl.nees_series(x_off, P_off, q_off, gt_traj)
@show mean(n_on.pos), HybridZuptInsJl.consistency_ratio(n_on.pos, n_on.lower, n_on.upper)
@show mean(n_off.pos), HybridZuptInsJl.consistency_ratio(n_off.pos, n_off.lower, n_off.upper)

# 2. whiteness + NIS level
w = HybridZuptInsJl.whiteness_test(d_on; maxlag=15)
@show HybridZuptInsJl.nis_summary(d_on; dof=4), w.pvalue
@show HybridZuptInsJl.noise_state_correlation(d_on).rho

# 3. inflation sweep
HybridZuptInsJl.print_sweep(HybridZuptInsJl.inflation_sweep(inertial, simdata, gt_traj, params; gt_available=gta))

# 4. oracle control
_, _, _, _, _, _, _, _, _, d_or, q_or, x_or, P_or = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial, simdata, gt_traj, params;
    cov_update=true, measurement_source=:oracle_noisy, gt_available=gta)
##
step_trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}()
for (name, traj) in trajs
    step_trajs[name] = traj[segs]
end

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(step_trajs, gt_traj_aligned[step_seg])
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned)
fig_dist = HybridZuptInsJl.plot_position_distance_error(step_trajs, gt_traj_aligned[segs])
is_step = [x ∈ segs for x in 1:N]
fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned, is_step)

fig_out = HybridZuptInsJl.plot_regression_results(pred_outputs, true_outputs["model + HSGP nocovupd"])
fig_traj_calib = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned)
fig_traj = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(step_trajs, gt_traj_aligned[step_seg]; show_heading=true, start=95, stop=105, heading_stride=1)
fig_var_nocovupd = HybridZuptInsJl.plot_channels(hcat(var_hist_nocovupd...))
fig_var_covupd = HybridZuptInsJl.plot_channels(hcat(var_hist_covupd...))
fig_nis_nocovupd = HybridZuptInsJl.plot_channels(hcat(pred_nis_nocovupd...))
fig_nis_covupd = HybridZuptInsJl.plot_channels(hcat(pred_nis_covupd...))

display(GLMakie.Screen(), fig_var_nocovupd)
display(GLMakie.Screen(), fig_var_covupd)
display(GLMakie.Screen(), fig_nis_covupd)
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(step_trajs, gt_traj_aligned[segs])

# fig_in_gp = HybridZuptInsJl.plot_input_features(tr_input_gp)
# fig_in_hsgp = HybridZuptInsJl.plot_input_features(tr_input)

# fig_2D = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned; samples=1000000)
# fig_3d = HybridZuptInsJl.plot_trajectory_3d(hsgp_ins_traj, gt_traj_aligned; samples=15000)
# fig_betahist = HybridZuptInsJl.plot_beta_evolution(betahist)
