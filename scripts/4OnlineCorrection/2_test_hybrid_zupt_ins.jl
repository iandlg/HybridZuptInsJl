include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 30
hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    12 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-26T13:06:11.411.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    100 => "out/3OfflineCorrection/6_HypOpt/ANG2/BODY-TWOD_STEP_DT/ANG2_BODY_TWOD_STEP_DT_best_fold8_2026-06-08T10:50:24.741.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
data_key = "ANG2" # "ANG2" # meta["data_key"]

data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = 15 # meta["trial_id"]
m = hsgp_p.m
margin = meta["margin"]
# z_thresh = meta["z_thresh"]
train_ratio = 0.4

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
true_outputs = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()

# Run online correction
hsgp_p = HybridZuptInsJl.HsgpParameters(
    hsgp_p.hp, hsgp_p.d, 200, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)

N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]

trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}()
zupt, trajs["model"], step_seg, _, _, _ = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, gt_available=gt_available, feature_type=FEATURE_TYPE, correct=false
)

# Run static mean correction
# static_corrector = HybridZuptInsJl.StaticMeanCorrector()
# zupt, static_ins_traj, step_seg, true_outputs["model + static"], pred_outputs["model + static"], _, _, _ = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
#     inertial_updated, sim_config_updated, gt_traj_aligned;
#     corrector=static_corrector, x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
# )

# zupt, gp_ins_traj, step_seg, true_outputs["model + GP"], pred_outputs["model + GP"], tr_input_gp = HybridZuptInsJl.hybrid_zupt_aided_ins_gp(
#     inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
#     x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
# )

zupt, trajs["model + HSGP covupd"], step_seg, true_outputs["model + HSGP covupd"], pred_outputs["model + HSGP covupd"], betahist, tr_input, var_hist_covupd, pred_nis_covupd = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, gt_available=gt_available, feature_type=FEATURE_TYPE, cov_update=true
)

zupt, trajs["model + HSGP nocovupd"], step_seg, true_outputs["model + HSGP nocovupd"], pred_outputs["model + HSGP nocovupd"], betahist, tr_input, var_hist_nocovupd, pred_nis_nocovupd = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, gt_available=gt_available, feature_type=FEATURE_TYPE, cov_update=false
)
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
