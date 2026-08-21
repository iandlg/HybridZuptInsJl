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
trial_id = 5 # meta["trial_id"]
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
trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}()
varhist = Dict()
nis = Dict()

# Run online correction
train_ratio = 0.5
N = length(inertial)
n_train_cutoff = floor(Int, train_ratio * N)
gta = [n <= n_train_cutoff for n in 1:N]

# --- single vs two filter -----------------------------------------------
_, trajs["model"], _, _, _, _, _, _, _, d_base, q_base, x_base, P_base = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial, simdata, gt_traj, params; gt_available=gta, x_init=x_init, correct=false)
_, trajs["model + HSGP covupd"], _, true_outputs["model + HSGP covupd"], pred_outputs["model + HSGP covupd"], _, _, varhist["covupd"], nis["covupd"], d_on, q_on, x_on, P_on = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial, simdata, gt_traj, params; cov_update=true, gt_available=gta, feature_type=FEATURE_TYPE, x_init=x_init, ref_frame=FRAME)
_, trajs["model + HSGP nocovupd"], segs, true_outputs["model + HSGP nocovupd"], pred_outputs["model + HSGP nocovupd"], _, _, varhist["nocovupd"], nis["nocovupd"], d_off, q_off, x_off, P_off = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial, simdata, gt_traj, params; cov_update=false, gt_available=gta, feature_type=FEATURE_TYPE, x_init=x_init, ref_frame=FRAME)

# 1. NEES envelope
n_base = HybridZuptInsJl.nees_series(x_base, P_base, q_base, gt_traj)
n_on = HybridZuptInsJl.nees_series(x_on, P_on, q_on, gt_traj)
n_off = HybridZuptInsJl.nees_series(x_off, P_off, q_off, gt_traj)
@show mean(n_on.pos), HybridZuptInsJl.consistency_ratio(n_on.pos, n_on.lower, n_on.upper)
@show mean(n_off.pos), HybridZuptInsJl.consistency_ratio(n_off.pos, n_off.lower, n_off.upper)

fig_nees_pos = HybridZuptInsJl.plot_nees_comparison(
    OrderedDict("baseline" => n_base, "cov_update=true" => n_on, "cov_update=false" => n_off);
    block=:pos, title="NEES consistency (position)")

# 2. whiteness + NIS level
w_on = HybridZuptInsJl.whiteness_test(d_on; maxlag=15)
w_off = HybridZuptInsJl.whiteness_test(d_off; maxlag=15)
@show HybridZuptInsJl.nis_summary(d_on; dof=4), w_on.pvalue
@show HybridZuptInsJl.noise_state_correlation(d_on).rho
@show HybridZuptInsJl.noise_state_correlation(d_off).rho

fig_whiteness = HybridZuptInsJl.plot_innovation_whiteness(
    OrderedDict("cov_update=true" => w_on, "cov_update=false" => w_off))

fig_noise_state_corr = HybridZuptInsJl.plot_noise_state_correlation(
    OrderedDict(
        "cov_update=true" => HybridZuptInsJl.noise_state_correlation(d_on),
        "cov_update=false" => HybridZuptInsJl.noise_state_correlation(d_off)))

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

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(step_trajs, gt_traj[segs])
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj)
fig_dist = HybridZuptInsJl.plot_position_distance_error(step_trajs, gt_traj[segs])
is_step = [x ∈ segs for x in 1:N]
fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj, is_step)

fig_out = HybridZuptInsJl.plot_regression_results(pred_outputs, true_outputs["model + HSGP nocovupd"])
fig_traj_calib = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj)
fig_traj = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(step_trajs, gt_traj[segs]; show_heading=true, start=95, stop=105, heading_stride=1)
fig_var_nocovupd = HybridZuptInsJl.plot_channels(hcat(varhist["nocovupd"]...))
fig_var_covupd = HybridZuptInsJl.plot_channels(hcat(varhist["covupd"]...))
fig_nis_nocovupd = HybridZuptInsJl.plot_channels(hcat(nis["nocovupd"]...))
fig_nis_covupd = HybridZuptInsJl.plot_channels(hcat(nis["covupd"]...))

fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(step_trajs, gt_traj[segs])

# fig_in_gp = HybridZuptInsJl.plot_input_features(tr_input_gp)
# fig_in_hsgp = HybridZuptInsJl.plot_input_features(tr_input)

# fig_2D = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned; samples=1000000)
# fig_3d = HybridZuptInsJl.plot_trajectory_3d(hsgp_ins_traj, gt_traj_aligned; samples=15000)
# fig_betahist = HybridZuptInsJl.plot_beta_evolution(betahist)
