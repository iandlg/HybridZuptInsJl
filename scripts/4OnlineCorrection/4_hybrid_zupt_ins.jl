include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 22

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

FRAME = HybridZuptInsJl.HEADING # HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = 15 # meta["trial_id"]
m = hsgp_p.m
margin = meta["margin"]
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
N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]
gt_available[1:min(500, N)] .= false

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

# Run online correction
hsgp_p = HybridZuptInsJl.HsgpParameters(
    hsgp_p.hp, hsgp_p.d, 200, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)

# default_corr = HybridZuptInsJl.DefaultCorrector(round(Int, N / 60))
# zupt, zupt_ins_traj, step_seg, def_corr_traj = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME)

# static_corr = HybridZuptInsJl.StaticCorrector(round(Int, N / 60))
# zupt, zupt_ins_traj, step_seg, stat_corr_traj = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, gt_traj_aligned, static_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME)

splitHsgp_corr = HybridZuptInsJl.SplitHybridCorrector(round(Int, N / 60), hsgp_p, FEATURE_TYPE)
zupt, zupt_ins_traj, step_seg, hsgp1_corr_traj = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME)
trajs = OrderedDict(
    "zupt ins" => zupt_ins_traj[step_seg],
    "step_wise" => def_corr_traj,
    "static correction" => stat_corr_traj,
    "split hsgp correction" => hsgp1_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=18, stop=25)
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])