include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 11
hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    3 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json"
)[hsgp_p_key]

# Load parameters with corresponding metatdata
hsgp_p, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)

data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision"
)[meta["data_key"]]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])

trial_id = meta["trial_id"]
m = hsgp_p.m
margin = meta["margin"]
train_ratio = 0.4

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)
## Extract the aligned initial state from the trajectory
x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(
        ins_traj_aligned.R_nb[:, :, 1]
    )
)
true_outputs = Dict{String,Any}()
pred_outputs = Dict{String,Any}()

# Run online correction
hsgp_p = HybridZuptInsJl.HsgpParameters(
    hsgp_p.hp, hsgp_p.d, 500, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)

zupt, hsgp_ins_traj, step_seg, true_outputs["hsgp"], pred_outputs["hsgp"] = HybridZuptInsJl.hybrid_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
)

zupt, classic_ins_traj, step_seg, true_outputs["model"], pred_outputs["model"] = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned;
    x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
)


step_trajs = OrderedDict(
    "model" => classic_ins_traj[segs],
    "model + online HSGP" => hsgp_ins_traj[segs],
)

trajs = OrderedDict(
    "model" => classic_ins_traj,
    "model + online HSGP" => hsgp_ins_traj,
)


fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(step_trajs, gt_traj_aligned[segs])
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned)

fig_2D = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(hsgp_ins_traj, gt_traj_aligned; samples=2000)
fig_3d = HybridZuptInsJl.plot_trajectory_3d(hsgp_ins_traj, gt_traj_aligned; samples=2000)
