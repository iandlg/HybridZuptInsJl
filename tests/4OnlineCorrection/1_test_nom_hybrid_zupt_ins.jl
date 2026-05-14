include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# Choose Parameters file
hsgp_p_key = 1
hsgp_p_path = Dict{Int,String}(
    1 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-14T12:01:36.700.json"
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

@info "First position from ZUPT aided INS : $(ins_traj_aligned.pos[:,1])"

## Extract the aligned initial state from the trajectory
x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(
        ins_traj_aligned.R_nb[:, :, 1]
    )
)

# Run online correction
hsgp_corrector = HybridZuptInsJl.HsgpCorrector(hsgp_p)
zupt, hybrid_ins_traj, step_seg, y_train = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned;
    corrector=hsgp_corrector, x_init=x_init, train_ratio=train_ratio
)

zupt, classic_ins_traj, step_seg, y_train = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, correction=false, train_ratio=train_ratio
)

step_trajs = OrderedDict(
    "model" => classic_ins_traj[segs],
    "model + online HSGP" => hybrid_ins_traj[segs]
)

fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(step_trajs, gt_traj_aligned[segs])