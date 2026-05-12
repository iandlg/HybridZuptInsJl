include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie
using JSON

data_dir = "data/angermann_high_precision"
trial_id = 15
hsgp_p, meta, _ = HybridZuptInsJl.from_json(
    HybridZuptInsJl.HsgpParameters, "out/3OfflineCorrection/HsgpResults/3d_body_hsgp_params.json")
data_dir = meta["data_dir"]
trial_id = meta["trial_id"]
train_ratio = 0.6
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])


imu = HybridZuptInsJl.InertialData(data_dir, trial_id)
gt = HybridZuptInsJl.Trajectory(data_dir, trial_id)
Δt = -0.05
gt = HybridZuptInsJl.Trajectory(
    (gt.t .+ Δt), gt.pos,
    gt.R_nb, gt.vel
)
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; inertial=imu, gt_traj=gt
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
zupt, hybrid_ins_traj, step_seg, y_train = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, correction=true, train_ratio=train_ratio
)

zupt, classic_ins_traj, step_seg, y_train = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_p;
    x_init=x_init, correction=false, train_ratio=train_ratio
)

step_trajs = Dict(
    "model" => classic_ins_traj[segs],
    "model + Online HSGP" => hybrid_ins_traj[segs]
)

fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(step_trajs, gt_traj_aligned[segs])