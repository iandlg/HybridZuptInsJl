include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;


hp = JSON.parsefile("out/3OfflineCorrection/VariabilityResults/nn_outlier.json", HybridZuptInsJl.SeHyperparams)
data_dir = "data/angermann_high_precision"
trial_id = 15
FRAME = HybridZuptInsJl.BODY

imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(
    (gt.t .+ -0.05),
    gt.pos,
    gt.R_nb,
    gt.vel
)

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; inertial=imu, gt_traj=gt
)

true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs; ref_frame=FRAME)

gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections, hp;
    n_restarts_optimizer=0)

static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections, hp)

hsgp_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_hsgp_corrections, hp;
    m=25, margin=2.0
)


true_output_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    true_outputs["yaw"],
    true_outputs["pos"],
    segs; ref_frame=FRAME
)

gp_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    gp_corrections["yaw"],
    gp_corrections["pos"],
    segs; ref_frame=FRAME
)

hsgp_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    hsgp_corrections["yaw"],
    hsgp_corrections["pos"],
    segs; ref_frame=FRAME
)

static_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    static_corrections["yaw"],
    static_corrections["pos"],
    segs; ref_frame=FRAME
)

trajs = Dict{String,HybridZuptInsJl.Trajectory}(
    "model" => ins_traj_aligned[segs],
    "model + static" => static_traj,
    "model + GP" => gp_traj,
    "model + HSGP" => hsgp_traj
)
fig_rmse = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])
