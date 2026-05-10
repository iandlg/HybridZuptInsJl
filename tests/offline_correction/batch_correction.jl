include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie
using JSON

data_dir = "data/angermann_high_precision"
imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, 15
)


hyper = JSON.parsefile("out/hyperparameters/py_hypers.json", HybridZuptInsJl.SeHyperparams)
@show typeof(hyper.yaw)


N_steps = length(segs)
@info "N_step = $N_steps"
true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs)

N_train = size(input_feature, 2)
@info "N_train = $N_train"

gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections, hyper;
    n_restarts_optimizer=5)

static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections, hyper)

true_output_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    true_outputs["yaw"],
    true_outputs["pos"],
    segs
)

gp_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    gp_corrections["yaw"],
    gp_corrections["pos"],
    segs
)

static_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    static_corrections["yaw"],
    static_corrections["pos"],
    segs
)

trajs = Dict{String,HybridZuptInsJl.Trajectory}(
    "model" => ins_traj_aligned[segs],
    "model + static" => static_traj,
    "model + GP" => gp_traj
)


outputs = Dict(
    "gp" => gp_corrections,
    "static" => static_corrections
)

# Create your figures
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[segs])
fig_rmse = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])
fig_output = HybridZuptInsJl.plot_regression_results(true_outputs, outputs)
