include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, JSON, OrderedCollections
import Dates

data_dirs = Dict{String,String}(
    "ANG" => "data/angermann_high_precision"
)
data_key = "ANG"


data_dir = data_dirs[data_key]
trial_id = 15
FRAME = HybridZuptInsJl.BODY
FEATURE_TYPE = HybridZuptInsJl.THREED_STEP
n_restarts_optimizer = 0

imu = HybridZuptInsJl.InertialData(data_dir, trial_id)
gt = HybridZuptInsJl.Trajectory(data_dir, trial_id)
Δt = -0.05
gt = HybridZuptInsJl.Trajectory(
    (gt.t .+ Δt),
    gt.pos,
    gt.R_nb,
    gt.vel
)
ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; inertial=imu, gt_traj=gt
)

hypvect = JSON.parsefile(
    "out/hyperparameters/fixed_jl_hypers_body_3d_list.json", Vector{HybridZuptInsJl.SeHyperparams})
hyper = hypvect[1]


true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs; ref_frame=FRAME, feature_type=FEATURE_TYPE)

N_train = size(input_feature, 2)
@info "N_train = $N_train"

gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections, hyper;
    n_restarts_optimizer=n_restarts_optimizer)

static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections, hyper)

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

static_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    static_corrections["yaw"],
    static_corrections["pos"],
    segs; ref_frame=FRAME
)

trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}(
    "model" => ins_traj_aligned[segs],
    "model + static" => static_traj,
    "model + GP" => gp_traj)

outputs = OrderedDict(
    "gp" => gp_corrections,
    "static" => static_corrections
)

# Create figures
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[segs]; samples=20)
fig_rmse = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])
fig_output = HybridZuptInsJl.plot_regression_results(outputs, true_outputs)


## Save things 
outdir = "out/3OfflineCorrection/OptimizationResults"
time = string(Dates.now())
filename = "$(data_key)$(trial_id)_$(FRAME)_$(FEATURE_TYPE)_$(time).json"

HybridZuptInsJl.to_json(joinpath(outdir, filename), hyperparameters;
    metadata=Dict{String,Any}(
        "data_dir" => data_dir,
        "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        "n_restarts_optimizer" => n_restarts_optimizer,
        "time_offset" => Δt)
)
