include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, JSON, OrderedCollections
import Dates, Optim

data_key = "ANG2"
trial_id = 15
FRAME = HybridZuptInsJl.BODY
FEATURE_TYPE = HybridZuptInsJl.THREED_STEP
n_restarts_optimizer = 1
kern_lo = [-3.0, -3.0]
kern_hi = [3.0, 3.0]
log_kern_bounds = [kern_lo, kern_hi]
log_noise_bounds = [[-4.0], [2.0]]
method_key = "NM"
normalize_input = true
normalize_output = true
ard = false

data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]
method = Dict{String,Any}(
    "CG" => Optim.ConjugateGradient(),
    "NM" => Optim.NelderMead(),
    "LBFGS" => Optim.LBFGS()
)[method_key]

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)

true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs; ref_frame=FRAME, feature_type=FEATURE_TYPE
)

gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections;
    n_restarts_optimizer=n_restarts_optimizer,
    log_kern_bounds=log_kern_bounds,
    log_noise_bounds=log_noise_bounds,
    method=method,
    normalize_x=normalize_input,
    normalize_y=normalize_output,
    ard=ard
)

static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections)

true_output_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    true_outputs,
    segs; ref_frame=FRAME
)

gp_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    gp_corrections,
    segs; ref_frame=FRAME
)

static_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    static_corrections,
    segs; ref_frame=FRAME
)

fig_stat = HybridZuptInsJl.plot_trajectory_xyz_euler(static_traj)

trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}(
    "model" => ins_traj_aligned[segs],
    "model + static" => static_traj,
    "model + GP" => gp_traj
)

outputs = OrderedDict(
    "gp" => gp_corrections,
    "static" => static_corrections
)

rmses = OrderedDict{String,Float64}(
    "model" => HybridZuptInsJl.rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])[end],
    "model + static" => HybridZuptInsJl.rmse(static_traj, gt_traj_aligned[segs])[end],
    "model + GP" => HybridZuptInsJl.rmse(gp_traj, gt_traj_aligned[segs])[end],
)

# Create figures
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[segs]; samples=20)
fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[segs])
fig_output = HybridZuptInsJl.plot_regression_results(outputs, true_outputs)
fig_rmse = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])


## Save things 
outdir = "out/3OfflineCorrection/1_OptimizationResults"
time = string(Dates.now())
filename = "$(data_key)$(trial_id)_$(FRAME)_$(FEATURE_TYPE)_$(time).json"

HybridZuptInsJl.to_json(joinpath(outdir, filename), hyperparameters;
    metadata=Dict{String,Any}(
        "data_key" => data_key,
        "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        "n_restarts_optimizer" => n_restarts_optimizer,
        "log_kern_bounds" => log_kern_bounds,
        "log_noise_bounds" => log_noise_bounds,
        "rmse" => rmses,
        "method" => method_key,
        "normalize_input" => normalize_input,
        "normalize_output" => normalize_output,
        "ard" => ard
    )
)
