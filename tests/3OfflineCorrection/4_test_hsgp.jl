include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using JSON, Statistics

hsgp_hp = JSON.parsefile("out/3OfflineCorrection/VariabilityResults/handtune_hsgp.json", HybridZuptInsJl.SeHyperparams)
gp_hp = JSON.parsefile("out/3OfflineCorrection/VariabilityResults/nn_outlier.json", HybridZuptInsJl.SeHyperparams)
data_dir = "data/angermann_high_precision"
trial_id = 15
FRAME = HybridZuptInsJl.BODY
FEATURE_TYPE = HybridZuptInsJl.THREED_STEP
m = 500
margin = 2.5

imu = HybridZuptInsJl.InertialData(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(data_dir, 15)
gt = HybridZuptInsJl.Trajectory(
    (gt.t .- 0.05),
    gt.pos,
    gt.R_nb,
    gt.vel
)

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; inertial=imu, gt_traj=gt
)

true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs; ref_frame=FRAME, feature_type=FEATURE_TYPE)

gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections, gp_hp;
    n_restarts_optimizer=0)

static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections, gp_hp)

hsgp_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_hsgp_corrections, hsgp_hp;
    m=m, margin=margin
)

## Save hsgp parameters
μ = mean(input_feature, dims=2)[:]
σ = std(input_feature, dims=2)[:]
feature_scaled = (input_feature .- μ) ./ σ
LL = margin * maximum(abs, feature_scaled, dims=2)[:]
p = HybridZuptInsJl.HsgpParameters(
    hsgp_hp, size(input_feature, 1), m, σ, μ, LL
)
HybridZuptInsJl.to_json("out/3OfflineCorrection/HsgpResults/3d_body_hsgp_params.json", p;
    metadata=Dict(
        "data_dir" => data_dir,
        "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        "margin" => margin
    )
)

## Apply corrections
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
outputs = Dict(
    "static" => static_corrections,
    "GP" => gp_corrections,
    "HSGP" => hsgp_corrections
)

# Show output Statistics
@info "Yaw correction standard deviation : $(std(true_outputs["yaw"]))"
@info "Yaw correction mean : $(mean(true_outputs["yaw"]))"

fig_rmse_hspg = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])
fig_regression = HybridZuptInsJl.plot_regression_results(outputs, true_outputs)
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[segs])
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(ins_traj_aligned, gt_traj_aligned)