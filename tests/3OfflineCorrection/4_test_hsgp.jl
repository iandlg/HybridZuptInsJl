include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using JSON, Statistics
using OrderedCollections

hyp_key = 1
hyp_path = Dict{Int,String}(
    1 => "out/3OfflineCorrection/VariabilityResults/PY_ANG15_BODY_THREED_STEP.json",
    2 => "out/3OfflineCorrection/VariabilityResults/ANG15_BODY_THREED_STEP_2026-05-13T14:14:24.485.json"
)[hyp_key]

hp, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.SeHyperparams, hyp_path)
data_dir = "data/angermann_high_precision"
trial_id = 15
FRAME = HybridZuptInsJl.BODY
FEATURE_TYPE = HybridZuptInsJl.THREED_STEP
m = 1000
margin = 2.5

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)

fig_3d_traj = HybridZuptInsJl.plot_trajectory_3d(ins_traj_aligned[segs[1:20]], gt_traj_aligned[segs[1:20]])
fig_rmse = HybridZuptInsJl.plot_position_rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])

display(ins_traj_aligned.pos[:, 1])

true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs; ref_frame=FRAME, feature_type=FEATURE_TYPE)

gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections, hp;
    n_restarts_optimizer=0)

static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections, hp)

hsgp_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_hsgp_corrections, hp;
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

trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}(
    "model" => ins_traj_aligned[segs],
    "model + static" => static_traj,
    "model + GP" => gp_traj,
    "model + HSGP" => hsgp_traj
)
outputs = OrderedDict(
    "static" => static_corrections,
    "GP" => gp_corrections,
    "HSGP" => hsgp_corrections
)

# Show output Statistics
@info "Yaw correction standard deviation : $(std(true_outputs["yaw"]))"
@info "Yaw correction mean : $(mean(true_outputs["yaw"]))"

fig_rmse_hspg = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])
fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[segs])
fig_regression = HybridZuptInsJl.plot_regression_results(outputs, true_outputs)
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[segs])
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(ins_traj_aligned, gt_traj_aligned)