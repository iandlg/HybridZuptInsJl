include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using Statistics
using OrderedCollections, Dates

hyp_key = 100
hyp_path = Dict{Int,String}(
    10 => "out/3OfflineCorrection/2_VariabilityResults/PY_ANG15_BODY_THREED_STEP.json",
    20 => "out/3OfflineCorrection/2_VariabilityResults/ANG15_BODY_THREED_STEP_2026-05-13T14:14:24.485.json",
    30 => "out/3OfflineCorrection/2_VariabilityResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:06:28.867.json",
    31 => "out/3OfflineCorrection/2_VariabilityResults/ANG215_BODY_THREED_STEP_2026-06-06T17:22:12.377.json",
    40 => "out/3OfflineCorrection/2_VariabilityResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:03.913.json",
    41 => "out/3OfflineCorrection/2_VariabilityResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-22T11:18:28.679.json",
    100 => "out/3OfflineCorrection/6_HypOpt/ANG2/BODY-TWOD_STEP_DT/ANG2_BODY_TWOD_STEP_DT_best_fold3_2026-06-07T12:30:49.054.json",
)[hyp_key]

hp, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.SeHyperparams, hyp_path)
data_key = meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]
normalize_input = meta["normalize_input"]
normalize_output = meta["normalize_output"]
FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])
trial_id = 13
m = 200
margin = 1.5
z_thresh = 10.0

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id; sim_config=HybridZuptInsJl.InsConfig(calibration_distance_m=2.0)
)

fig_3d_traj = HybridZuptInsJl.plot_trajectory_3d(ins_traj_aligned[segs[1:20]], gt_traj_aligned[segs[1:20]])
fig_rmse = HybridZuptInsJl.plot_position_rmse(ins_traj_aligned[segs], gt_traj_aligned[segs])

true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
    ins_traj_aligned, gt_traj_aligned, segs; ref_frame=FRAME, feature_type=FEATURE_TYPE)

@info "---- Computing Static Corrections ----"
static_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_static_corrections, hp)

@info "---- Computing GP Corrections ----"
gp_corrections, hyperparameters = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_gp_corrections, hp;
    n_restarts_optimizer=0,
    normalize_x=normalize_input,
    normalize_y=normalize_output,
    ard=get(meta, "ard", false)
)
@info "---- Computing HSGP Corrections ----"
hsgp_corrections, _ = HybridZuptInsJl.compute_corrections(
    input_feature, true_outputs, HybridZuptInsJl.compute_hsgp_corrections, hp;
    m=m, margin=margin, feature_type=FEATURE_TYPE, normalize_x=normalize_input,
    normalize_y=normalize_output)

# Apply corrections
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

hsgp_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    hsgp_corrections,
    segs; ref_frame=FRAME
)

static_traj = HybridZuptInsJl.apply_corrections(
    ins_traj_aligned,
    static_corrections,
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
@info "Yaw correction standard deviation : $(std(true_outputs.data))"
@info "Yaw correction mean : $(mean(true_outputs.data))"

fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[segs])
fig_regression = HybridZuptInsJl.plot_regression_results(outputs, true_outputs)
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[segs])
fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(ins_traj_aligned, gt_traj_aligned)
fig_rmse_hspg = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[segs])


## Save hsgp parameters
outdir = "out/3OfflineCorrection/3_HsgpResults"
time = string(Dates.now())
filename = "$(meta["data_key"])$(meta["trial_id"])_$(FRAME)_$(FEATURE_TYPE)_$(time).json"

d = size(input_feature, 1)

input_stats = normalize_input ? [
    mean(input_feature, dims=2)[:],
    std(input_feature, dims=2)[:]
] : [fill(0.0, d), fill(1.0, d)]

output_stats = normalize_output ? [
    mean(true_outputs.data, dims=2)[:],
    std(true_outputs.data, dims=2)[:]
] : [fill(0.0, 4), fill(1.0, 4)]

feature_scaled = (input_feature .- input_stats[1]) ./ input_stats[2]
# LL = margin * maximum(abs, feature_scaled, dims=2)[:]

mask = all(feature_scaled .<= z_thresh, dims=1)[:]   # (n,)
LL = margin * maximum(abs, feature_scaled[:, mask], dims=2)[:] # (d,)

p = HybridZuptInsJl.HsgpParameters(
    hp, size(input_feature, 1), m, LL;
    input_stats=input_stats, output_stats=output_stats
)
HybridZuptInsJl.to_json(joinpath(outdir, filename), p;
    metadata=Dict(
        "data_key" => meta["data_key"],
        "hyp_path" => hyp_path,
        "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        "margin" => margin,
        "normalize_input" => normalize_input,
        "normalize_output" => normalize_output,
        "z_thresh" => z_thresh
    )
)