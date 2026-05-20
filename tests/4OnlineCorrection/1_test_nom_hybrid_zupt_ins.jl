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
    hsgp_p.hp, hsgp_p.d, 1000, hsgp_p.LL;
    input_stats=hsgp_p.input_stats,
    output_stats=hsgp_p.output_stats
)
hsgp_corrector = HybridZuptInsJl.HsgpCorrector(hsgp_p)
zupt, hsgp_ins_traj, step_seg, true_outputs["hsgp"], pred_outputs["hsgp"] = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned;
    corrector=hsgp_corrector, x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
)

# Run static mean correction
static_corrector = HybridZuptInsJl.StaticMeanCorrector()
zupt, static_ins_traj, step_seg, true_outputs["static"], pred_outputs["static"] = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned;
    corrector=static_corrector, x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
)


zupt, classic_ins_traj, step_seg, true_outputs["model"], pred_outputs["model"] = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned;
    x_init=x_init, train_ratio=train_ratio, feature_type=FEATURE_TYPE
)

gp_corrector = HybridZuptInsJl.ExactGpCorrector(hsgp_p)
zupt, gp_ins_traj, step_seg, true_outputs["gp"], pred_outputs["gp"] = HybridZuptInsJl.hybrid_nominal_zupt_aided_ins(
    inertial_updated, sim_config_updated, gt_traj_aligned;
    x_init=x_init, train_ratio=train_ratio, corrector=gp_corrector, feature_type=FEATURE_TYPE
)


step_trajs = OrderedDict(
    "model" => classic_ins_traj[segs],
    "model + static" => static_ins_traj[segs],
    "model + GP" => gp_ins_traj[segs],
    "model + online HSGP" => hsgp_ins_traj[segs],
)

training_outputs = OrderedDict{String,Dict{String,VecOrMat{Float64}}}(
    "model + static" => true_outputs["static"],
    "model + GP" => true_outputs["gp"],
    "model + HSGP" => true_outputs["hsgp"]
)
pred_outputs = OrderedDict{String,Dict{String,VecOrMat{Float64}}}(
    "model + static" => pred_outputs["static"],
    "model + GP" => pred_outputs["gp"],
    "model + HSGP" => pred_outputs["hsgp"]
)

fig_train_out = HybridZuptInsJl.plot_regression_results(training_outputs, true_outputs["model"])
fig_pred_out = HybridZuptInsJl.plot_regression_results_no_rmse(pred_outputs, true_outputs["model"])
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(step_trajs, gt_traj_aligned[segs])
