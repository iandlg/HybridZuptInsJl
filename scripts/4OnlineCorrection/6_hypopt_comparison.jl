include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections

# --- Choose Parameters file
hsgp_p_key = 30

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json"
)[hsgp_p_key]

# --- Load parameters with corresponding metatdata
hsgp_base, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)

data_key = "ANG2" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])
m = 200
margin = meta["margin"]

trial_id = 15 # meta["trial_id"]
train_ratio = 0.45

## --- Load Data ---
res = HybridZuptInsJl.collect_trial_io_online(data_dir, trial_id; frame=FRAME, feature_type=FEATURE_TYPE)
@assert !isnothing(res)
target, input, _, _, seg = res

## --- Optimize Hyper Parameters ---
output_symbols = ["pos_1", "pos_2", "pos_3", "yaw"]
d = size(input.data, 1)

xmin = minimum(input.data', dims=1)[:]
xmax = maximum(input.data', dims=1)[:]

pm = 0.1 * minimum(xmax - xmin)
LL = [xmin' .- pm; xmax' .+ pm]

mid = (LL[1, :] .+ LL[2, :]) ./ 2
Lvec = (LL[2, :] .- LL[1, :]) ./ 2
@show Lvec
x = input.data' .- mid'
xt = input.data' .- mid'

input_stats = [
    mid, fill(1.0, d)
]
pred_data = similar(target.data)
pred_std = similar(target.data)
hyps = Dict{String,Any}()

for (idx, symb) in enumerate(output_symbols)
    pred_data[idx, :], pred_std[idx, :], theta, lik, _ = HybridZuptInsJl.hsgp_regression(
        input.data', target.data[idx, :], input.data', m;
        use_linear=false, LL=LL
    )

    hyps[symb] = theta[1:3]
end
pred_std = sqrt.(pred_std)

pred = HybridZuptInsJl.CorrectionIO(target.t, pred_data, pred_std)
fig_regr = HybridZuptInsJl.plot_regression_results(pred, target)

hsgp_opt = HybridZuptInsJl.HsgpParameters(
    HybridZuptInsJl.SeHyperparams(hyps), d, m, Lvec; input_stats=input_stats)

hsgp_base = HybridZuptInsJl.HsgpParameters(
    hsgp_base.hp, hsgp_base.d, m, hsgp_base.LL;
    input_stats=hsgp_base.input_stats,
    output_stats=hsgp_base.output_stats
)


## --- Run Correction using both Hyper Parameter Sets ---
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)

# Extract the aligned initial state from the trajectory
x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(
        ins_traj_aligned.R_nb[:, :, 1]
    )
)
N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]

true_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()
pred_outputs = Dict{String,HybridZuptInsJl.CorrectionIO}()

io_data = OrderedDict()

default_corr = HybridZuptInsJl.DefaultCorrector(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Default"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

tatic_corr2 = HybridZuptInsJl.StaticCorrectorV2(round(Int, N / 60))
_, _, stat_corr_traj2, io_data["Static2"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, tatic_corr2;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

slamHsgp_corr_base = HybridZuptInsJl.SlamCorrector(round(Int, N / 60), hsgp_base)
_, _, slamHsgpBase_corr_traj, io_data["SlamHsgp Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr_base;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

slamHsgp_corr_opt = HybridZuptInsJl.SlamCorrector(round(Int, N / 60), hsgp_opt)
_, _, slamHsgpOpt_corr_traj, io_data["SlamHsgp Opt"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr_opt;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
end

trajs = OrderedDict(
    "Base" => def_corr_traj,
    "Static" => stat_corr_traj2,
    "SLAM Base" => slamHsgpBase_corr_traj,
    "SLAM Opt" => slamHsgpOpt_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=1, stop=10, show_heading=true, heading_stride=1)
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg], train_ratio; show_index_ticks=true)
end
with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end

fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Default"]["target"])
fig_in = HybridZuptInsJl.plot_regression_results(input_data, io_data["Default"]["input"])
