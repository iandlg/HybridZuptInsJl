include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames, Statistics, Random
import Optim, GaussianProcesses, GLMakie

# ── Config ────────────────────────────────────────────────────────────────────
data_key = "ANG2"
FRAME = HybridZuptInsJl.HEADING
FEATURE_TYPE = HybridZuptInsJl.TWOD_STEP_DT
n_restarts_optimizer = 8
log_kern_bounds = [[-3.0, -3.0], [3.0, 3.0]]
log_noise_bounds = [[-4.0], [2.0]]
method_key = "NM"
normalize_input = true
normalize_output = false
ard = false
train_frac = 0.8
n_folds = 10
rng_seed = 42

# HSGP parameters 
margin = 1.6
m = 300
z_thresh = 10.0

base_dir = "out/4OnlineCorrection/6_HypOpt"
# Create combined folder name: e.g., "BODY-TWOD_STEP_DT"
combo_dir = joinpath(base_dir, data_key, string(FRAME) * "-" * string(FEATURE_TYPE))
mkpath(combo_dir)

data_dir = Dict(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]
method = Dict{String,Any}(
    "CG" => Optim.ConjugateGradient(),
    "NM" => Optim.NelderMead(),
    "LBFGS" => Optim.LBFGS()
)[method_key]


# ── Load data ─────────────────────────────────────────────────────────────────
trial_ids = HybridZuptInsJl.list_trial_ids(data_dir)
train_id = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 16]
test_id = setdiff(trial_ids, train_id)

dataset = HybridZuptInsJl.collect_dataset(data_dir, trial_ids;
    frame=FRAME, feature_type=FEATURE_TYPE)

df = HybridZuptInsJl.io_dataframe(dataset)
df = transform(groupby(df, :trial_id), :t => (x -> length(unique(x))) => :trial_step_count)

HybridZuptInsJl.plot_channel_boxplots(df; show_trials=true, save_path=joinpath(combo_dir, "channel_io_stats.png"))
##
trial_id = 15
m = 500
normalize_x = true
normlize_y = false
train_ratio = 0.4
output, input, corr_traj, gt_traj, segs = dataset[trial_id]
output_symbols = ["pos_1", "pos_2", "pos_3", "yaw"]
d = size(input, 1)

pred_data = similar(output.data)
pred_std = similar(output.data_std)

hyps = Dict{String,Any}()

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

for (idx, symb) in enumerate(output_symbols)
    pred_data[idx, :], pred_std[idx, :], theta, lik, _ = HybridZuptInsJl.hsgp_regression(
        input.data', output.data[idx, :], input.data', m;
        use_linear=false, LL=LL
    )

    hyps[symb] = theta[1:3]

end
pred_std = sqrt.(pred_std)

pred = HybridZuptInsJl.CorrectionIO(output.t, pred_data, pred_std)
fig_regr = HybridZuptInsJl.plot_regression_results(pred, output)

hsgp_p = HybridZuptInsJl.HsgpParameters(
    HybridZuptInsJl.SeHyperparams(hyps), d, m, Lvec; input_stats=input_stats)

## 
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
gt_available[1:min(500, N)] .= true

io_data = OrderedDict()

splitHsgp_corr = HybridZuptInsJl.SplitHybridCorrector(round(Int, N / 60), hsgp_p, FEATURE_TYPE)
zupt, step_seg, hsgp1_corr_traj, io_data["SplitHsgp"] = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name"] = io_dict["target"]
end

trajs = OrderedDict(
    "zupt ins" => corr_traj,
    "split hsgp correction" => hsgp1_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=18, stop=25)
fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg])
fig_out = HybridZuptInsJl.plot_regression_results(io_data["SplitHsgp"]["prediction"], output)

