include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections, Statistics, Random

# --- Choose Parameters file
hsgp_p_key = 40

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    40 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_DT/ANG15_HEADING_TWOD_STEP_DT_2026-07-10T15:06:17.927.json"
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
normalize_y = true

trial_id = 15 # meta["trial_id"]
train_ratio = 0.45

## --- Load Data ---
train_ids = [1, 2, 4, 8, 9, 13, 14, 16]
test_ids = [3, 13, 15]
trial_ids = vcat(train_ids, test_ids)

dataset = HybridZuptInsJl.collect_dataset(
    data_dir, trial_ids;
    frame=FRAME,
    feature_type=FEATURE_TYPE,
)

# Guard against trials that failed to load
missing_train = filter(id -> !haskey(dataset, id), train_ids)
missing_test = filter(id -> !haskey(dataset, id), test_ids)
isempty(missing_train) || @warn "Missing train trials: $missing_train"
isempty(missing_test) || @warn "Missing test trials: $missing_test"
train_ids = filter(id -> haskey(dataset, id), train_ids)
test_ids = filter(id -> haskey(dataset, id), test_ids)

train_results = [dataset[id] for id in train_ids]
test_results = [dataset[id] for id in test_ids]

# Each result tuple: (target::CorrectionIO, input::CorrectionIO, corr_traj, gt_traj, step_seg)
train_out = HybridZuptInsJl.concatenate_io([res[1] for res in train_results])
train_in = HybridZuptInsJl.concatenate_io([res[2] for res in train_results])
test_out = HybridZuptInsJl.concatenate_io([res[1] for res in test_results])
test_in = HybridZuptInsJl.concatenate_io([res[2] for res in test_results])

## --- Fit stats on TRAIN ONLY, apply to both train and test ---
output_symbols = ["pos_1", "pos_2", "pos_3", "yaw"]
d = size(train_in.data, 1)

μ_x = vec(mean(train_in.data, dims=2))
σ_x = vec(std(train_in.data, dims=2))
train_in_norm = (train_in.data .- μ_x) ./ σ_x
test_in_norm = (test_in.data .- μ_x) ./ σ_x
train_in_stats = [μ_x, σ_x]

μ_y = normalize_y ? vec(mean(train_out.data, dims=2)) : fill(0.0, 4)
σ_y = normalize_y ? vec(std(train_out.data, dims=2)) : fill(1.0, 4)
train_out_norm = (train_out.data .- μ_y) ./ σ_y
train_out_stats = [μ_y, σ_y]

# Bounding box in normalized space (based on TRAIN input range)
xmin_norm = vec(minimum(train_in_norm, dims=2))
xmax_norm = vec(maximum(train_in_norm, dims=2))
pm = 0.5 * minimum(xmax_norm - xmin_norm)
LL_norm = [xmin_norm' .- pm; xmax_norm' .+ pm]
mid_norm = (LL_norm[1, :] .+ LL_norm[2, :]) ./ 2
Lvec_norm = (LL_norm[2, :] .- LL_norm[1, :]) ./ 2
@show Lvec_norm

## --- Optimize hyperparameters per output, predict on TEST ---
pred_data = similar(test_out.data)
pred_var = similar(test_out.data)
hyps = Dict{String,Any}()
rng = Random.Xoshiro(123)
for (idx, symb) in enumerate(output_symbols)
    noise_lower_orig = maximum(train_out.data_std[idx, :])
    noise_lower_norm = noise_lower_orig / σ_y[idx]
    default_lower, default_upper = 1e-6, 1e6
    lower = [noise_lower_norm; fill(default_lower, 3)]
    # lower = fill(default_lower, 4)
    upper = fill(default_upper, 4)

    # First pass: fit on train, predict on test to get β for uncertainty propagation
    _, _, theta, lik, _, _, per_dim_eigvals, β =
        HybridZuptInsJl.hsgp_regression(
            train_in_norm', train_out_norm[idx, :],
            test_in_norm', m;
            use_linear=false, LL=LL_norm, lower=lower, rng=rng, upper=upper, theta=[0.01, 0.01, 0.01, 1.0]
        )

    # Propagate train input uncertainty through the fitted GP derivative
    x_scaled_norm = train_in_norm' .- mid_norm'
    dfdx = zeros(size(x_scaled_norm, 1), d)
    for di in 1:d
        Phi_dx = HybridZuptInsJl.calc_eigenvectors_dx(x_scaled_norm, Lvec_norm, per_dim_eigvals, di)
        dfdx[:, di:di] = Phi_dx * β
    end
    train_in_std_norm = train_in.data_std[:, :] ./ σ_x
    var_x_norm = sum(dfdx' .^ 2 .* train_in_std_norm .^ 2; dims=1)

    # Refine noise lower bound with propagated input uncertainty, refit on train, predict on TEST
    noise_lower_norm += sqrt(maximum(var_x_norm))
    lower = [noise_lower_norm; fill(default_lower, 3)]
    # lower = fill(default_lower, 4)

    pred_norm, pred_var_norm, theta, lik, _, _, _, _ =
        HybridZuptInsJl.hsgp_regression(
            train_in_norm', train_out_norm[idx, :],
            test_in_norm', m;
            use_linear=false, LL=LL_norm, lower=lower, rng=rng, upper=upper,
            theta=[0.01, 0.01, 0.01, 1.0]
        )

    pred_data[idx, :] = pred_norm .* σ_y[idx] .+ μ_y[idx]
    pred_var[idx, :] = pred_var_norm .* σ_y[idx] .^ 2
    hyps[symb] = theta[1:3]
end

@info "Optimized yaw hyperparameters: " hyps["yaw"]

pred = HybridZuptInsJl.CorrectionIO(test_out.t, pred_data, sqrt.(pred_var))

# hyps["pos_1"] = hsgp_base.hp.pos_1
# hyps["pos_2"] = hsgp_base.hp.pos_2
# hyps["pos_3"] = hsgp_base.hp.pos_3

hsgp_opt = HybridZuptInsJl.HsgpParameters(
    HybridZuptInsJl.SeHyperparams(hyps), d, m, Lvec_norm;
    input_stats=train_in_stats, output_stats=train_out_stats, mid_norm=mid_norm
)

hsgp_base = HybridZuptInsJl.HsgpParameters(
    hsgp_base.hp, hsgp_base.d, m, hsgp_base.LL;
    input_stats=hsgp_base.input_stats,
    output_stats=hsgp_base.output_stats
)
hsgp_opt
fig_regr = HybridZuptInsJl.plot_regression_results(pred, test_out)
## --- Run Correction using both Hyper Parameter Sets ---
trial_id = 15
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

slamHsgp_corr_opt = HybridZuptInsJl.SlamCorrector(round(Int, N / 60), hsgp_opt)
_, _, slamHsgpOpt_corr_traj, io_data["SlamHsgp Opt"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr_opt;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

default_corr = HybridZuptInsJl.DefaultCorrector(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Default"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

tatic_corr = HybridZuptInsJl.StaticCorrector(round(Int, N / 60))
_, _, stat_corr_traj, io_data["Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, tatic_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

slamHsgp_corr_base = HybridZuptInsJl.SlamCorrector(round(Int, N / 60), hsgp_base)
_, _, slamHsgpBase_corr_traj, io_data["SlamHsgp Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr_base;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

# splitHsgp_corr = HybridZuptInsJl.SplitHybridCorrector(round(Int, N / 60), hsgp_base)
# _, _, hsgp1_corr_traj, io_data["SplitHsgp Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
#     inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
#     x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

input_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
input_data_norm = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
output_data_norm = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
residual_data = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()

for (method_name, io_dict) in io_data
    input_data["$method_name : Input"] = io_dict["input"]
    output_data["$method_name : Prediction"] = io_dict["prediction"]
    input_data_norm["$method_name : Input Norm"] = io_dict["input_norm"]
    output_data_norm["$method_name : Prediction Norm"] = io_dict["prediction_norm"]
    residual_data["$method_name : Residual"] = io_dict["residual"]
end

trajs = OrderedDict(
    "Base" => def_corr_traj,
    "Static" => stat_corr_traj,
    "SLAM Base" => slamHsgpBase_corr_traj,
    "SLAM Opt" => slamHsgpOpt_corr_traj,
    # "SPLIT Base" => hsgp1_corr_traj
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
fig_out_norm = HybridZuptInsJl.plot_regression_results(output_data_norm, io_data["SlamHsgp Opt"]["target_norm"])

fig_in = HybridZuptInsJl.plot_regression_results(input_data, io_data["Default"]["input"])
# display(GLMakie.Screen(), fig_in)
fig_in_norm = HybridZuptInsJl.plot_regression_results(input_data_norm, io_data["Default"]["input_norm"])
# display(GLMakie.Screen(), fig_in_norm)
fig_res = HybridZuptInsJl.plot_regression_results(residual_data)
display(hsgp_opt.input_stats)
display(hsgp_opt.output_stats)
## -- Save hyperparameters ---
using Dates
base_dir = "out/4OnlineCorrection/6_HypOpt"
combo_dir = joinpath(base_dir, data_key, string(FRAME) * "-" * string(FEATURE_TYPE))
mkpath(combo_dir)


time = string(Dates.now())
filename = "$(meta["data_key"])$(meta["trial_id"])_$(FRAME)_$(FEATURE_TYPE)_$(time).json"
HybridZuptInsJl.to_json(joinpath(combo_dir, filename), hsgp_opt;
    metadata=Dict(
        "data_key" => data_key,
        "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        "normalize_input" => true,
        "normalize_output" => normalize_y,
        "train_ids" => train_ids
    )
)