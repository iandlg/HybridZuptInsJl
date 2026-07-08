include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections, Statistics

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
normalize_y = false

trial_id = 15 # meta["trial_id"]
train_ratio = 0.45

## --- Load Data ---
res = HybridZuptInsJl.collect_trial_io_online(data_dir, trial_id; frame=FRAME, feature_type=FEATURE_TYPE)
@assert !isnothing(res)
target, input, _, _, seg = res

## --- Optimize Hyper Parameters ---
output_symbols = ["pos_1", "pos_2", "pos_3", "yaw"]
d = size(input.data, 1)
hsgp_train_ratio = 0.6
N = length(input)
n_train_cutoff = floor(Int, hsgp_train_ratio * N)

# --- Normalize inputs ---
μ_x = vec(mean(input.data[:, 1:n_train_cutoff], dims=2))
σ_x = vec(std(input.data[:, 1:n_train_cutoff], dims=2))
input_norm = (input.data .- μ_x) ./ σ_x

# Bounding box in normalized space
xmin_norm = vec(minimum(input_norm[:, 1:n_train_cutoff], dims=2))
xmax_norm = vec(maximum(input_norm[:, 1:n_train_cutoff], dims=2))
pm = 0.5 * minimum(xmax_norm - xmin_norm)
LL_norm = [xmin_norm' .- pm; xmax_norm' .+ pm]

mid_norm = (LL_norm[1, :] .+ LL_norm[2, :]) ./ 2
Lvec_norm = (LL_norm[2, :] .- LL_norm[1, :]) ./ 2
@show Lvec_norm

input_stats = [μ_x, σ_x]   # mean and std for input

# --- Normalize outputs ---

μ_y = normalize_y ? vec(mean(target.data, dims=2)) : fill(0.0, 4)
σ_y = normalize_y ? vec(std(target.data, dims=2)) : fill(1.0, 4)
target_norm = (target.data .- μ_y) ./ σ_y
output_stats = [μ_y, σ_y]

pred_data = similar(target.data[:, (n_train_cutoff+1):end])
pred_var = similar(target.data[:, (n_train_cutoff+1):end])
hyps = Dict{String,Any}()

for (idx, symb) in enumerate(output_symbols)
    # Noise lower bound (original -> normalized)
    noise_lower_orig = max(target.data_std[idx, 1:n_train_cutoff]...)
    noise_lower_norm = noise_lower_orig / σ_y[idx]
    if symb == "yaw"
        noise_lower_norm += 0.1
    end
    @info "Lower noise bound for $symb : σn ≥ $noise_lower_orig (original), $noise_lower_norm (normalized)"
    default_lower = 1e-9
    lower = [noise_lower_norm; 0.1; fill(default_lower, 2)]
    @show lower

    # First fit in normalized space
    pred_norm, pred_var_norm, theta, lik, _, mid_returned, per_dim_eigvals, β =
        HybridZuptInsJl.hsgp_regression(
            input_norm[:, 1:n_train_cutoff]', target_norm[idx, 1:n_train_cutoff],
            input_norm[:, (n_train_cutoff+1):end]', m;
            use_linear=false, LL=LL_norm, lower=lower
        )

    # Input uncertainty in normalized space
    x_scaled_norm = input_norm[:, 1:n_train_cutoff]' .- mid_norm'
    dfdx = zeros(size(x_scaled_norm, 1), d)
    for di in 1:d
        Phi_dx = HybridZuptInsJl.calc_eigenvectors_dx(
            x_scaled_norm, Lvec_norm, per_dim_eigvals, di
        )
        dfdx[:, di:di] = Phi_dx * β
    end
    input_std_norm = input.data_std[:, 1:n_train_cutoff] ./ σ_x
    var_x_norm = dfdx' .^ 2 .* input_std_norm .^ 2
    var_x_norm = sum(var_x_norm; dims=1)
    @show max(var_x_norm...)

    # Update noise bound with input uncertainty and re‑run
    noise_lower_norm += sqrt(max(var_x_norm...))
    lower = [noise_lower_norm; fill(default_lower, 3)]
    @show lower

    pred_norm, pred_var_norm, theta, lik, _, _, _, _ =
        HybridZuptInsJl.hsgp_regression(
            input_norm[:, 1:n_train_cutoff]', target_norm[idx, 1:n_train_cutoff],
            input_norm[:, (n_train_cutoff+1):end]', m;
            use_linear=false, LL=LL_norm, lower=lower
        )

    # Unnormalize predictions
    pred_data[idx, :] = pred_norm * σ_y[idx] .+ μ_y[idx]
    pred_var[idx, :] = pred_var_norm * σ_y[idx] .^ 2

    hyps[symb] = theta[1:3]
end

@info "Optimized yaw hyper parameters : " hyps["yaw"]
@info "Base yaw hyper parameters : " hsgp_base.hp.yaw
pred = HybridZuptInsJl.CorrectionIO(
    target.t[(n_train_cutoff+1):end], pred_data, sqrt.(pred_var)
)

hsgp_opt = HybridZuptInsJl.HsgpParameters(
    HybridZuptInsJl.SeHyperparams(hyps), d, m, Lvec_norm;
    input_stats=input_stats, output_stats=output_stats, mid_norm=mid_norm
)

hsgp_base = HybridZuptInsJl.HsgpParameters(
    hsgp_base.hp, hsgp_base.d, m, hsgp_base.LL;
    input_stats=hsgp_base.input_stats,
    output_stats=hsgp_base.output_stats
)

fig_regr = HybridZuptInsJl.plot_regression_results(pred, target)

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

slamHsgp_corr_opt = HybridZuptInsJl.SlamCorrector(round(Int, N / 60), hsgp_opt)
_, _, slamHsgpOpt_corr_traj, io_data["SlamHsgp Opt"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, slamHsgp_corr_opt;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

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

splitHsgp_corr = HybridZuptInsJl.SplitHybridCorrector(round(Int, N / 60), hsgp_base)
_, _, hsgp1_corr_traj, io_data["SplitHsgp Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, splitHsgp_corr;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

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
    "Static" => stat_corr_traj2,
    "SLAM Base" => slamHsgpBase_corr_traj,
    "SLAM Opt" => slamHsgpOpt_corr_traj,
    "SPLIT Base" => hsgp1_corr_traj
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
