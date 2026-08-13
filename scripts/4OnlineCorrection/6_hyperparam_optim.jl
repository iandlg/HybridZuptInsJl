include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie, OrderedCollections, Statistics, Random

# --- Choose Parameters file
hsgp_p_key = 43

hsgp_p_path = Dict{Int,String}(
    11 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_THREED_STEP_2026-05-15T16:25:17.521.json",
    20 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T13:07:52.881.json",
    21 => "out/3OfflineCorrection/3_HsgpResults/ANG15_BODY_TWOD_STEP_DT_2026-05-15T14:02:45.772.json",
    22 => "out/3OfflineCorrection/3_HsgpResults/ANG215_BODY_THREED_STEP_2026-06-06T17:33:50.999.json",
    30 => "out/3OfflineCorrection/3_HsgpResults/ANG15_HEADING_TWOD_STEP_DT_2026-05-15T14:50:57.036.json",
    40 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_DT/ANG15_HEADING_TWOD_STEP_DT_2026-07-10T15:06:17.927.json",
    41 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG23_HEADING_TWOD_STEP_YAW_2026-07-13T12:28:30.952.json",
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json", # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json", # no ouptut norm; DCSC
)[hsgp_p_key]

# --- Load parameters with corresponding metatdata
hsgp_base, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, hsgp_p_path)
base_FRAME = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
base_FEATURE_TYPE = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])  # # 

data_key = "DCSC" # meta["data_key"]
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack"
)[data_key]

m = 200


## --- Load Data ---
train_ids = Dict{String,Vector{Int}}(
    "ANG2" => [1, 2, 3, 4, 8, 9, 13, 15, 16],
    "DCSC" => [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
)[data_key]
test_ids = [14] # [3, 13, 15] 
Dict{String,Vector{Int}}(
    "ANG2" => [14],
    "DCSC" => [10]
)
trial_ids = vcat(train_ids, test_ids)
FRAME = HybridZuptInsJl.HEADING
FEATURE_TYPE = HybridZuptInsJl.TWOD_STEP_YAW

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

outlier_removal_params = OrderedDict(
    "dims" => :output, # nothing or :input or :both
    "method" => "mahalanobis",
    "threshold" => 3.0,
    "alpha" => 0.975,
)
if !isnothing(outlier_removal_params["dims"])
    train_in, train_out = HybridZuptInsJl.remove_outliers(train_in, train_out;
        method=outlier_removal_params["method"],
        threshold=outlier_removal_params["threshold"],
        alpha=outlier_removal_params["alpha"],
        dims=outlier_removal_params["dims"]
    )
end
# test_in, test_out = HybridZuptInsJl.remove_outliers(test_in, test_out;
#     method="zscore", threshold=3.0, dims=:both)
## --- Fit stats on TRAIN ONLY, apply to both train and test ---

d = size(train_in.data, 1)

normalization_params = OrderedDict(
    "margin" => 0.5,
    "normalize_x" => true,
    "normalize_y" => false
)

# Input preprocessing
inp = HybridZuptInsJl.compute_input_preprocessing(train_in.data;
    normalize_x=normalization_params["normalize_x"], margin=normalization_params["margin"])
train_in_norm = (train_in.data .- inp.μ) ./ inp.σ
test_in_norm = (test_in.data .- inp.μ) ./ inp.σ

# Output normalisation
outp = HybridZuptInsJl.compute_output_normalisation(train_out.data; normalize_y=normalization_params["normalize_y"])
train_out_norm = (train_out.data .- outp.μ) ./ outp.σ

## --- Optimize hyperparameters per output, predict on TEST ---
output_symbols = ["pos_1", "pos_2", "pos_3", "yaw"]

pred_data = similar(test_out.data)
pred_var = similar(test_out.data)
hyps = Dict{String,Any}()
rng = Random.Xoshiro(123)

optim_params = OrderedDict(
    "use_output_std_as_sigma_n_bound" => true,
    "use_input_std_as_sigma_n_bound" => false,
    "default_upper" => 1e3,
    "default_lower" => 1e-3
)
for (idx, symb) in enumerate(output_symbols)
    # Define bounds on σ_n, ℓ, σ_f, σ_lin
    lower = fill(optim_params["default_lower"], 4)
    if optim_params["use_output_std_as_sigma_n_bound"]
        noise_lower_orig = maximum(train_out.data_std[idx, :])
        max_output_std = noise_lower_orig / outp.σ[idx]
        lower[1] = max_output_std
    end
    upper = fill(optim_params["default_upper"], 4)

    # First pass: fit on train, predict on test to get β for uncertainty propagation
    pred_norm, pred_var_norm, theta, lik, _, _, per_dim_eigvals, β =
        HybridZuptInsJl.hsgp_regression(
            train_in_norm', train_out_norm[idx, :],
            test_in_norm', m;
            use_linear=false, LL=inp.LL_norm, lower=lower, rng=rng, upper=upper, #theta=[0.01, 0.01, 0.01, 1.0]
        )

    if optim_params["use_input_std_as_sigma_n_bound"]
        # Propagate train input uncertainty through the fitted GP derivative
        x_scaled_norm = train_in_norm' .- inp.mid_norm'
        dfdx = zeros(size(x_scaled_norm, 1), d)
        for di in 1:d
            Phi_dx = HybridZuptInsJl.calc_eigenvectors_dx(x_scaled_norm, inp.Lvec_norm, per_dim_eigvals, di)
            dfdx[:, di:di] = Phi_dx * β
        end
        train_in_std_norm = train_in.data_std[:, :] ./ inp.σ
        var_x_norm = sum(dfdx' .^ 2 .* train_in_std_norm .^ 2; dims=1)

        # Refine noise lower bound with propagated input uncertainty, refit on train, predict on TEST
        lower[1] += sqrt(maximum(var_x_norm))

        pred_norm, pred_var_norm, theta, lik, _, _, _, _ =
            HybridZuptInsJl.hsgp_regression(
                train_in_norm', train_out_norm[idx, :],
                test_in_norm', m;
                use_linear=false, LL=inp.LL_norm, lower=lower, rng=rng, upper=upper,
                # theta=[0.01, 0.01, 0.01, 1.0]
            )
    end


    pred_data[idx, :] = pred_norm .* outp.σ[idx] .+ outp.μ[idx]
    pred_var[idx, :] = pred_var_norm .* outp.σ[idx] .^ 2
    hyps[symb] = theta[1:3]
end

@info "Optimized yaw hyperparameters: " hyps["yaw"]

pred = HybridZuptInsJl.CorrectionIO(test_out.t, pred_data, sqrt.(pred_var))

# hyps["pos_1"] = hsgp_base.hp.pos_1
# hyps["pos_2"] = hsgp_base.hp.pos_2
# hyps["pos_3"] = hsgp_base.hp.pos_3

hsgp_opt = HybridZuptInsJl.HsgpParameters(
    HybridZuptInsJl.SeHyperparams(hyps), d, m, inp.Lvec_norm;
    input_stats=[inp.μ, inp.σ],
    output_stats=[outp.μ, outp.σ],
    mid_norm=inp.mid_norm
)

hsgp_base = HybridZuptInsJl.HsgpParameters(
    hsgp_base.hp, hsgp_base.d, m, hsgp_base.LL;
    input_stats=hsgp_base.input_stats,
    output_stats=hsgp_base.output_stats
)

fig_regr = HybridZuptInsJl.plot_regression_results(pred, test_out)
## --- Run Correction using both Hyper Parameter Sets ---
trial_id = 5
train_ratio = 0.45
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

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

defaultEstimator = HybridZuptInsJl.BaseEstimator(round(Int, N / 60))
zupt, step_seg, def_corr_traj, io_data["Default"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, defaultEstimator;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

decoupledStatic = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60); corrected_channels=output_channels)
_, _, stat_corr_traj, io_data["Decoupled Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, decoupledStatic;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


jointHsgp_base = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60), hsgp_base; corrected_channels=output_channels)
_, _, jointHsgp_base_traj, io_data["Joint HSGP Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, jointHsgp_base;
    x_init=x_init, gt_available=gt_available, ref_frame=base_FRAME, feature_type=base_FEATURE_TYPE)

jointHsgp_opt = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60), hsgp_opt; corrected_channels=output_channels)
_, _, jointHsgp_opt_traj, io_data["Joint HSGP Opt"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, gt_traj_aligned, jointHsgp_opt;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


# splitHsgp_corr = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60), hsgp_base)
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
    "Default" => def_corr_traj,
    "Decoupled Static" => stat_corr_traj,
    "Joint HSGP Base" => jointHsgp_base_traj,
    "Joint HSGP Opt" => jointHsgp_opt_traj,
    # "SPLIT Base" => hsgp1_corr_traj
)

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=1, stop=10, show_heading=true, heading_stride=1)
with_theme(theme_ggplot2()) do
    fig_rmse_hybrid = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
end
with_theme(theme_ggplot2()) do
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end

fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Default"]["target"])
fig_out_norm = HybridZuptInsJl.plot_regression_results(output_data_norm, io_data["Joint HSGP Opt"]["target_norm"])

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
filename = "$(data_key)_$(FRAME)_$(FEATURE_TYPE)_$(time).json"
HybridZuptInsJl.to_json(joinpath(combo_dir, filename), hsgp_opt;
    metadata=Dict(
        "data_key" => data_key,
        "trial_id" => trial_id,
        "ref_frame" => FRAME,
        "feature_type" => FEATURE_TYPE,
        "normalize_input" => normalization_params["normalize_x"],
        "normalize_output" => normalization_params["normalize_y"],
        "train_ids" => train_ids,
        "outlier_removal" => outlier_removal_params,
        "normalization" => normalization_params,
        "optimization_parameters" => optim_params
    )
)