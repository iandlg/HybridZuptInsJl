include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
using GLMakie, OrderedCollections, Statistics, Random

# --- Choose Parameters file (see HSGP_PARAM_PATHS in scripts/5Results/_common.jl)
hsgp_p_key = 42
m = 200

# --- Load parameters with corresponding metatdata.
# `load_hsgp_params` already rebuilds the parameter set with this script's `m`,
# so there is no second HsgpParameters(...) reconstruction further down.
hsgp_base, base_FRAME, base_FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

data_key = "ANG2" # meta["data_key"]
data_dir_path = data_dir(data_key)

## --- Load Data ---
# Train/test split lives in _common.jl (TRAIN_IDS/TEST_IDS) so it stays in step
# with the other scripts. Both are keyed by `data_key`: the test list used to be
# hardcoded to [14] with the dataset-keyed dict below it discarded, so a DCSC run
# silently tested on ANG2's trial 14.
train_trial_ids = train_ids(data_key)
test_trial_ids = test_ids(data_key)
all_trial_ids = vcat(train_trial_ids, test_trial_ids)
FRAME = HybridZuptInsJl.HEADING
FEATURE_TYPE = HybridZuptInsJl.TWOD_STEP_YAW

dataset = HybridZuptInsJl.collect_dataset(
    data_dir_path, all_trial_ids;
    frame=FRAME,
    feature_type=FEATURE_TYPE,
)

# Guard against trials that failed to load
missing_train = filter(id -> !haskey(dataset, id), train_trial_ids)
missing_test = filter(id -> !haskey(dataset, id), test_trial_ids)
isempty(missing_train) || @warn "Missing train trials: $missing_train"
isempty(missing_test) || @warn "Missing test trials: $missing_test"
train_trial_ids = filter(id -> haskey(dataset, id), train_trial_ids)
test_trial_ids = filter(id -> haskey(dataset, id), test_trial_ids)

train_results = [dataset[id] for id in train_trial_ids]
test_results = [dataset[id] for id in test_trial_ids]

# Each result tuple: (target::CorrectionIO, input::CorrectionIO, corr_traj, gt_traj, step_seg)
train_out = HybridZuptInsJl.concatenate_io([res[1] for res in train_results])
train_in = HybridZuptInsJl.concatenate_io([res[2] for res in train_results])
test_out = HybridZuptInsJl.concatenate_io([res[1] for res in test_results])
test_in = HybridZuptInsJl.concatenate_io([res[2] for res in test_results])

# `keep_fraction` replaces the old chi-squared `alpha`: the trim is stated directly
# rather than inferred from a Gaussian assumption these residuals do not satisfy.
# 0.85 reproduces what alpha=0.497 was actually doing here (it kept 84.7%, 642/758)
# while saying so on the tin.
outlier_removal_method = "mahalanobis"
outlier_removal_params = OrderedDict(
    "dims" => :output, # nothing or :input or :both
    "method" => "mahalanobis",
    "threshold" => 3.0,
    "keep_fraction" => 0.847,
)

# Quick look at the D² distribution the cut acts on, before it is applied.
# Red line = the threshold; everything right of it gets dropped.
if !isnothing(outlier_removal_params["dims"])
    keep_frac = outlier_removal_params["keep_fraction"]
    mahal_spaces = OrderedDict{String,Matrix{Float64}}()
    outlier_removal_params["dims"] in (:input, :both) && (mahal_spaces["input"] = train_in.data)
    outlier_removal_params["dims"] in (:output, :both) && (mahal_spaces["output"] = train_out.data)

    fig_mahal = Figure()
    for (i, (label, M)) in enumerate(mahal_spaces)
        sqd, d_mahal = HybridZuptInsJl.mahal_sqdistances(M)
        thr = quantile(sqd, keep_frac)
        ax = Axis(fig_mahal[i, 1];
            xlabel="Mahalanobis D²", ylabel="count",
            title="$label (d=$d_mahal, n=$(length(sqd))): keep $keep_frac → D² < $(round(thr, digits=2)), " *
                  "mean $(round(mean(sqd), digits=2)), median $(round(median(sqd), digits=2))")
        hist!(ax, sqd; bins=400)
        vlines!(ax, thr; color=:red)
    end
    display(GLMakie.Screen(), fig_mahal)
end

if !isnothing(outlier_removal_params["dims"])
    train_in, train_out = HybridZuptInsJl.remove_outliers(train_in, train_out;
        method=outlier_removal_params["method"],
        threshold=outlier_removal_params["threshold"],
        keep_fraction=outlier_removal_params["keep_fraction"],
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
    "default_lower" => 1e-3,
    "default_noise_lower" => 1e-1,
    "yaw" => OrderedDict(
        "default_var_lower" => 1e-3,
        "default_ls_lower" => 1e-3
    )
)
for (idx, symb) in enumerate(output_symbols)
    # Define bounds on σ_n, ℓ, σ_f, σ_lin
    lower = fill(optim_params["default_lower"], 4)
    lower[1] = optim_params["default_noise_lower"]

    if symb == output_symbols[4]
        lower[3] = optim_params["yaw"]["default_var_lower"]
        lower[2] = optim_params["yaw"]["default_ls_lower"]
    end

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

fig_regr = HybridZuptInsJl.plot_regression_results(pred, test_out)
## --- Run Correction using both Hyper Parameter Sets ---
trial_id = 15
train_ratio = 0.5
output_channels = [:pos_1, :pos_2, :yaw] # [:pos_1, :pos_2, :pos_3, :yaw]

ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir_path, trial_id
)
# Add noise to training data
noise_spec = HybridZuptInsJl.NoiseSpec(; pos_std=0.05, att_std=5*pi/180, tag="Position & Heading Noise (0.05m, ±5°)")
noise_spec = HybridZuptInsJl.NoiseSpec()
noisy_gt_traj = HybridZuptInsJl.add_gaussian_noise(gt_traj_aligned; pos_std=noise_spec.pos_std, att_std=noise_spec.att_std)

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
    inertial_updated, sim_config_updated, noisy_gt_traj, defaultEstimator;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)

decoupledStatic = HybridZuptInsJl.DecoupledStaticEstimator(round(Int, N / 60); corrected_channels=output_channels)
_, _, stat_corr_traj, io_data["Decoupled Static"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, decoupledStatic;
    x_init=x_init, gt_available=gt_available, ref_frame=FRAME, feature_type=FEATURE_TYPE)


Hsgp_base = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60); params=hsgp_base, corrected_channels=output_channels)
_, _, jointHsgp_base_traj, io_data["Decoupled HSGP Base"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, Hsgp_base;
    x_init=x_init, gt_available=gt_available, ref_frame=base_FRAME, feature_type=base_FEATURE_TYPE)

Hsgp_opt = HybridZuptInsJl.DecoupledHsgpEstimator(round(Int, N / 60); params=hsgp_opt, corrected_channels=output_channels)
_, _, jointHsgp_opt_traj, io_data["Decoupled HSGP Opt"], _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
    inertial_updated, sim_config_updated, noisy_gt_traj, Hsgp_opt;
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
GLMakie.activate!()

fig_ori = HybridZuptInsJl.plot_groundtruth_vs_inertial_orientations(trajs, gt_traj_aligned[step_seg])
fig_xyz = HybridZuptInsJl.plot_groundtruth_vs_inertial_xyz(trajs, gt_traj_aligned[step_seg])
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj_aligned[step_seg]; start=1, stop=10, show_heading=true, heading_stride=1)

fig_rmse_hybrid = with_theme(theme_ggplot2()) do
    HybridZuptInsJl.plot_position_rmse(trajs, gt_traj_aligned[step_seg]; show_index_ticks=true)
end
fig_dist = with_theme(theme_ggplot2()) do
    HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj_aligned[step_seg], gt_available[step_seg])
end

fig_out = HybridZuptInsJl.plot_regression_results(output_data, io_data["Default"]["target"])
fig_out_norm = HybridZuptInsJl.plot_regression_results(output_data_norm, io_data["Decoupled HSGP Opt"]["target_norm"])

fig_in = HybridZuptInsJl.plot_regression_results(input_data, io_data["Default"]["input"])
# display(GLMakie.Screen(), fig_in)
fig_in_norm = HybridZuptInsJl.plot_regression_results(input_data_norm, io_data["Default"]["input_norm"])
# display(GLMakie.Screen(), fig_in_norm)
fig_res = HybridZuptInsJl.plot_regression_results(residual_data)
display(hsgp_opt.input_stats)
display(hsgp_opt.output_stats)
## -- Save hyperparameters ---
# NOT results_path(): this writes a trained artifact, not a figure. The directory
# convention below is what HSGP_PARAM_PATHS in _common.jl indexes, so a new file
# saved here can be added there by key and picked up by every other script.
# `Dates` comes from _common.jl.
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
        "train_ids" => train_trial_ids,
        "test_ids" => test_trial_ids,
        "hsgp_p_key" => hsgp_p_key,
        "outlier_removal" => outlier_removal_params,
        "normalization" => normalization_params,
        "optimization_parameters" => optim_params
    )
)