include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using OrderedCollections, DataFrames, Statistics, Random
import Optim, GaussianProcesses, GLMakie

# ── Config ────────────────────────────────────────────────────────────────────
data_key = "ANG2"
FRAME = HybridZuptInsJl.BODY
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

base_dir = "out/3OfflineCorrection/6_HypOpt"
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

train_io = first.(dataset[train_id], 2)
train_out = HybridZuptInsJl.concatenate_io(first.(train_io))
train_in = hcat(last.(train_io)...)

test_data = dataset[test_id]

# ── 10-fold shuffle CV hyperparameter search ──────────────────────────────────
rng = MersenneTwister(rng_seed)
n_train_total = length(train_out)
n_select = round(Int, train_frac * n_train_total)

all_hyps = Vector{HybridZuptInsJl.SeHyperparams}(undef, n_folds)
io_stats = Vector{Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}}}}(undef, n_folds)
input_domains = Vector{Vector{Float64}}(undef, n_folds)

for fold in 1:n_folds
    @info "===== Fold $fold / $n_folds ====="
    idx = sort(randperm(rng, n_train_total)[1:n_select])

    input_stats = normalize_input ? [
        mean(train_in[:, idx], dims=2)[:],
        std(train_in[:, idx], dims=2)[:]
    ] : [fill(0.0, d), fill(1.0, d)]

    output_stats = normalize_output ? [
        mean(train_out[idx].data, dims=2)[:],
        std(train_out[idx].data, dims=2)[:]
    ] : [fill(0.0, 4), fill(1.0, 4)]

    # Compute input domain
    feature_scaled = (train_in[:, idx] .- input_stats[1]) ./ input_stats[2]
    mask = normalize_input ? (all(feature_scaled .<= z_thresh, dims=1)[:]) : (1:size(feature_scaled, 2))
    LL = margin * maximum(abs, feature_scaled[:, mask], dims=2)[:] # (d,)

    _, hyps = HybridZuptInsJl.compute_corrections(
        train_in[:, idx], train_out[idx],
        HybridZuptInsJl.compute_gp_corrections;
        n_restarts_optimizer=n_restarts_optimizer,
        log_kern_bounds=log_kern_bounds,
        log_noise_bounds=log_noise_bounds,
        method=method,
        normalize_x=normalize_input,
        normalize_y=normalize_output,
        ard=ard,
        n_fold=1
    )

    # Save parameters
    io_stats[fold] = (input_stats, output_stats)
    input_domains[fold] = LL
    all_hyps[fold] = hyps[1]
end

## ── Evaluate each fold's hyps on every test trial ────────────────────────────
# rmse_table[fold, trial] 
rmse_rate_gp = zeros(n_folds, length(test_id))
rmse_rate_ins = zeros(n_folds, length(test_id))   # same for all folds, filled once
rmse_rate_hsgp = zeros(n_folds, length(test_id))

for (fold, (hp, io_stat, LL)) in enumerate(zip(all_hyps, io_stats, input_domains))
    for (j, (test_out_j, test_in_j, ins_traj_j, gt_traj_j, segs_j)) in enumerate(test_data)
        # Compute GP trajectory
        gp_corr, _ = HybridZuptInsJl.compute_corrections(
            test_in_j, test_out_j,
            HybridZuptInsJl.compute_gp_corrections, hp;
            n_restarts_optimizer=0,
            normalize_x=normalize_input,
            normalize_y=normalize_output,
            ard=ard
        )
        gp_traj = HybridZuptInsJl.apply_corrections(ins_traj_j, gp_corr, segs_j; ref_frame=FRAME)

        # Compute HSGP trajectory
        @show io_stat[2]
        hsgp_corr, _ = HybridZuptInsJl.compute_corrections(
            test_in_j, test_out_j, HybridZuptInsJl.compute_hsgp_corrections, hp;
            m=m, margin=margin, normalize_x=normalize_input, z_thresh=z_thresh,
            normalize_y=normalize_output, input_stats=io_stat[1], output_stats=io_stat[2], LL=LL
        )
        hsgp_traj = HybridZuptInsJl.apply_corrections(ins_traj_j, hsgp_corr, segs_j; ref_frame=FRAME)

        tt_dist = HybridZuptInsJl.total_distance(gt_traj_j[segs_j])
        rmse_rate_gp[fold, j] = HybridZuptInsJl.rmse(gp_traj, gt_traj_j[segs_j])[end] / tt_dist
        rmse_rate_hsgp[fold, j] = HybridZuptInsJl.rmse(hsgp_traj, gt_traj_j[segs_j])[end] / tt_dist
        rmse_rate_ins[fold, j] = HybridZuptInsJl.rmse(ins_traj_j[segs_j], gt_traj_j[segs_j])[end] / tt_dist
    end
end

mean_rmse_rate_per_fold = vec(mean(rmse_rate_gp, dims=2))   # (n_folds,)
best_fold = argmin(mean_rmse_rate_per_fold)
best_hyp = all_hyps[best_fold]
@info "Best fold: $best_fold  HSGP mean-RMSE rate=$(round(mean_rmse_rate_per_fold[best_fold], digits=4))"

# Create DataFrame: rows = folds, columns = trial IDs
trial_labels = [string("Trial_", id) for id in test_id]
df_gp = DataFrame(rmse_rate_gp, :auto)
rename!(df_gp, [Symbol(t) for t in trial_labels])
# Display
println("\nRMSE rate (RMSE / total distance) for GP corrections per fold and trial:")
show(df_gp, allrows=true)

## Plots per test trial using best fold hyps

# Test hyperparameters on test trials
for (id, (test_out, test_in, ins_traj, gt_traj, segs)) in enumerate(test_data)
    id = test_id[id]
    @info "---- Trial $id -----"

    gp_corrections, _ = HybridZuptInsJl.compute_corrections(
        test_in, test_out, HybridZuptInsJl.compute_gp_corrections, best_hyp;
        n_restarts_optimizer=0, normalize_x=normalize_input, normalize_y=normalize_output, ard=ard)
    gp_traj = HybridZuptInsJl.apply_corrections(ins_traj, gp_corrections, segs; ref_frame=FRAME)
    gp_traj = HybridZuptInsJl.Trajectory(gt_traj.t[segs], gp_traj.pos, gp_traj.R_nb, gp_traj.vel)

    static_corrections, _ = HybridZuptInsJl.compute_corrections(
        test_in, test_out, HybridZuptInsJl.compute_static_corrections)
    static_traj = HybridZuptInsJl.apply_corrections(ins_traj, static_corrections, segs; ref_frame=FRAME)
    static_traj = HybridZuptInsJl.Trajectory(gt_traj.t[segs], static_traj.pos, static_traj.R_nb, static_traj.vel)

    hsgp_corr, _ = HybridZuptInsJl.compute_corrections(
        test_in, test_out, HybridZuptInsJl.compute_hsgp_corrections, best_hyp;
        m=m, margin=margin, normalize_x=normalize_input, z_thresh=z_thresh,
        normalize_y=normalize_output, input_stats=io_stats[best_fold][1], output_stats=io_stats[best_fold][2], LL=input_domains[best_fold]
    )
    hsgp_traj = HybridZuptInsJl.apply_corrections(ins_traj, hsgp_corr, segs; ref_frame=FRAME)
    hsgp_traj = HybridZuptInsJl.Trajectory(gt_traj.t[segs], hsgp_traj.pos, hsgp_traj.R_nb, hsgp_traj.vel)

    ins_rmse = HybridZuptInsJl.rmse(ins_traj[segs], gt_traj[segs])[end]
    static_rmse = HybridZuptInsJl.rmse(static_traj, gt_traj[segs])[end]
    gp_rmse = HybridZuptInsJl.rmse(gp_traj, gt_traj[segs])[end]
    hsgp_rmse = HybridZuptInsJl.rmse(hsgp_traj, gt_traj[segs])[end]

    trajs = OrderedDict{String,HybridZuptInsJl.Trajectory}(
        "model" => ins_traj[segs],
        "model + static" => static_traj,
        "model + GP" => gp_traj,
        "model + HSGP" => hsgp_traj
    )

    outputs = OrderedDict{String,HybridZuptInsJl.CorrectionIO}(
        "static" => static_corrections,
        "GP" => gp_corrections,
        "HSGP" => hsgp_corr
    )

    # Create combined folder name: e.g., "BODY-TWOD_STEP_DT"
    trial_dir = joinpath(combo_dir, string(id), "Plot")
    mkpath(trial_dir)

    # Save plots
    fig_dist = HybridZuptInsJl.plot_position_distance_error(trajs, gt_traj[segs])
    GLMakie.save(joinpath(trial_dir, "distance_error.png"), fig_dist)

    fig_rmse = HybridZuptInsJl.plot_position_rmse(trajs, gt_traj[segs])
    GLMakie.save(joinpath(trial_dir, "rmse.png"), fig_rmse)

    fig_2d = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(trajs, gt_traj[segs]; samples=20)
    GLMakie.save(joinpath(trial_dir, "trajectory_2d.png"), fig_2d)

    fig_out = HybridZuptInsJl.plot_regression_results(outputs, test_out)
    GLMakie.save(joinpath(trial_dir, "regression.png"), fig_out)
end



## ── Hyperparameter variability plot ──────────────────────────────────────────
let
    output_keys = ["pos_1", "pos_2", "pos_3", "yaw"]
    param_names = ["σ_n", "ℓ", "σ_f"]

    fig = GLMakie.Figure(size=(1200, 800))
    colors = GLMakie.Makie.wong_colors()

    for (oi, ok) in enumerate(output_keys)
        for (pi, pn) in enumerate(param_names)
            ax = GLMakie.Axis(fig[oi, pi];
                title="$ok — $pn",
                xlabel="Fold", ylabel="Value")

            vals = [getfield(all_hyps[f], Symbol(ok))[pi] for f in 1:n_folds]
            GLMakie.barplot!(ax, 1:n_folds, vals; color=colors[oi])
            GLMakie.scatter!(ax, [best_fold], [vals[best_fold]];
                color=:red, markersize=14, marker=:star5, label="best")
        end
    end
    GLMakie.save(joinpath(combo_dir, "hyp_variability.png"), fig)
    display(fig)
end

# ── Per-fold RMSE across test trials ─────────────────────────────────────────
let
    fig = GLMakie.Figure(size=(900, 500))
    ax = GLMakie.Axis(fig[1, 1];
        title="Mean RMSE rate per fold",
        xlabel="Fold", ylabel="RMSE (m)")

    GLMakie.barplot!(ax, 1:n_folds, mean_rmse_rate_per_fold; color=:steelblue, alpha=0.8)
    GLMakie.scatter!(ax, [best_fold], [mean_rmse_rate_per_fold[best_fold]];
        color=:red, markersize=16, marker=:star5, label="best")
    GLMakie.hlines!(ax, [mean(vec(rmse_rate_ins))]; color=:black, linestyle=:dash,
        linewidth=1.5, label="INS baseline")
    GLMakie.axislegend(ax)
    GLMakie.save(joinpath(combo_dir, "fold_rmse.png"), fig)
    display(fig)
end

# ── Summary DataFrame ─────────────────────────────────────────────────────────
df = DataFrame(
    fold=1:n_folds,
    mean_gp_rmse=mean_rmse_rate_per_fold,
    min_gp_rmse=vec(minimum(rmse_rate_gp, dims=2)),
    max_gp_rmse=vec(maximum(rmse_rate_gp, dims=2)),
)
println("\n=== Fold RMSE Summary ===")
show(df, allrows=true)

## ── Save best hyperparameters ─────────────────────────────────────────────────
import Dates
outdir = combo_dir
mkpath(outdir)
filename = joinpath(outdir,
    "$(data_key)_$(FRAME)_$(FEATURE_TYPE)_best_fold$(best_fold)_$(Dates.now()).json")

hsgp_hyps = HybridZuptInsJl.HsgpParameters(
    best_hyp, size(train_in, 1), m, input_domains[best_fold];
    input_stats=io_stats[best_fold][1], output_stats=io_stats[best_fold][2]
)

HybridZuptInsJl.to_json(filename, hsgp_hyps;
    metadata=Dict{String,Any}(
        "data_key" => data_key,
        "ref_frame" => string(FRAME),
        "feature_type" => string(FEATURE_TYPE),
        "method_key" => method_key,
        "train_trial_ids" => train_id,
        "best_fold" => best_fold,
        "mean_rmse_rate_per_fold" => mean_rmse_rate_per_fold,
        "n_folds" => n_folds,
        "train_frac" => train_frac,
        "rng_seed" => rng_seed,
        "normalize_input" => normalize_input,
        "normalize_output" => normalize_output,
        "margin" => margin,
        "z_thresh" => z_thresh
    )
)
@info "Saved best hyperparameters → $filename"