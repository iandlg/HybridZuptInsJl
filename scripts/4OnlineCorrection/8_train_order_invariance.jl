# Is the trained model independent of the order the training tracks are fed in?
# Train the same estimator on trial A then B, and on B then A, and plot the two
# resulting β vectors from `get_model` against each other.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
using GLMakie, LinearAlgebra

hsgp_p_key = 42
m = 200
hsgp_p, FRAME, FEATURE_TYPE, _ = load_hsgp_params(hsgp_p_key; m=m)

data_key = "DCSC"
trial_a, trial_b = 5, 14
output_channels = [:pos_1, :pos_2, :yaw]
estimator_type = HybridZuptInsJl.DecoupledHsgpEstimator

sim_config = HybridZuptInsJl.InsConfig()
data_dir_path = data_dir(data_key)

function load_trial(trial_id::Int)
    ins_traj, gt_traj, _, _, inertial, cfg = HybridZuptInsJl.compute_aligned_ins_trajectory(
        data_dir_path, trial_id; sim_config=sim_config
    )
    x_init = vcat(
        ins_traj.pos[:, 1],
        ins_traj.vel[:, 1],
        HybridZuptInsJl.matrix_to_euler(ins_traj.R_nb[:, :, 1])
    )
    return (inertial=inertial, cfg=cfg, gt=gt_traj, x_init=x_init, N=length(inertial))
end

trials = Dict(id => load_trial(id) for id in (trial_a, trial_b))

"""Train sequentially over `order`, carrying the model from one track to the next.
Returns the final `(β, Σβ)` and the per-track io data that was used to train it."""
function train_sequence(order)
    model = nothing
    io = Dict{Int,Any}()
    for id in order
        t = trials[id]
        est = estimator_type(round(Int, t.N / 60); params=hsgp_p, corrected_channels=output_channels)
        _, _, _, io[id], model = HybridZuptInsJl.hybrid_zupt_aided_insv2(
            t.inertial, t.cfg, t.gt, est;
            x_init=t.x_init, gt_available=fill(true, t.N),
            ref_frame=FRAME, feature_type=FEATURE_TYPE,
            init_model=model, posyaw_measurement_update=false
        )
    end
    return model, io
end

(β_ab, Σ_ab), train_io_ab = train_sequence((trial_a, trial_b))
(β_ba, Σ_ba), train_io_ba = train_sequence((trial_b, trial_a))

σ_ab, σ_ba = sqrt.(diag(Σ_ab)), sqrt.(diag(Σ_ba))

label_ab = "$(trial_a) → $(trial_b)"
label_ba = "$(trial_b) → $(trial_a)"

fig = Figure(size=(1100, 900))
ax1 = Axis(fig[1, 1], ylabel="β", title="Training order invariance ($(data_key), $(nameof(estimator_type)))")
lines!(ax1, β_ab, label=label_ab)
lines!(ax1, β_ba, label=label_ba, linestyle=:dash)
axislegend(ax1)

ax2 = Axis(fig[2, 1], ylabel="β difference")
lines!(ax2, β_ab .- β_ba, color=:firebrick)

ax3 = Axis(fig[3, 1], ylabel="σ(β)")
lines!(ax3, σ_ab, label=label_ab)
lines!(ax3, σ_ba, label=label_ba, linestyle=:dash)
axislegend(ax3)

ax4 = Axis(fig[4, 1], xlabel="coefficient index", ylabel="σ(β) difference")
lines!(ax4, σ_ab .- σ_ba, color=:firebrick)

# Channel block boundaries (β is 4 blocks of m: pos_1, pos_2, pos_3, yaw)
for ax in (ax1, ax2, ax3, ax4)
    vlines!(ax, [m, 2m, 3m], color=(:gray, 0.5), linestyle=:dot)
end

# Training data both orders are fed (same two tracks, only the order differs)
fig_train_in = [
    HybridZuptInsJl.plot_input_features(
        train_io_ab[id]["input"].data, train_io_ab[id]["input"].t;
        feature_std=train_io_ab[id]["input"].data_std, title="Trial $(id) : training input")
    for id in (trial_a, trial_b)
]
fig_train_out = HybridZuptInsJl.plot_regression_results(Dict{String,HybridZuptInsJl.CorrectionIO}(
    "Trial $(trial_a)" => train_io_ab[trial_a]["target"],
    "Trial $(trial_b)" => train_io_ab[trial_b]["target"],
))

"""Channel-wise difference of one track's training data (`:data` or `:data_std`)
between the two orders."""
function train_data_diff(id, key, field)
    a, b = train_io_ab[id][key], train_io_ba[id][key]
    A, B = getfield(a, field), getfield(b, field)
    n = min(size(A, 2), size(B, 2))
    return A[:, 1:n] .- B[:, 1:n], a.t[1:n]
end

fig_train_diff = Pair{String,Any}[
    "train_$(key)_$(field)_diff_$(id)" => HybridZuptInsJl.plot_input_features(
        train_data_diff(id, key, field)...;
        title="Trial $(id) : training $(key) $(field) difference, ($(label_ab)) - ($(label_ba))")
    for id in (trial_a, trial_b) for key in ("input", "target") for field in (:data, :data_std)
]

@info "max |Δβ| = $(maximum(abs, β_ab .- β_ba)), max |β| = $(maximum(abs, β_ab))"
@info "max |Δσ| = $(maximum(abs, σ_ab .- σ_ba)), max σ = $(maximum(σ_ab))"
for id in (trial_a, trial_b), key in ("input", "target"), field in (:data, :data_std)
    @info "trial $id: max |Δ $key $field| = $(maximum(abs, train_data_diff(id, key, field)[1]))"
end

figures = [
    "beta" => fig,
    "train_input_$(trial_a)" => fig_train_in[1],
    "train_input_$(trial_b)" => fig_train_in[2],
    "train_target" => fig_train_out,
    fig_train_diff...,
]

foreach(display, last.(figures))

# Vector copies, so the coefficient plots can actually be zoomed into
CairoMakie.activate!()
for (name, f) in figures
    save(stamped("8_TrackOrderInvariance", name), f)
end
