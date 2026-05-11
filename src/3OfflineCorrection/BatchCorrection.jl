"""
    compute_training_io(traj::Trajectory, traj_gt::Trajectory, step_seg::Vector{Int};
                        ref_frame::ReferenceFrame = BODY) ->
                        (output_yawdiff::Vector{Float64},
                         output_pos::Matrix{Float64},
                         input_feature::Matrix{Float64})

Compute training inputs and outputs for GP regression from a pair of trajectories.

# Arguments
- `traj`: Inertial trajectory, temporally aligned with `traj_gt`.
- `traj_gt`: Ground truth trajectory, temporally aligned with `traj`.
- `step_seg`: Indices marking step boundaries in the trajectory (1‑based, length `n_steps`).
- `ref_frame`: Reference frame for step vector computation (`BODY` or `HEADING`).

# Returns
- `ouputs`: Dictionnary with `yaw` and `pos` keys .
- `input_feature`: Inertial step vectors in `ref_frame`, size `3 x (n_steps-1)`.

# Throws
- `ArgumentError` if trajectories are not temporally compatible (requires `TimeSeries.is_compatible`).
"""
function compute_training_io(
    traj::Trajectory,
    traj_gt::Trajectory,
    step_seg::Vector{Int};
    ref_frame::ReferenceFrame=BODY
)
    if !is_compatible(traj, traj_gt)
        throw(ArgumentError("TimeSeries must be compatible."))
    end

    # Unwrap yaw angles to avoid discontinuities
    euler_nb = matrix_to_euler(traj.R_nb)
    euler_nb_gt = matrix_to_euler(traj_gt.R_nb)

    inertial_yaw = unwrap(euler_nb[3, :])   # row 3 = yaw (since 1: roll, 2: pitch, 3: yaw)
    gt_yaw = unwrap(euler_nb_gt[3, :])

    # Defin output dictionnary
    outputs = Dict{String,Union{Vector{Float64},Matrix{Float64}}}()

    # Yaw difference over steps: diff(gt) - diff(inertial)
    outputs["yaw"] = diff(gt_yaw[step_seg]) - diff(inertial_yaw[step_seg])   # length n_steps-1

    # Select step‑vector function based on reference frame
    funs = Dict(
        BODY => step_vectors_body,
        HEADING => step_vectors_heading,
    )
    if !haskey(funs, ref_frame)
        throw(ArgumentError("Unsupported reference frame: $ref_frame"))
    end

    input_feature = funs[ref_frame](traj, step_seg)      # 3 x (n_steps-1)
    gt_steps = funs[ref_frame](traj_gt, step_seg)


    outputs["pos"] = gt_steps - input_feature

    Δt = diff(step_seg)
    # input_feature[3, :] = Δt
    # input_feature = vcat(input_feature, Δt')

    return outputs, input_feature
end

"""
    apply_corrections(traj::Trajectory,
                      yawdiff_correction::Vector{Float64},
                      pos_correction::Matrix{Float64},
                      segs::Vector{Int};
                      ref_frame::ReferenceFrame = BODY) -> Trajectory

Apply yaw and position corrections to a full inertial trajectory, producing a corrected
trajectory sampled at the step segment indices.

# Arguments
- `traj`: Complete inertial trajectory to correct.
- `yawdiff_correction`: Per‑step yaw correction (radians), length `n_steps-1`.
- `pos_correction`: Per‑step position correction (metres) in the chosen reference frame,
  size `3 x (n_steps-1)`.
- `segs`: Indices marking step boundaries in `traj` (length `n_steps`).
- `ref_frame`: Reference frame for step vector computation (`BODY` or `HEADING`).

# Returns
- Corrected `Trajectory` sampled at the step indices, with updated positions and orientations.
"""
function apply_corrections(
    traj::Trajectory,
    yawdiff_correction::Vector{Float64},
    pos_correction::Matrix{Float64},
    segs::Vector{Int};
    ref_frame::ReferenceFrame=BODY
)
    n_steps = length(segs)
    # Extract Euler angles at step boundaries (3 x n_samples)
    euler = matrix_to_euler(traj.R_nb)
    yaw = euler[3, :]
    unwrap!(yaw)

    diff_yaw = diff(yaw[segs]) + yawdiff_correction
    # Reconstruct cumulative yaw: start with first yaw, then cumsum of diffs
    new_yaws = cumsum(vcat([yaw[segs[1]]], diff_yaw))
    new_yaws = mod.(new_yaws .+ π, 2π) .- π

    # Prepare new Euler angle array (only yaw changes, roll/pitch keep original)
    if ref_frame == BODY
        new_euler = copy(euler[:, segs])          # keep original roll/pitch
        new_euler[3, :] = new_yaws
    elseif ref_frame == HEADING
        # In heading frame, we set roll/pitch to zero (as in original Python)
        new_euler = zeros(size(euler[:, segs]))
        new_euler[3, :] = new_yaws
    else
        throw(ArgumentError("Unsupported Reference Frame: $ref_frame"))
    end

    # Convert corrected Euler angles to rotation matrices (3x3xn_steps)
    new_R = euler_to_matrix(new_euler)   # shape (3,3,n_steps)

    # --- Integrate corrected positions -----------------------------------------
    # Select step‑vector function
    funs = Dict(
        BODY => step_vectors_body,
        HEADING => step_vectors_heading,
    )

    # Inertial step vectors in the chosen reference frame (3 x n_steps-1)
    steps = funs[ref_frame](traj, segs)

    # Position integration: pos_out[:, k] = new_R[:,:,k-1] * (steps[:,k-1] + pos_correction[:,k-1]) + pos_out[:,k-1]
    pos_out = Matrix{Float64}(undef, 3, n_steps)
    pos_out[:, 1] = traj.pos[:, segs[1]]

    for k in 2:n_steps
        pos_out[:, k] = new_R[:, :, k-1] * (steps[:, k-1] + pos_correction[:, k-1]) + pos_out[:, k-1]
    end

    # --- Build output trajectory ------------------------------------------------
    # Final orientation matrices (converted from the corrected Euler angles)
    new_euler = copy(euler[:, segs])
    new_euler[3, :] = new_yaws
    R_nb_new = euler_to_matrix(new_euler)
    @show size(traj.t[segs])
    return Trajectory(
        traj.t[segs],      # time at step boundaries
        pos_out,           # 3 x n_steps
        R_nb_new,          # 3 x 3 x n_steps
        traj.vel === nothing ? nothing : traj.vel[:, segs]
    )
end

function mcmc_with_priors!(gp::GaussianProcesses.GPE;
    n_samples::Int=200,
    burn_in::Int=100,
    kern_prior=Normal(0.0, 1.0),   # applied to all kernel params
    noise_prior=Normal(-2.0, 1.0))  # applied to logNoise

    # Set priors on kernel params
    n_kern_params = GaussianProcesses.num_params(gp.kernel)
    kern_priors = fill(kern_prior, n_kern_params)
    GaussianProcesses.set_priors!(gp.kernel, kern_priors)

    # Set prior on noise
    GaussianProcesses.set_priors!(gp.logNoise, [noise_prior])

    # Run MCMC
    chain = GaussianProcesses.mcmc(gp; nIter=n_samples + burn_in)
    @show size(chain)

    # chain is (n_params × n_iterations) — drop burn-in
    chain_post = chain[:, burn_in+1:end]

    # average_params = Distributions.StatsBase.mean(chain_post, dims=2)[:]
    best_idx = argmax([
        begin
            GaussianProcesses.set_params!(gp, chain_post[:, i])
            GaussianProcesses.update_target!(gp)
            gp.target
        end for i in axes(chain_post, 2)
    ])

    GaussianProcesses.set_params!(gp, chain_post[:, best_idx])

    GaussianProcesses.update_target!(gp)

    return chain_post
end

function optimize_with_restarts!(gp::GaussianProcesses.GPE, n_restarts::Int;
    kern_lo::Vector{Float64}=fill(-3.0, GaussianProcesses.num_params(gp.kernel)),
    kern_hi::Vector{Float64}=fill(3.0, GaussianProcesses.num_params(gp.kernel)),
    noise_bounds::Vector{Float64}=[-2.0, 0.0]
)

    # kernbounds covers ONLY kernel params, NOT the GP noise (logNoise)
    # num_params(gp.kernel) gives the right count for kernbounds
    @assert length(kern_lo) == length(kern_hi) == GaussianProcesses.num_params(gp.kernel) """
        Bounds length ($(length(kern_lo))) must match kernel param count \
        ($(GaussianProcesses.num_params(gp.kernel))), not total GP params \
        ($(length(GaussianProcesses.get_params(gp)))).
    """

    best_target = -Inf
    best_params = GaussianProcesses.get_params(gp)   # includes logNoise

    for i in 1:n_restarts
        # Sample random kernel params within bounds
        kern_params = kern_lo .+ (kern_hi .- kern_lo) .* rand(length(kern_lo))
        kern_params = clamp.(kern_params, kern_lo, kern_hi)   # guarantee inside bounds

        # logNoise is separate — sample in its own range
        log_noise = noise_bounds[1] + (noise_bounds[2] - noise_bounds[1]) * rand()
        log_noise = clamp(log_noise, noise_bounds[1], noise_bounds[2])


        @info "Restart $i; starting parameters : log_σ_n = $log_noise, log_ℓ = $(kern_params[1]), log_σ_f = $(kern_params[2])"

        # set_params! expects [kernel_params..., logNoise]
        GaussianProcesses.set_params!(gp, vcat(log_noise, kern_params))
        GaussianProcesses.update_target!(gp)
        GaussianProcesses.optimize!(gp;
            kernbounds=[kern_lo, kern_hi], noisebounds=noise_bounds)

        if gp.target > best_target
            best_target = gp.target
            best_params = GaussianProcesses.get_params(gp)
            @debug "Restart $i: log-likelihood = $best_target"
        end
    end

    GaussianProcesses.set_params!(gp, best_params)
    GaussianProcesses.update_target!(gp)
    return best_target
end


function compute_gp_corrections(
    x::Matrix{Float64}, y::Vector{Float64},
    kernel::Union{Nothing,GaussianProcesses.Kernel}=nothing;
    hyperparameter::Union{Nothing,Vector{Float64}}=nothing,
    n_restarts_optimizer::Int=5
)
    n_features, n_samples = size(x)
    @assert length(y) == n_samples "x and y must have same number of samples"

    D = 10
    y_testing_gp = zeros(n_samples)
    hyperparams = zeros(D, 4)

    # --- scale features ---------------------------------------------------
    μ = mean(x, dims=2)[:]
    sigma = std(x, dims=2)[:]
    sigma[sigma.==0.0] .= 1.0
    x_scaled = (x .- μ) ./ sigma

    log_ℓ, log_σ_f, log_σ_n = 0.0, 0.0, -1.0
    if !isnothing(hyperparameter)
        log_σ_f = log(hyperparameter[1])
        log_ℓ = log(hyperparameter[2])
        log_σ_n = log(hyperparameter[3])
    end

    make_kernel() = kernel === nothing ? GaussianProcesses.SE(log_ℓ, log_σ_f) : deepcopy(kernel)

    # logNoise is handled separately inside the function
    kern_lo = [-1.0, -3.0]
    kern_hi = [3.0, 2.0]
    noise_bounds = [-1.71, -0.1]

    for i in 1:D
        @info "------------ Fold $i ------------"
        test_start = floor(Int, (i - 1) * n_samples / D) + 1
        test_end = floor(Int, i * n_samples / D)
        train_ind = vcat(1:test_start-1, test_end+1:n_samples)
        test_ind = test_start:test_end

        isempty(train_ind) && continue

        x_train = x_scaled[:, train_ind]
        y_train = y[train_ind]
        x_test = x_scaled[:, test_ind]

        gp = GaussianProcesses.GP(x_train, y_train,
            GaussianProcesses.MeanZero(), make_kernel(), log_σ_n)
        log_σ_n, log_ℓ, log_σ_f = GaussianProcesses.get_params(gp)
        # @info "hyperparameters before optim : log_σ_n = $log_σ_n, log_ℓ = $log_ℓ, log_σ_f = $log_σ_f"
        if n_restarts_optimizer > 0
            # y_pred, _ = GaussianProcesses.predict_y(gp, x_test)
            # target = sqrt(mean((y_test .- y_pred) .^ 2))
            # best_params = GaussianProcesses.get_params(gp)
            # for i in 1:1
            #     optimize_with_restarts!(gp, 3; lo=lo, hi=hi)
            #     y_pred, _ = GaussianProcesses.predict_y(gp, x_test)

            #     new_target = sqrt(mean((y_test .- y_pred) .^ 2))
            #     if new_target < target
            #         target = new_target
            #         @info "new target $target"
            #         best_params = GaussianProcesses.get_params(gp)
            #     end
            # end
            optimize_with_restarts!(gp, n_restarts_optimizer;
                kern_lo=kern_lo, kern_hi=kern_hi, noise_bounds=noise_bounds)
            # mcmc_with_priors!(gp, n_samples=500, burn_in=100)
        end

        y_pred, _ = GaussianProcesses.predict_y(gp, x_test)
        y_testing_gp[test_ind] = y_pred

        # params = [log(σ_n), log(ℓ), log(σ_f)]  — extract directly
        log_σ_n, log_ℓ, log_σ_f = GaussianProcesses.get_params(gp)

        @info "Fold $i Parameters: lg_σ_n = $(round(log_σ_n, digits=3)), lg_ℓ = $(round(log_ℓ, digits=3)), lg_σ_f = $(round(log_σ_f, digits=3))"
        hyperparams[i, :] = [gp.target, exp(log_σ_f), exp(log_ℓ), exp(log_σ_n)]
        #                     lml        sigma_f        len_scale    sigma_n
    end

    return y_testing_gp, hyperparams
end


function compute_static_corrections(
    x::Matrix{Float64},
    y::Vector{Float64};
    hyperparameter::Union{Nothing,Vector{Float64}}=nothing,
    n_restarts_optimizer::Int=5
)
    D = 10
    _, n_samples = size(x)
    y_pred = fill(0.0, n_samples)
    for i in 1:D
        test_start = floor(Int, (i - 1) * n_samples / D) + 1
        test_end = floor(Int, i * n_samples / D)
        train_ind = vcat(1:test_start-1, test_end+1:n_samples)
        test_ind = test_start:test_end
        y_pred[test_ind] = fill(mean(y[train_ind]), length(test_ind))
    end
    return y_pred, nothing
end
"""
    compute_corrections(
    input_feature::Matrix{Float64},
    output_yawdiff::Vector{Float64},
    output_pos::Matrix{Float64},
    correction_method::Function;
    method_kwargs...
    ) -> Dict{String,Any}

Apply a given correction method (e.g. GP, static mean) to the yaw and each position
component, returning a dictionary of results.

# Arguments
- `input_feature`: input features, size `(n_features, n_samples)`.
- `output_yawdiff`: yaw target, length `n_samples`.
- `output_pos`: position targets, size `(3, n_samples)`.
- `correction_method`: function with signature `(x::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> (predictions::Vector{Float64}, hyperparams::Any)`.
- `method_kwargs`: any extra keyword arguments passed to `correction_method` (e.g. `n_restarts_optimizer`, `kernel`, etc.).

# Returns
- `predictions`: Dictionary with keys `yaw` `pos`
- `hyperparameters` : Dictionary with keys  `yaw` `pos_1` ...
"""
function compute_corrections(
    input_feature::Matrix{Float64},
    outputs::Dict{String,Union{Vector{Float64},Matrix{Float64}}},
    correction_method::Function,
    hyperparameters::Union{Nothing,SeHyperparams}=nothing;
    kwargs...
)

    n_samples = size(input_feature, 2)
    @assert length(outputs["yaw"]) == n_samples
    @assert size(outputs["pos"], 2) == n_samples

    predictions = Dict{String,Union{Vector{Float64},Matrix{Float64}}}()
    hyperparams = Dict{String,Union{Matrix{Float64},Nothing}}()

    # Yaw correction
    @info "Computing Yaw corrections"
    y_pred, hyper = correction_method(input_feature, outputs["yaw"]; hyperparameter=hyperparameters.yaw, kwargs...)
    predictions["yaw"] = y_pred
    if !isnothing(hyper)
        hyperparams["yaw"] = hyper
    end
    # Position corrections
    predictions["pos"] = Matrix{Float64}(undef, 3, n_samples)
    for d in 1:3
        @info "Computing pos_$d corrections"
        d_hyp = getfield(hyperparameters, Symbol("pos_$d"))
        y_pred, hyper = correction_method(input_feature, outputs["pos"][d, :]; hyperparameter=d_hyp, kwargs...)
        predictions["pos"][d, :] = y_pred
        if !isnothing(hyper)
            hyperparams["pos_$d"] = hyper
        end
    end

    return predictions, hyperparams
end

