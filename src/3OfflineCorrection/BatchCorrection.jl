"""
    compute_training_io(
    traj::Trajectory,
    traj_gt::Trajectory,
    step_seg::Vector{Int};
    ref_frame::ReferenceFrame = BODY,
    feature_type::FeatureType = THREED_STEP
) ->
(output::CorrectionOutput, input_feature::Matrix{Float64})

Compute training inputs and outputs for GP regression from a pair of trajectories.

# Arguments
- `traj`: Inertial trajectory, temporally aligned with `traj_gt`.
- `traj_gt`: Ground truth trajectory, temporally aligned with `traj`.
- `step_seg`: Indices marking step boundaries in the trajectory (1‑based, length `n_steps`).
- `ref_frame`: Reference frame for step vector computation (`BODY` or `HEADING`).
- `feature_type` : Type of feature to use during the regression (`THREED_STEP` ,`TWOD_STEP_DT`, `THREED_STEP_DT`).

# Returns
- `output`: CorrectionOutput with position (x,y,z) and yaw corrections at step times.
- `input_feature`: Inertial step vectors in `ref_frame`, size varies by feature_type.

# Throws
- `ArgumentError` if trajectories are not temporally compatible (requires `TimeSeries.is_compatible`).
"""
function compute_training_io(
    traj::Trajectory,
    traj_gt::Trajectory,
    step_seg::Vector{Int};
    ref_frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
)::Tuple{CorrectionOutput,Matrix{Float64}}
    if !is_compatible(traj, traj_gt)
        throw(ArgumentError("TimeSeries must be compatible."))
    end
    Nsteps = length(step_seg) - 1

    # Unwrap yaw angles to avoid discontinuities
    euler_nb = matrix_to_euler(traj.R_nb)
    euler_nb_gt = matrix_to_euler(traj_gt.R_nb)

    inertial_yaw = unwrap(euler_nb[3, :])   # row 3 = yaw (since 1: roll, 2: pitch, 3: yaw)
    gt_yaw = unwrap(euler_nb_gt[3, :])

    # Create output array
    outputs = zeros(Float64, (4, Nsteps))

    # Yaw difference over steps: diff(gt) - diff(inertial)
    outputs[4, :] = diff(gt_yaw[step_seg]) - diff(inertial_yaw[step_seg])

    # Select step‑vector function based on reference frame
    funs = Dict(
        BODY => step_vectors_body,
        HEADING => step_vectors_heading,
    )
    if !haskey(funs, ref_frame)
        throw(ArgumentError("Unsupported reference frame: $ref_frame"))
    end

    ins_step = funs[ref_frame](traj, step_seg)      # 3 x (n_steps-1)
    gt_steps = funs[ref_frame](traj_gt, step_seg)

    outputs[1:3, :] = gt_steps - ins_step

    # Feature selectors based on the enum
    feature_selectors = Dict(
        THREED_STEP => () -> ins_step,                             # 3 × N
        TWOD_STEP_DT => () -> vcat(ins_step[1:2, :], diff(traj.t[step_seg])'),          # 3 × N (x, y, dt)
        THREED_STEP_DT => () -> vcat(ins_step, diff(traj.t[step_seg])'),                  # 4 × N (x, y, z, dt)
    )
    input_feature = feature_selectors[feature_type]()

    # Create CorrectionOutput at step boundaries
    correction_output = CorrectionOutput(traj.t[step_seg[1:end-1]], outputs, nothing)

    return correction_output, input_feature
end

"""
    apply_corrections(
        traj::Trajectory,
        corrections::Union{CorrectionOutput, Matrix{Float64}},
        segs::Vector{Int};
        ref_frame::ReferenceFrame = BODY
    ) -> Trajectory

Apply position and yaw corrections to an inertial trajectory, producing a corrected
trajectory sampled at the step segment indices.

# Arguments
- `traj`: Complete inertial trajectory to correct.
- `corrections`: CorrectionOutput or (4 x n_steps) matrix with [pos1, pos2, pos3, yaw] corrections.
- `segs`: Indices marking step boundaries in `traj` (length `n_steps`).
- `ref_frame`: Reference frame for step vector computation (`BODY` or `HEADING`).

# Returns
- Corrected `Trajectory` sampled at the step indices, with updated positions and orientations.
"""
function apply_corrections(
    traj::Trajectory,
    corrections::CorrectionOutput,
    segs::Vector{Int};
    ref_frame::ReferenceFrame=BODY
)::Trajectory
    apply_corrections(traj, corrections.output, segs; ref_frame=ref_frame)
end

function apply_corrections(
    traj::Trajectory,
    predictions::Matrix{Float64},
    segs::Vector{Int};
    ref_frame::ReferenceFrame=BODY
)::Trajectory
    n_steps = length(segs)
    # Extract Euler angles at step boundaries (3 x n_samples)
    euler = matrix_to_euler(traj.R_nb)
    yaw = euler[3, :]
    unwrap!(yaw)

    diff_yaw = diff(yaw[segs]) + predictions[4, :]
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
        pos_out[:, k] = new_R[:, :, k-1] * (steps[:, k-1] + predictions[1:3, k-1]) + pos_out[:, k-1]
    end

    # --- Build output trajectory ------------------------------------------------
    # Final orientation matrices (converted from the corrected Euler angles)
    new_euler = copy(euler[:, segs])
    new_euler[3, :] = new_yaws
    R_nb_new = euler_to_matrix(new_euler)

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
    log_kern_bounds::Vector{Vector{Float64}}=[[-1.0, -3.0], [3.0, 2.0]],
    log_noise_bounds::Vector{Vector{Float64}}=[[-1.71], [-0.1]],
    method::Any=Optim.LBFGS()
)

    # kernbounds covers ONLY kernel params, NOT the GP noise (logNoise)
    # num_params(gp.kernel) gives the right count for kernbounds
    kern_lo = log_kern_bounds[1]
    kern_hi = log_kern_bounds[2]
    noise_lo = log_noise_bounds[1]
    noise_hi = log_noise_bounds[2]


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
        log_noise = noise_lo .+ (noise_hi .- noise_lo) .* rand(length(noise_lo))
        log_noise = clamp.(log_noise, noise_lo, noise_hi)

        @info "Restart $i; starting parameters : log_σ_n = $(log_noise[1]), log_ℓ = $(kern_params[1]), log_σ_f = $(kern_params[2])"
        # set_params! expects [kernel_params..., logNoise]
        GaussianProcesses.set_params!(gp, vcat(log_noise, kern_params))
        GaussianProcesses.update_target!(gp)
        try
            GaussianProcesses.optimize!(gp;
                kernbounds=log_kern_bounds, noisebounds=log_noise_bounds, method=method)
            if gp.target > best_target
                best_target = gp.target
                best_params = GaussianProcesses.get_params(gp)
                @debug "Restart $i: log-likelihood = $best_target"
            end
        catch e
            @warn "Error during optimisation : $e"
        end

    end

    GaussianProcesses.set_params!(gp, best_params)
    GaussianProcesses.update_target!(gp)
    return best_target
end


function compute_gp_corrections(
    x::Matrix{Float64}, y::Vector{Float64};
    kernel::Union{Nothing,GaussianProcesses.Kernel}=nothing,
    hyperparameter::Union{Nothing,Vector{Float64}}=nothing,
    n_restarts_optimizer::Int=5,
    log_kern_bounds::Vector{Vector{Float64}}=[[-1.0, -3.0], [3.0, 2.0]],
    log_noise_bounds::Vector{Vector{Float64}}=[[-1.71], [-0.1]],
    method::Any=Optim.LBFGS(),
    normalize_y::Bool=true,
    normalize_x::Bool=true
)::Tuple{Vector{Float64},Vector{Float64},Matrix{Float64}}
    n_features, n_samples = size(x)
    @assert length(y) == n_samples "x and y must have same number of samples"

    D = 10
    y_testing_gp = zeros(n_samples)
    y_testing_gp_std = zeros(n_samples)
    hyperparams = zeros(D, 4)

    # --- scale features ---------------------------------------------------
    x_μ = mean(x, dims=2)[:]
    x_σ = std(x, dims=2)[:]
    x_σ[x_σ.==0.0] .= 1.0
    x_scaled = normalize_x ? (x .- x_μ) ./ x_σ : x

    # --- scale outptut ---------------------------------------------------
    y_μ = mean(y)
    y_σ = std(y)
    y_σ = y_σ == 0.0 ? 1.0 : y_σ
    y_scaled = normalize_y ? (y .- y_μ) ./ y_σ : y


    log_ℓ, log_σ_f, log_σ_n = 0.0, 0.0, -1.0
    if !isnothing(hyperparameter)
        log_σ_f = log(hyperparameter[1])
        log_ℓ = log(hyperparameter[2])
        log_σ_n = log(hyperparameter[3])
    end

    make_kernel() = kernel === nothing ? GaussianProcesses.SE(log_ℓ, log_σ_f) : deepcopy(kernel)

    # logNoise is handled separately inside the function
    kern_lo = log_kern_bounds[1]
    kern_hi = log_kern_bounds[2]
    noise_lo = log_noise_bounds[1]
    noise_hi = log_noise_bounds[2]

    for i in 1:D
        @info "------------ Fold $i ------------"
        test_start = floor(Int, (i - 1) * n_samples / D) + 1
        test_end = floor(Int, i * n_samples / D)
        train_ind = vcat(1:test_start-1, test_end+1:n_samples)
        test_ind = test_start:test_end

        isempty(train_ind) && continue

        x_train = x_scaled[:, train_ind]
        y_train = y_scaled[train_ind]
        x_test = x_scaled[:, test_ind]

        gp = GaussianProcesses.GP(x_train, y_train,
            GaussianProcesses.MeanZero(), make_kernel(), log_σ_n)
        log_σ_n, log_ℓ, log_σ_f = GaussianProcesses.get_params(gp)
        @info "hyperparameters before optim : log_σ_n = $log_σ_n, log_ℓ = $log_ℓ, log_σ_f = $log_σ_f"
        if n_restarts_optimizer > 0
            optimize_with_restarts!(gp, n_restarts_optimizer;
                log_kern_bounds=log_kern_bounds, log_noise_bounds=log_noise_bounds, method=method)
            # mcmc_with_priors!(gp, n_samples=500, burn_in=100)
        end

        y_pred, y_var = GaussianProcesses.predict_y(gp, x_test)
        y_testing_gp[test_ind] = normalize_y ? y_pred .* y_σ .+ y_μ : y_pred
        y_testing_gp_std[test_ind] = normalize_y ? sqrt.(y_var) .* y_σ : sqrt.(y_var)

        log_σ_n, log_ℓ, log_σ_f = GaussianProcesses.get_params(gp)

        @info "Fold $i Parameters: lg_σ_n = $(round(log_σ_n, digits=3)), lg_ℓ = $(round(log_ℓ, digits=3)), lg_σ_f = $(round(log_σ_f, digits=3))"
        hyperparams[i, :] = [gp.target, exp(log_σ_f), exp(log_ℓ), exp(log_σ_n)]
    end

    return y_testing_gp, y_testing_gp_std, hyperparams
end

"""
    compute_hsgp_corrections(
        x::Matrix{Float64}, y::Vector{Float64};
        m::Int=25, hp::Vector{Float64},
        margin::Float64=1.8
    )

Estimate output corrections using Hilbert Space Gaussian Process (HSGP) approximation
with 10‑fold cross‑validation. Uses fixed kernel hyperparameters (no optimisation).

# Arguments
- `x`: Input features, size `(d, N)` where `d` = dimension, `N` = number of samples.
- `y`: Target values, length `N`.
- `m`: Number of basis functions (eigenvalues) to keep.
- `hp`:  [σ_f, ℓs, σ_n].
- `margin`: Factor to extend the domain beyond the scaled data range.

# Returns
- `predictions`: Vector of length `N` with cross‑validated HSGP predictions.
- `hyperparams`: Dummy matrix `(10, 4)` filled with `[0.0, sigma_f, mean(ls), sigma_n]`
  for tracking; matches the output format of `compute_gp_corrections`.
"""
function compute_hsgp_corrections(
    x::Matrix{Float64}, y::Vector{Float64};
    m::Int=25,
    hyperparameter::Vector{Float64}=[0.0, 0.0, -1.0],
    margin::Float64=1.8,
    normalize_x::Bool=true,
    normalize_y::Bool=true
)::Tuple{Vector{Float64},Vector{Float64},Matrix{Float64}}
    d, N = size(x)
    @assert length(y) == N

    # --- scale features ---------------------------------------------------
    x_μ = mean(x, dims=2)[:]
    x_σ = std(x, dims=2)[:]
    x_σ[x_σ.==0.0] .= 1.0
    x_scaled = normalize_x ? (x .- x_μ) ./ x_σ : x

    # --- scale outptut ---------------------------------------------------
    y_μ = mean(y)
    y_σ = std(y)
    y_σ = y_σ == 0.0 ? 1.0 : y_σ
    y_scaled = normalize_y ? (y .- y_μ) ./ y_σ : y

    # 2. Domain bounds L = margin * max(|x_scaled|) per dimension
    L_vec = margin * maximum(abs, x_scaled, dims=2)[:]   # (d,)
    @show L_vec
    # 3. Compute eigenvalues (per‑dimension components)
    eigvals = calc_eigenvalues(L_vec, m, d)   # (m, d)

    # Pre‑compute sqrt of eigenvalues for PSD
    ω = sqrt.(eigvals)                        # (m, d)

    # 4. Pre‑compute PSD values (depends only on ω, ls, sigma_f)
    psd = power_spectral_density(ω, hyperparameter[2], hyperparameter[1])   # (m,)

    # 5. 10‑fold cross‑validation
    D_folds = 10
    predictions = zeros(N)
    predictions_std = zeros(N)

    for fold in 1:D_folds
        test_start = floor(Int, (fold - 1) * N / D_folds) + 1
        test_end = floor(Int, fold * N / D_folds)
        train_ind = vcat(1:test_start-1, test_end+1:N)
        test_ind = test_start:test_end

        isempty(train_ind) && continue
        n_train = length(train_ind)

        x_train = x_scaled[:, train_ind]    # (d, n_train)
        y_train = y_scaled[train_ind]
        x_test = x_scaled[:, test_ind]     # (d, n_test)

        # 6. Basis matrices
        phi = calc_eigenvectors(x_train', L_vec, eigvals)       # (n_train, m)
        phi_star = calc_eigenvectors(x_test', L_vec, eigvals)   # (n_test, m)

        # 3. Build matrix A = var_n * diag(1/psd) + ΦᵀΦ
        A = hyperparameter[3]^2 * Diagonal(1.0 ./ psd) + phi' * phi                    # (m, m)

        # 4. Solve A α = Φᵀ y
        α = A \ (phi' * y_train)                                         # (m,)

        # 5. Posterior mean
        hsgp_mean = phi_star * α                                         # (N_test,)

        # 6. Posterior covariance: var_n * Φ_* * A^{-1} * Φ_*ᵀ
        # Solve A * W = Φ_*ᵀ  => W = A \ phi_star'
        W = A \ phi_star'                                                # (m, N_test)

        hsgp_var = hyperparameter[3]^2 * vec(sum(phi_star .* W', dims=2))         # (N_test,)
        hsgp_std = sqrt.(hsgp_var)

        predictions[test_ind] = normalize_y ? hsgp_mean .* y_σ .+ y_μ : hsgp_mean
        predictions_std[test_ind] = normalize_y ? hsgp_std .* y_σ : hsgp_std
    end

    # 10. Dummy hyperparameter matrix (for interface compatibility)
    dummy_hyper = zeros(D_folds, 4)
    for i in 1:D_folds
        dummy_hyper[i, :] = vcat([0.0], hyperparameter[:])
    end

    return predictions, predictions_std, dummy_hyper
end

function compute_static_corrections(
    x::Matrix{Float64},
    y::Vector{Float64};
    hyperparameter::Union{Nothing,Vector{Float64}}=nothing,
    n_restarts_optimizer::Int=5
)::Tuple{Vector{Float64},Nothing,Nothing}
    D = 10
    _, n_samples = size(x)
    # y_pred = fill(0.0, n_samples)
    # for i in 1:D
    #     test_start = floor(Int, (i - 1) * n_samples / D) + 1
    #     test_end = floor(Int, i * n_samples / D)
    #     train_ind = vcat(1:test_start-1, test_end+1:n_samples)
    #     test_ind = test_start:test_end
    #     y_pred[test_ind] = fill(mean(y[train_ind]), length(test_ind))
    # end
    y_pred = fill(mean(y), n_samples)
    return y_pred, nothing, nothing
end
"""
    compute_corrections(
        input_feature::Matrix{Float64},
        outputs::CorrectionOutput,
        correction_method::Function,
        hyperparameters::Union{Nothing,SeHyperparams}=nothing;
        feature_type::FeatureType=THREED_STEP,
        kwargs...
    ) -> Tuple{CorrectionOutput, Union{Nothing, Vector{SeHyperparams}}}

Apply a given correction method (e.g. GP, static mean) to the yaw and each position
component, returning predictions as CorrectionOutput and hyperparameters.

# Arguments
- `input_feature`: input features, size `(n_features, n_samples)`.
- `outputs`: Target values, either CorrectionOutput.
- `correction_method`: function with signature `(x::Matrix{Float64}, y::Vector{Float64}; kwargs...) -> (predictions::Vector{Float64}, predictions_std, hyperparams::Any)`.
- `hyperparameters`: Optional SeHyperparams for fixed kernel parameters.
- `feature_type`: Type of input features (THREED_STEP, TWOD_STEP_DT, THREED_STEP_DT).
- `kwargs`: Extra keyword arguments passed to `correction_method`.

# Returns
- `predictions`: CorrectionOutput with position (x,y,z) and yaw corrections, including uncertainties if available.
- `hyperparameters`: Vector of SeHyperparams structs (one per fold/method invocation).
"""
function compute_corrections(
    input_feature::Matrix{Float64},
    outputs::CorrectionOutput,
    correction_method::Function,
    hyperparameters::Union{Nothing,SeHyperparams}=nothing;
    feature_type::FeatureType=THREED_STEP,
    kwargs...
)::Tuple{CorrectionOutput,Union{Nothing,Vector{SeHyperparams}}}

    n_samples = size(input_feature, 2)
    @assert length(outputs) == n_samples

    hp = hyperparameters
    if isnothing(hyperparameters)
        hp = SeHyperparams(
            zeros(Float64, 3), zeros(Float64, 3), zeros(Float64, 3), zeros(Float64, 3)
        )
    end

    predictions = zeros(Float64, size(outputs.output))
    predictions_std = nothing
    hyperparams_dict = Dict{String,Union{Matrix{Float64},Nothing}}()

    # Yaw correction
    @info "Computing Yaw corrections"
    yaw_pred, yaw_std, hyperparams_dict["yaw"] = correction_method(input_feature, outputs.output[4, :]; hyperparameter=hp.yaw, kwargs...)
    predictions[4, :] = yaw_pred
    if !isnothing(yaw_std)
        if isnothing(predictions_std)
            predictions_std = zeros(Float64, size(outputs.output))
        end
        predictions_std[4, :] = yaw_std
    end

    # Position corrections
    for d in 1:3
        @info "Computing pos_$d corrections"
        if feature_type == TWOD_STEP_DT && d == 3 # Avoid ill posed problem in x,y inputs
            hyperparams_dict["pos_$d"] = nothing
            @info "$feature_type , $d"
            continue
        end

        d_hyp = getfield(hp, Symbol("pos_$d"))
        pos_pred, pos_std, hyperparams_dict["pos_$d"] = correction_method(input_feature, outputs.output[d, :]; hyperparameter=d_hyp, kwargs...)
        predictions[d, :] = pos_pred
        if !isnothing(pos_std)
            if isnothing(predictions_std)
                predictions_std = zeros(Float64, size(outputs.output))
            end
            predictions_std[d, :] = pos_std
        end
    end

    # Convert hyperparams to list of SeHyperparams structs per fold
    hyperparams = _convert_hyperparams_to_struct_list(hyperparams_dict)

    # Create and return CorrectionOutput
    pred_output = CorrectionOutput(outputs.t, predictions, predictions_std)
    return pred_output, hyperparams
end


