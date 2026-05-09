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
    pos_out[:, 1] = traj.pos[:, segs[1]]   # first step position (1‑based index)

    for k in 2:n_steps
        pos_out[:, k] = new_R[:, :, k-1] * (steps[:, k-1] + pos_correction[:, k-1]) + pos_out[:, k-1]
    end

    # --- Build output trajectory ------------------------------------------------
    # Final orientation matrices (converted from the corrected Euler angles)
    R_nb_new = euler_to_matrix(new_euler)
    @show size(traj.t[segs])
    return Trajectory(
        traj.t[segs],      # time at step boundaries
        pos_out,           # 3 x n_steps
        R_nb_new,          # 3 x 3 x n_steps
        traj.vel === nothing ? nothing : traj.vel[:, segs]
    )
end


function optimize_with_restarts!(gp::GaussianProcesses.GPE, n_restarts::Int;
    lo::Vector{Float64}=fill(-3.0, GaussianProcesses.num_params(gp.kernel)),
    hi::Vector{Float64}=fill(3.0, GaussianProcesses.num_params(gp.kernel)))

    # kernbounds covers ONLY kernel params, NOT the GP noise (logNoise)
    # num_params(gp.kernel) gives the right count for kernbounds
    @assert length(lo) == length(hi) == GaussianProcesses.num_params(gp.kernel) """
        Bounds length ($(length(lo))) must match kernel param count \
        ($(GaussianProcesses.num_params(gp.kernel))), not total GP params \
        ($(length(GaussianProcesses.get_params(gp)))).
    """

    best_target = -Inf
    best_params = GaussianProcesses.get_params(gp)   # includes logNoise

    for i in 1:n_restarts
        try
            # Sample random kernel params within bounds
            kern_params = lo .+ (hi .- lo) .* rand(length(lo))
            kern_params = clamp.(kern_params, lo, hi)   # guarantee inside bounds

            # logNoise is separate — sample in its own range
            log_noise = -8.0 + 10.0 * rand()   # uniform in [-8, 2]

            # set_params! expects [kernel_params..., logNoise]
            GaussianProcesses.set_params!(gp, vcat(log_noise, kern_params))
            GaussianProcesses.update_target!(gp)
            GaussianProcesses.optimize!(gp; kernbounds=[lo, hi])

            if gp.target > best_target
                best_target = gp.target
                best_params = GaussianProcesses.get_params(gp)
                @debug "Restart $i: log-likelihood = $best_target"
            end
        catch e
            @warn "Restart $i failed: $e"
        end
    end

    GaussianProcesses.set_params!(gp, best_params)
    GaussianProcesses.update_target!(gp)
    return best_target
end


function compute_gp_corrections(
    x::Matrix{Float64}, y::Vector{Float64};
    kernel::Union{Nothing,GaussianProcesses.Kernel}=nothing,
    n_restarts_optimizer::Int=5,
    hyperparameter::Union{Nothing,Vector{Any}}=nothing
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
    lo = [-4.0, -4.0]
    hi = [4.0, 4.0]

    for i in 1:D
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
        @info "hyperparameters before optim : log_σ_n = $log_σ_n, log_ℓ = $log_ℓ, log_σ_f = $log_σ_f"
        if n_restarts_optimizer > 0
            optimize_with_restarts!(gp, n_restarts_optimizer; lo=lo, hi=hi)
        end

        y_pred, _ = GaussianProcesses.predict_y(gp, x_test)
        y_testing_gp[test_ind] = y_pred

        # params = [log(σ_n), log(ℓ), log(σ_f)]  — extract directly
        log_σ_n, log_ℓ, log_σ_f = GaussianProcesses.get_params(gp)

        @info "hyperparameters after optim : log_σ_n = $log_σ_n, log_ℓ = $log_ℓ, log_σ_f = $log_σ_f"
        hyperparams[i, :] = [gp.target, exp(log_σ_f), exp(log_ℓ), exp(log_σ_n)]
        #                     lml        sigma_f        len_scale    sigma_n
    end

    return y_testing_gp, hyperparams
end


function compute_static_corrections(
    x::Matrix{Float64},
    y::Vector{Float64},
    hyperparameter::Union{Nothing,Vector{Any}}=nothing
)
    return fill(mean(y), size(y)), hyperparameter
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
    correction_method::Function;
    hyperparameters::Union{Dict{String,Vector{Float64}}}=nothing
)

    n_samples = size(input_feature, 2)
    @assert length(outputs["yaw"]) == n_samples
    @assert size(outputs["pos"], 2) == n_samples

    predictions = Dict{String,Union{Vector{Float64},Matrix{Float64}}}()
    hyperparams = Dict{String,Union{Matrix{Float64},Nothing}}()

    # Yaw correction
    @info "Computing Yaw corrections"
    y_pred, hyper = correction_method(input_feature, outputs["yaw"], hyperparameters["yaw"])
    predictions["yaw"] = y_pred
    hyperparams["yaw"] = hyper

    # Position corrections
    predictions["pos"] = Matrix{Float64}(undef, 3, n_samples)
    for d in 1:3
        @info "Computing pos_$d corrections"
        y_pred, hyper = correction_method(input_feature, outputs["pos"][d, :], hyperparameters["pos_$d"])
        predictions["pos"][d, :] = y_pred
        hyperparams["pos_$d"] = hyper
    end

    return predictions, hyperparams
end