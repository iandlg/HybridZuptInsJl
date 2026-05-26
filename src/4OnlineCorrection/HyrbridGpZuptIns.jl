function hybrid_zupt_aided_ins_gp(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    params::HsgpParameters;   # params still used for input/output_stats and noise hyperparams
    x_init::Vector{Float64}=zeros(9),
    train_ratio::Float64=0.5,
    ref_frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
    correct::Bool=true
)
    is_compatible(inertial, gt_traj) ||
        throw(ArgumentError("TimeSeries need to be aligned."))

    u = inertial.u
    N = size(u, 2)
    Ts = simdata.Ts
    g_vec = simdata.g isa Float64 ? [0.0, 0.0, simdata.g] : collect(simdata.g)

    zupt, _ = detect_zupt(u, simdata)

    Q, R_meas, H = init_filter(simdata)
    I9 = Matrix{Float64}(I, 9, 9)

    x = zeros(9, N)
    quat = zeros(4, N)
    dx = zeros(9, N)
    dx_timeupd = zeros(9, N)
    dx_smooth = zeros(9, N)
    P = zeros(9, 9, N)
    P_timeupd = zeros(9, 9, N)
    P_smooth = zeros(9, 9, N)
    F_store = zeros(9, 9, N)

    P[1:3, 1:3, 1] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
    P[4:6, 4:6, 1] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
    P[7:9, 7:9, 1] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

    x[:, 1] = x_init
    quat[:, 1] = matrix_to_quat(euler_to_matrix(x_init[7:9]))

    n_train_cutoff = floor(Int, train_ratio * N)
    gt_available = [n <= n_train_cutoff for n in 1:N]

    true_outputs = Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}}(
        "output" => Vector{Float64}[], "t" => Float64[]
    )
    pred_outputs = Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}}(
        "output" => Vector{Float64}[], "t" => Float64[]
    )
    training_inputs = Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}}(
        "input" => Vector{Float64}[], "t" => Float64[]
    )
    training_targets = Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}}(
        "output" => Vector{Float64}[], "t" => Float64[]
    )

    seg_start = 2
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[]

    # ── GP setup (replaces HSGP beta/P_beta/eigenfunction machinery) ────────────
    p = 4   # number of outputs: pos_1, pos_2, pos_3, yaw
    output_names = ["pos_1", "pos_2", "pos_3", "yaw"]

    # Feature dimensionality driven by feature_type (same as before)
    d_feat = feature_type == THREED_STEP ? 3 :
             feature_type == TWOD_STEP_DT ? 3 :   # 2 step dims + Δt
             feature_type == THREED_STEP_DT ? 4 : 3

    # One independent GP per output dimension, SE kernel, log-parameterised.
    # Noise level taken from params.hp as before (field order = output_names).
    function make_gp(out_idx::Int)
        field = Symbol(output_names[out_idx])
        log_sf = log(getfield(params.hp, field)[1])   # signal std 
        log_ls = log(getfield(params.hp, field)[2])   # length scale
        log_noise = log(getfield(params.hp, field)[3]) # noise std
        GaussianProcesses.GPE(; mean=GaussianProcesses.MeanZero(), kernel=GaussianProcesses.SE(log_ls, log_sf), logNoise=log_noise)
    end

    gps = [make_gp(i) for i in 1:p]

    # Accumulated (normalised) training data
    X_train = Matrix{Float64}(undef, d_feat, 0)   # d_feat × n_obs
    Y_train = Matrix{Float64}(undef, p, 0)   # p       × n_obs

    # Define parameters
    sigma_pos_gt = 1e-2
    sigma_ψ_gt = 1e-3
    sigma_gt = vcat(fill(sigma_pos_gt, 3), sigma_ψ_gt)
    sigma_dt = 1e-5

    R_aug_GT_nb = zeros(Float64, (p, p))
    R_aug_GT_nb[p, p] = 1.0
    R_aug_nb = zeros(Float64, (p, p))
    R_aug_nb[p, p] = 1.0

    training_inputs = CorrectionIO(d_feat, false)
    training_outputs = CorrectionIO(p, true)


    while true
        ΔP = zeros(Float64, 9, 9)
        ΔP[1:3, 1:3] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
        ΔP[4:6, 4:6] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
        ΔP[7:9, 7:9] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

        for n in seg_start:seg_end
            x[:, n], quat[:, n] = navigation_equations(
                x[:, n-1], u[:, n], quat[:, n-1], Ts, g_vec)
            F_store[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)

            dx[:, n] = F_store[:, :, n] * dx[:, n-1]
            P[:, :, n] = F_store[:, :, n] * P[:, :, n-1] * F_store[:, :, n]' + G * Q * G'
            ΔP = F_store[:, :, n] * ΔP * F_store[:, :, n]' + G * Q * G'
            dx_timeupd[:, n] = dx[:, n]
            P_timeupd[:, :, n] = P[:, :, n]

            if zupt[n]
                S = H * P[:, :, n] * H' + R_meas
                ΔS = H * ΔP * H' + R_meas
                K = P[:, :, n] * H' / S
                ΔK = ΔP * H' / ΔS
                dx[:, n] = dx[:, n] - K * (dx[4:6, n] - x[4:6, n])
                P[:, :, n] = (I9 - K * H) * P[:, :, n]
                ΔP = (I9 - ΔK * H) * ΔP
            end

            P[:, :, n] = (P[:, :, n] + P[:, :, n]') / 2
            ΔP = (ΔP + ΔP') / 2

            if update!(step_detector, zupt[n])
                push!(step_seg, n)
                seg_end = n
                break
            end
        end

        # ── RTS smoother (unchanged) ──────────────────────────────────────────
        dx_smooth[:, seg_end] = dx[:, seg_end]
        P_smooth[:, :, seg_end] = P[:, :, seg_end]
        for n in seg_end-1:-1:seg_start
            A = P[:, :, n] * F_store[:, :, n]' / P_timeupd[:, :, n+1]
            dx_smooth[:, n] = dx[:, n] + A * (dx_smooth[:, n+1] - dx_timeupd[:, n+1])
            P_smooth[:, :, n] = P[:, :, n] + A * (P_smooth[:, :, n+1] - P_timeupd[:, :, n+1]) * A'
            P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n]') / 2
        end

        compensate_internal_states!(
            view(x, :, seg_start:seg_end),
            -dx_smooth[:, seg_start:seg_end],
            view(quat, :, seg_start:seg_end)
        )

        # ── Per-step correction ───────────────────────────────────────────────
        if length(step_seg) > 1
            prev_step = step_seg[end-1]
            curr_step = step_seg[end]

            R_nb = euler_to_matrix(x[7:9, :])
            R_aug_nb[1:3, 1:3] = R_nb[:, :, prev_step]
            R_aug_GT_nb[1:3, 1:3] = gt_traj.R_nb[:, :, prev_step]

            yaw_ins_seg = unwrap(matrix_to_euler(R_nb[:, :, prev_step:curr_step])[3, :])
            step_ins_ψ = yaw_ins_seg[end] - yaw_ins_seg[1]
            yaw_gt_seg = unwrap(matrix_to_euler(gt_traj.R_nb[:, :, prev_step:curr_step])[3, :])
            step_gt_ψ = yaw_gt_seg[end] - yaw_gt_seg[1]

            step_ins_aug = R_aug_nb' * vcat(x[1:3, curr_step] - x[1:3, prev_step], step_ins_ψ)
            step_gt_aug = R_aug_GT_nb' * vcat(gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step], step_gt_ψ)
            target = step_gt_aug - step_ins_aug

            # Normalise target
            target_norm = (target .- params.output_stats[1]) ./ params.output_stats[2]

            # Target covariance (for saving / diagnostics)
            step_ins_aug_cov = R_aug_nb' * ΔP[[1, 2, 3, 9], [1, 2, 3, 9]] * R_aug_nb
            step_gt_aug_cov = R_aug_GT_nb' * Diagonal(sigma_gt .^ 2) * R_aug_GT_nb
            target_cov_norm = Diagonal(1 ./ params.output_stats[2]) *
                              (step_ins_aug_cov + step_gt_aug_cov) *
                              Diagonal(1 ./ params.output_stats[2])

            # ── Build normalised input feature ───────────────────────────────
            feature_raw = if feature_type == THREED_STEP
                step_ins_aug[1:3]
            elseif feature_type == TWOD_STEP_DT
                vcat(step_ins_aug[1:2], inertial.t[curr_step] - inertial.t[prev_step])
            else  # THREED_STEP_DT
                vcat(step_ins_aug[1:3], inertial.t[curr_step] - inertial.t[prev_step])
            end
            feature_norm = (feature_raw .- params.input_stats[1]) ./ params.input_stats[2]

            # ── Save diagnostics (target + std) ──────────────────────────────
            marg_std = sqrt.(diag(target_cov_norm))
            push!(true_outputs["t"], inertial.t[prev_step])
            push!(true_outputs["output"], target)
            if !haskey(true_outputs, "output_std")
                true_outputs["output_std"] = Vector{Float64}[]
            end
            push!(true_outputs["output_std"], marg_std)

            # ── GP training update (replaces beta/P_beta measurement_update) ─
            if correct && gt_available[prev_step] && gt_available[curr_step]
                # Accumulate normalised observations
                X_train = hcat(X_train, feature_norm)       # d_feat × n
                Y_train = hcat(Y_train, target_norm)        # p      × n

                append_io!(training_inputs, inertial.t[prev_step], feature_norm)
                append_io!(training_outputs, inertial.t[prev_step], target_norm, sqrt.(diag(target_cov_norm)))

                # Refit all GPs on the growing dataset
                for i in 1:p
                    GaussianProcesses.fit!(gps[i], X_train, Y_train[i, :])
                end
            end

            # ── GT position/yaw update (unchanged) ───────────────────────────
            H_gt = [I zeros(Float64, (3, 6)); zeros(Float64, (1, 8)) 1.0]

            if gt_available[prev_step] && gt_available[curr_step]
                dx[:, curr_step], P[:, :, curr_step] = measurement_update(
                    zeros(Float64, 9),
                    P[:, :, curr_step],
                    vcat(gt_traj.pos[:, curr_step], yaw_gt_seg[end]) -
                    vcat(x[[1, 2, 3], curr_step], yaw_ins_seg[end]),
                    H_gt,
                    Diagonal(sigma_gt .^ 2)
                )
                x[:, curr_step], quat[:, curr_step] = comp_internal_states(
                    x[:, curr_step], dx[:, curr_step], quat[:, curr_step])
            end

            # ── GP prediction + correction (replaces HSGP predict block) ─────
            if !gt_available[curr_step] && correct && size(X_train, 2) > 0

                R_nb = euler_to_matrix(x[7:9, :])
                R_aug_nb[1:3, 1:3] = R_nb[:, :, prev_step]

                yaw_ins_seg = unwrap(matrix_to_euler(R_nb[:, :, prev_step:curr_step])[3, :])
                step_ins_ψ = yaw_ins_seg[end] - yaw_ins_seg[1]
                step_ins_aug = R_aug_nb' * vcat(x[1:3, curr_step] - x[1:3, prev_step], step_ins_ψ)

                feature_raw = if feature_type == THREED_STEP
                    step_ins_aug[1:3]
                elseif feature_type == TWOD_STEP_DT
                    vcat(step_ins_aug[1:2], inertial.t[curr_step] - inertial.t[prev_step])
                else
                    vcat(step_ins_aug[1:3], inertial.t[curr_step] - inertial.t[prev_step])
                end
                feature_norm = (feature_raw .- params.input_stats[1]) ./ params.input_stats[2]

                # Query each GP: predict_f returns (mean_vec, var_vec) for a matrix of test pts
                x_test = reshape(feature_norm, d_feat, 1)   # d_feat × 1
                pred_mean_norm = zeros(p)
                pred_var_norm = zeros(p)
                for i in 1:p
                    μ, σ² = GaussianProcesses.predict_f(gps[i], x_test)
                    pred_mean_norm[i] = μ[1]
                    pred_var_norm[i] = σ²[1]
                end

                # Denormalise
                pred_estim = pred_mean_norm .* params.output_stats[2] .+ params.output_stats[1]
                pred_std = sqrt.(pred_var_norm) .* params.output_stats[2]

                # Rotate back to navigation frame
                y_estim = R_aug_nb * pred_estim
                y_cov = R_aug_nb * Diagonal(pred_std .^ 2) * R_aug_nb'

                # EKF measurement update with GP prediction as pseudo-measurement
                H_correction = [Matrix(I(3)) zeros(Float64, (3, 6)); zeros(Float64, (1, 8)) 1.0]
                dx[:, curr_step], P[:, :, curr_step] = measurement_update(
                    zeros(Float64, 9),
                    P[:, :, curr_step],
                    y_estim,
                    H_correction,
                    y_cov .* 1e-4
                )
                x[:, curr_step], quat[:, curr_step] = comp_internal_states(
                    x[:, curr_step],
                    dx[:, curr_step],
                    quat[:, curr_step]
                )

                # Save prediction
                push!(pred_outputs["t"], inertial.t[prev_step])
                push!(pred_outputs["output"], y_estim)
                if !haskey(pred_outputs, "output_std")
                    pred_outputs["output_std"] = Vector{Float64}[]
                end
                push!(pred_outputs["output_std"], sqrt.(diag(y_cov)))
            end
        end

        dx[:, seg_end] .= 0.0
        P[1:2, 9, seg_end] .= 0.0
        P[9, 1:2, seg_end] .= 0.0

        if seg_end != N
            seg_start = seg_end + 1
            seg_end = N
        else
            break
        end
    end

    R_nb_final = euler_to_matrix(x[7:9, :])
    zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])

    # Trim the trailing spurious entry (same logic as original)
    for d in (true_outputs, pred_outputs)
        for k in keys(d)
            if d[k] isa Vector && !isempty(d[k])
                d[k] = d[k][1:end-1]
            end
        end
    end

    return zupt, zupt_ins_traj, step_seg,
    CorrectionIO(true_outputs), CorrectionIO(pred_outputs),
    training_inputs, training_outputs
end