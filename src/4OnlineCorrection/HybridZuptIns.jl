function hybrid_zupt_aided_ins(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    params::HsgpParameters;
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

    # Feature dimensionality driven by feature_type (same as before)
    d_feat = feature_type == THREED_STEP ? 3 :
             feature_type == TWOD_STEP_DT ? 3 :   # 2 step dims + Δt
             feature_type == THREED_STEP_DT ? 4 : 3

    seg_start = 2
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[]

    # HSGP variables
    p = 4
    per_dim_eigvals = calc_eigenvalues(params.LL, params.m, params.d)
    J_ϕ = zeros(params.d, params.m)
    omega = sqrt.(per_dim_eigvals)
    psd = zeros(Float64, p * params.m)

    for (idx, field) in enumerate(fieldnames(SeHyperparams))
        psd[((idx-1)*params.m+1):(idx*params.m)] = power_spectral_density(
            omega,
            getfield(params.hp, field)[2],
            getfield(params.hp, field)[1]
        )
    end
    output_names = ["pos_1", "pos_2", "pos_3", "yaw"]
    beta = zeros(Float64, p * params.m)
    beta_hist = Vector{Float64}[]
    P_beta = Matrix(Diagonal(psd))
    # P_beta = I(p * params.m) .* 1e-3
    # Allocate memory
    R_aug_GT_nb = zeros(Float64, (p, p))
    R_aug_nb = zeros(Float64, (p, p))
    R_aug_GT_nb[p, p] = 1.0
    R_aug_nb[p, p] = 1.0

    # Define parameters 
    sigma_pos_gt = 1e-2
    sigma_ψ_gt = 1e-3
    sigma_gt = vcat(fill(sigma_pos_gt, 3), sigma_ψ_gt) # σ_GTx σ_GTy σ_GT_z σ_GTψ
    sigma_dt = 1e-4

    training_inputs = CorrectionIO(d_feat, true)
    training_outputs = CorrectionIO(p, true)
    pred_outputs = CorrectionIO(p, true)

    while true
        # ------------------- Step Covariance Reset -------------------------
        ΔP = zeros(Float64, 9, 9)
        ΔP[1:3, 1:3] = Matrix(Diagonal(sigma_initial_pos_array(simdata) .^ 2))
        ΔP[4:6, 4:6] = Matrix(Diagonal(sigma_initial_vel_array(simdata) .^ 2))
        ΔP[7:9, 7:9] = Matrix(Diagonal(sigma_initial_att_array(simdata) .^ 2))

        # ------------------- ZUPT aided INS Loop -------------------------
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

        dx_smooth[:, seg_end] = dx[:, seg_end]
        P_smooth[:, :, seg_end] = P[:, :, seg_end]

        for n in seg_end-1:-1:seg_start
            A = P[:, :, n] * F_store[:, :, n]' / P_timeupd[:, :, n+1]
            dx_smooth[:, n] = dx[:, n] + A * (dx_smooth[:, n+1] - dx_timeupd[:, n+1])
            P_smooth[:, :, n] = P[:, :, n] +
                                A * (P_smooth[:, :, n+1] - P_timeupd[:, :, n+1]) * A'
            P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n]') / 2
        end

        compensate_internal_states!(
            view(x, :, seg_start:seg_end),
            -dx_smooth[:, seg_start:seg_end],
            view(quat, :, seg_start:seg_end)
        )

        # ------------------- HSGP stepwise correction ------------------------
        if length(step_seg) > 1 && seg_end != N
            prev_step = step_seg[end-1]
            curr_step = step_seg[end]

            R_nb = quat_to_matrix(quat)
            R_aug_nb[1:3, 1:3] = R_nb[:, :, prev_step]
            R_aug_GT_nb[1:3, 1:3] = gt_traj.R_nb[:, :, prev_step]

            # ------------------- Construct target estimate -------------------------
            # Unwrap INS yaw
            yaw_ins_seg = unwrap(matrix_to_euler(R_nb[:, :, prev_step:curr_step])[3, :])
            step_ins_ψ = yaw_ins_seg[end] - yaw_ins_seg[1]

            # Unwrap GT yaw
            yaw_gt_seg = unwrap(matrix_to_euler(gt_traj.R_nb[:, :, prev_step:curr_step])[3, :])
            step_gt_ψ = yaw_gt_seg[end] - yaw_gt_seg[1]

            # Construct target estimate
            step_ins_aug = R_aug_nb' * vcat(x[1:3, curr_step] - x[1:3, prev_step], step_ins_ψ)
            step_gt_aug = R_aug_GT_nb' * vcat(gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step], step_gt_ψ)
            target = step_gt_aug - step_ins_aug

            # Normalize target 
            target_norm = (target - params.output_stats[1]) ./ params.output_stats[2]

            # ---------- Construct target covariance ------------
            step_ins_aug_cov = R_aug_nb' * ΔP[[1, 2, 3, 9], [1, 2, 3, 9]] * R_aug_nb
            step_gt_aug_cov = R_aug_GT_nb' * Diagonal(sigma_gt .^ 2) * R_aug_GT_nb

            # Normalize target covariance
            target_cov_norm = Diagonal(1 ./ params.output_stats[2]) * (step_ins_aug_cov + step_gt_aug_cov) * Diagonal(1 ./ params.output_stats[2])

            # ----------- Construct Input Feature --------------------
            feature_estimate_selector = Dict{FeatureType,Function}(
                THREED_STEP => () -> step_ins_aug[1:3],
                TWOD_STEP_DT => () -> vcat(step_ins_aug[1:2], inertial.t[curr_step] - inertial.t[prev_step]),
                THREED_STEP_DT => () -> vcat(step_ins_aug[1:3], inertial.t[curr_step] - inertial.t[prev_step]),
            )
            # Normalise estimated feature 
            feature_estimate_norm = (feature_estimate_selector[feature_type]() .- params.input_stats[1]) ./ params.input_stats[2]

            # ------------ Construct Feature Covariance ---------
            feature_cov_selector = Dict{FeatureType,Function}(
                THREED_STEP => () -> step_ins_aug_cov[1:3, 1:3],
                TWOD_STEP_DT => () -> [
                    step_ins_aug_cov[1:2, 1:2] zeros(Float64, (2, 1));
                    zeros(Float64, (1, 2)) sigma_dt^2
                ],
                THREED_STEP_DT => () -> [
                    step_ins_aug_cov[1:3, 1:3] zeros(Float64, (3, 1));
                    zeros(Float64, (1, 3)) sigma_dt^2
                ],
            )
            # Normalise covariance of input feature
            feature_cov_norm = Diagonal(1 ./ params.input_stats[2]) *
                               feature_cov_selector[feature_type]() * Diagonal(1 ./ params.input_stats[2])
            for d in axes(J_ϕ, 1)
                J_ϕ[d, :] = calc_eigenvectors_dx(
                    reshape(feature_estimate_norm, 1, params.d), params.LL, per_dim_eigvals, d)
            end
            B_estim = reshape(beta, (params.m, p))
            input_cov_norm = (J_ϕ * B_estim)' * feature_cov_norm * (J_ϕ * B_estim)
            tt_target_cov_norm = target_cov_norm + input_cov_norm

            # ---------- Construct measurement matrix H_update --------------
            eigvect = calc_eigenvectors(
                reshape(feature_estimate_norm, 1, params.d), params.LL, per_dim_eigvals)
            H_update = kron(I(p), eigvect)

            if correct && gt_available[prev_step] && gt_available[curr_step]
                noise_vect = Vector{Float64}(undef, p)
                for (idx, key) in enumerate(output_names)
                    noise_vect[idx] = getfield(params.hp, Symbol(key))[3]
                end
                beta, P_beta = measurement_update(
                    beta, P_beta, target_norm, H_update, Diagonal(noise_vect .^ 2)
                )
                push!(beta_hist, beta)
            end
            # ----------- Save target & Input ------------
            append_io!(training_inputs, inertial.t[prev_step], feature_estimate_norm, sqrt.(diag(feature_cov_norm)))
            append_io!(training_outputs, inertial.t[prev_step], target_norm, sqrt.(diag(tt_target_cov_norm)))

            # --------- Update the Position and orientation from GT ---------------
            if gt_available[prev_step] && gt_available[curr_step]
                # Construct Measurement matrix H_gt 
                H_gt = [
                    I zeros(Float64, (3, 6));
                    zeros(Float64, (1, 8)) 1.0
                ]

                # δx is zero since has been compensated after zupts
                dx[:, curr_step], P[:, :, curr_step] = measurement_update(
                    zeros(Float64, 9),
                    P[:, :, curr_step],
                    vcat(gt_traj.pos[:, curr_step], yaw_gt_seg[end]) - vcat(x[[1, 2, 3], curr_step], yaw_ins_seg[end]),
                    H_gt,
                    Diagonal(sigma_gt .^ 2)
                )

                # -------- Compensate error -------------
                x[:, curr_step], quat[:, curr_step] = comp_internal_states(
                    x[:, curr_step],
                    dx[:, curr_step],
                    quat[:, curr_step]
                )
            end

            if !gt_available[curr_step] && correct

                R_nb = quat_to_matrix(quat)
                R_aug_nb[1:3, 1:3] = R_nb[:, :, prev_step]

                # Unwrap INS yaw
                yaw_ins_seg = unwrap(matrix_to_euler(R_nb[:, :, prev_step:curr_step])[3, :])
                step_ins_ψ = yaw_ins_seg[end] - yaw_ins_seg[1]

                # Construct augmented step
                step_ins_aug = R_aug_nb' * vcat(x[1:3, curr_step] - x[1:3, prev_step], step_ins_ψ)

                # ----------- Construct Input Feature --------------------
                feature_estimate_selector = Dict{FeatureType,Function}(
                    THREED_STEP => () -> step_ins_aug[1:3],
                    TWOD_STEP_DT => () -> vcat(step_ins_aug[1:2], inertial.t[curr_step] - inertial.t[prev_step]),
                    THREED_STEP_DT => () -> vcat(step_ins_aug[1:3], inertial.t[curr_step] - inertial.t[prev_step]),
                )
                # Normalise estimated feature 
                feature_estimate_norm = (feature_estimate_selector[feature_type]() .- params.input_stats[1]) ./ params.input_stats[2]

                # ------------ Construct Covariance from Input Feature uncertainty ---------
                # Step covariance 
                step_ins_aug_cov = R_aug_nb' * ΔP[[1, 2, 3, 9], [1, 2, 3, 9]] * R_aug_nb

                feature_cov_selector = Dict{FeatureType,Function}(
                    THREED_STEP => () -> step_ins_aug_cov[1:3, 1:3],
                    TWOD_STEP_DT => () -> [
                        step_ins_aug_cov[1:2, 1:2] zeros(Float64, (2, 1));
                        zeros(Float64, (1, 2)) sigma_dt^2
                    ],
                    THREED_STEP_DT => () -> [
                        step_ins_aug_cov[1:3, 1:3] zeros(Float64, (3, 1));
                        zeros(Float64, (1, 3)) sigma_dt^2
                    ],
                )
                # Normalise covariance of input feature
                feature_cov_norm = Diagonal(1 ./ params.input_stats[2]) * feature_cov_selector[feature_type]() * Diagonal(1 ./ params.input_stats[2])

                for d in axes(J_ϕ, 1)
                    J_ϕ[d, :] = calc_eigenvectors_dx(
                        reshape(feature_estimate_norm, 1, params.d), params.LL, per_dim_eigvals, d)
                end
                B_estim = reshape(beta, (params.m, p))
                input_cov_norm = (J_ϕ * B_estim)' * feature_cov_norm * (J_ϕ * B_estim)

                # ------------- Construct measurement matrix H_correction --------------------
                H_correction = [
                    Matrix(I(3)) zeros(Float64, (3, 6));
                    zeros(Float64, (1, 8)) 1
                ]

                # ------------- Predict correction -----------------
                eigvect = calc_eigenvectors(
                    reshape(feature_estimate_norm, 1, params.d), params.LL, per_dim_eigvals)

                # Compute prediction estimate
                Φ = kron(I(p), eigvect)
                pred_estim_norm = Φ * beta

                # Compute prediction covariance
                pred_cov_norm = Φ * P_beta * Φ' + input_cov_norm

                # -------- Save prediction --------------
                append_io!(pred_outputs, inertial.t[prev_step], pred_estim_norm, sqrt.(diag(pred_cov_norm)))

                # -------- Process prediction for Measurement Update --------------
                pred_estim = pred_estim_norm .* params.output_stats[2] .+ params.output_stats[1]
                y_estim = R_aug_nb * pred_estim

                pred_cov = Diagonal(params.output_stats[2]) * (pred_cov_norm) * Diagonal(params.output_stats[2])
                y_cov = R_aug_nb * pred_cov * R_aug_nb'

                # δx is zero since has been compensated after zupts
                dx[:, curr_step], _ = measurement_update(
                    zeros(Float64, 9),
                    P[:, :, curr_step],
                    y_estim,
                    H_correction,
                    y_cov # .* 1e-3
                )

                # -------- Compensate error -------------
                x[:, curr_step], quat[:, curr_step] = comp_internal_states(
                    x[:, curr_step],
                    dx[:, curr_step],
                    quat[:, curr_step]
                )

            end
        end

        # R_nb_final = euler_to_matrix(x[7:9, :])
        # zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])

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

    return zupt, zupt_ins_traj, step_seg, training_outputs, pred_outputs, beta_hist, training_inputs
end