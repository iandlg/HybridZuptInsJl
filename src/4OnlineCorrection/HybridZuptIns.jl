function hybrid_zupt_aided_ins(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    params::HsgpParameters;
    x_init::Vector{Float64}=zeros(9),
    ref_frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
    gt_available::Vector{Bool}=zeros(Bool, length(gt_traj)),
    correct::Bool=true,
    cov_update::Bool=true,
    R_inflation::Float64=1.0,                    # y_cov scale factor for the sweep
    measurement_source::Symbol=:gp,              # :gp | :oracle | :oracle_noisy
    zero_xcov::Bool=true,                        # ablation for the P[1:2,9] zeroing
    diag_rec::StepDiagnostics=StepDiagnostics()
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

    # Feature dimensionality driven by feature_type (same as before)
    d_feat = feature_type == THREED_STEP ? 3 :
             feature_type == TWOD_STEP_DT ? 3 :   # 2 step dims + Δt
             feature_type == THREED_STEP_DT ? 4 : 3

    seg_start = 2
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[1]

    # HSGP variables
    p = 4
    per_dim_eigvals = calc_eigenvalues(params.LL, params.m, params.d)
    Φ = zeros(p, p * params.m)
    J_ϕ = zeros(params.d, params.m)
    omega = sqrt.(per_dim_eigvals)
    psd = zeros(Float64, p * params.m)

    for (idx, field) in enumerate(fieldnames(SeHyperparams))
        psd[((idx-1)*params.m+1):(idx*params.m)] = power_spectral_density(
            omega,
            getfield(params.hp, field)[2],
            getfield(params.hp, field)[3]
        )
    end
    output_names = ["pos_1", "pos_2", "pos_3", "yaw"]
    beta = zeros(Float64, p * params.m)
    beta_hist = Vector{Float64}[]
    P_beta = Matrix(Diagonal(psd))
    # P_beta = I(p * params.m) .* 1e-3

    # Define parameters
    sigma_gt = sigma_groundtruth_array(simdata) # σ_GTx σ_GTy σ_GT_z σ_GTψ

    training_inputs = CorrectionIO(d_feat, true)
    training_outputs = CorrectionIO(p, true)
    pred_outputs = CorrectionIO(p, true)

    ΔP = Matrix{Float64}(undef, 9, 9)

    var_hist = Vector{Float64}[]
    pred_nis = Vector{Float64}[]

    while true
        # ------------------- Step Covariance Reset -------------------------
        if ismissing(ΔP[1, 1])
            ΔP = zeros(Float64, 9, 9)
            ΔP[1:3, 1:3] = Matrix(Diagonal(sigma_initial_pos_array(simdata) .^ 2))
            ΔP[4:6, 4:6] = Matrix(Diagonal(sigma_initial_vel_array(simdata) .^ 2))
            ΔP[7:9, 7:9] = Matrix(Diagonal(sigma_initial_att_array(simdata) .^ 2))
        else
            ΔP = zeros(Float64, 9, 9)
        end
        # ΔP[1:3, 1:3] = Matrix(Diagonal(sigma_initial_pos_array(simdata) .^ 2))
        # ΔP[4:6, 4:6] = Matrix(Diagonal(sigma_initial_vel_array(simdata) .^ 2))
        # ΔP[7:9, 7:9] = Matrix(Diagonal(sigma_initial_att_array(simdata) .^ 2))

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
            push!(var_hist, diag(P[:, :, n]))

            if zupt[n]
                S = H * P[:, :, n] * H' + R_meas
                ΔS = H * ΔP * H' + R_meas
                K = P[:, :, n] * H' / S
                ΔK = ΔP * H' / ΔS

                ν_z = x[4:6, n] .- dx[4:6, n]          # innovation, dof = 3
                push!(diag_rec.zupt_k, n)
                push!(diag_rec.zupt_nis, mahalanobis(ν_z, S))

                dx[:, n] = dx[:, n] - K * (dx[4:6, n] - x[4:6, n])
                P[:, :, n] = (I9 - K * H) * P[:, :, n]
                ΔP = (I9 - ΔK * H) * ΔP
            end

            P[:, :, n] = (P[:, :, n] + P[:, :, n]') / 2
            ΔP = (ΔP + ΔP') / 2
            push!(var_hist, diag(P[:, :, n]))

            if update!(step_detector, zupt[n])
                push!(step_seg, n)
                seg_end = n
                break
            end
        end

        dx_smooth[:, seg_end] = dx[:, seg_end]
        P_smooth[:, :, seg_end] = P[:, :, seg_end]

        for n in (seg_end-1):-1:seg_start
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
        prev_step = step_seg[end-1]
        curr_step = step_seg[end]

        @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------ " maxlog = 5

        # ------------------- HSGP stepwise correction ------------------------
        if length(step_seg) > 1 && seg_end != N

            # ------------------- Compute stride error & INS stride -------------------
            stride_err, Σ_err, ins_stride, Σ_ins_stride, R_aug_wl = stride_error(ref_frame;
                R_wb=(quat_to_matrix(quat[:, prev_step]), quat_to_matrix(quat[:, curr_step])),
                Δp=x[1:3, curr_step] - x[1:3, prev_step],
                Σ_ΔpΔθ3=ΔP[[1, 2, 3, 9], [1, 2, 3, 9]],
                R_wb_gt=(gt_traj.R_nb[:, :, prev_step], gt_traj.R_nb[:, :, curr_step]),
                Δp_gt=gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step],
                Σ_ΔpΔθ3_gt=Diagonal(sigma_gt .^ 2)
            )
            target = stride_err

            # Normalize target
            target_norm = (target .- params.output_stats[1]) ./ params.output_stats[2]

            # Normalize target covariance
            target_cov_norm = Diagonal(1 ./ params.output_stats[2]) * Σ_err * Diagonal(1 ./ params.output_stats[2])

            # ----------- Construct & normalize Input Feature from the INS stride --------------------
            feature, feature_cov = compute_feature(feature_type;
                ins_stride=ins_stride, Σ_ins_stride=Σ_ins_stride,
                ΔT=inertial.t[curr_step] - inertial.t[prev_step]
            )
            feature_estimate_norm, feature_cov_norm = normalize_feature!(feature_type;
                feature=feature, Σ_feature=feature_cov,
                input_stats=params.input_stats, mid_norm=params.mid_norm
            )
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
            kron!(Φ, I(p), eigvect)

            if correct && gt_available[prev_step] && gt_available[curr_step]
                noise_vect = Vector{Float64}(undef, p)
                for (idx, key) in enumerate(output_names)
                    noise_vect[idx] = getfield(params.hp, Symbol(key))[1]
                end
                α = tr(Diagonal(noise_vect .^ 2)) / tr(tt_target_cov_norm)
                R = Diagonal(noise_vect .^ 2) + α * tt_target_cov_norm
                beta, P_beta = measurement_update(
                    beta, P_beta, target_norm, Φ, tt_target_cov_norm # Diagonal(noise_vect .^ 2)
                )
                push!(beta_hist, beta)
            end
            # ----------- Save target & Input ------------
            append_io!(training_inputs, inertial.t[prev_step], feature_estimate_norm, sqrt.(diag(feature_cov_norm)))
            append_io!(training_outputs, inertial.t[prev_step], target_norm, sqrt.(diag(tt_target_cov_norm)))

            # --------- Update the Position and orientation from GT ---------------

            if gt_available[prev_step] && gt_available[curr_step]

                H_gt = [
                    I zeros(Float64, (3, 6));
                    zeros(Float64, (1, 6)) ∂θ3_∂δθ_right(quat[:, curr_step])'
                ]

                yaw_gt_curr = matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3]
                yaw_ins_curr = matrix_to_euler(quat_to_matrix(quat[:, curr_step]))[3]

                # (nominal - true), matching the ZUPT convention
                Δθ = atan(sin(yaw_ins_curr - yaw_gt_curr),
                    cos(yaw_ins_curr - yaw_gt_curr))
                z_gt = vcat(x[1:3, curr_step] .- gt_traj.pos[:, curr_step], Δθ)

                # prior dx is zero: already compensated after the ZUPT smoother
                dx[:, curr_step], P[:, :, curr_step] = measurement_update(
                    zeros(Float64, 9),
                    P[:, :, curr_step],
                    z_gt,
                    H_gt,
                    Diagonal(sigma_gt .^ 2) #.* 10e1
                )
                push!(var_hist, diag(P[:, :, curr_step]))

                # dx is (nominal - true) -> SUBTRACT it, as the ZUPT path does
                x[:, curr_step], quat[:, curr_step] = comp_internal_states(
                    x[:, curr_step],
                    -dx[:, curr_step],
                    quat[:, curr_step]
                )
            end




            if !gt_available[curr_step] && correct

                H_correction = [
                    Matrix(I(3)) zeros(Float64, (3, 6));
                    zeros(Float64, (1, 6)) ∂θ3_∂δθ_right(quat[:, curr_step])'
                ]

                # ---- GP prediction, in TRAINING convention (true - nominal) --
                # The training pipeline is untouched: `target`, `target_norm`,
                # `training_outputs` and `pred_outputs` all stay in the sense
                # that stride_error() produced. Only the filter-facing copy is
                # converted, below.
                pred_estim_norm = Φ * beta
                pred_cov_norm = Φ * P_beta * Φ' + input_cov_norm

                append_io!(pred_outputs, inertial.t[prev_step],
                    pred_estim_norm, sqrt.(diag(pred_cov_norm)))

                pred_estim = pred_estim_norm .* params.output_stats[2] .+
                             params.output_stats[1]
                pred_cov = Diagonal(params.output_stats[2]) * pred_cov_norm *
                           Diagonal(params.output_stats[2])

                # ---- convert to filter convention: nominal - true ------------
                y_estim = -(R_aug_wl * pred_estim)
                y_cov = R_aug_wl * pred_cov * R_aug_wl'
                y_cov = Symmetric((y_cov + y_cov') / 2) * R_inflation

                # ---- ground-truth references, same convention, eval only -----
                y_true_stride = -(R_aug_wl * target)

                yaw_gt_c = matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3]
                yaw_ins_c = matrix_to_euler(quat_to_matrix(quat[:, curr_step]))[3]
                e_state_true = vcat(
                    x[1:3, curr_step] .- gt_traj.pos[:, curr_step],
                    atan(sin(yaw_ins_c - yaw_gt_c), cos(yaw_ins_c - yaw_gt_c))
                )

                # ---- oracle control -----------------------------------------
                y_meas = if measurement_source === :oracle
                    y_true_stride
                elseif measurement_source === :oracle_noisy
                    y_true_stride .+
                    cholesky(Symmetric(Matrix(y_cov))).L * randn(length(y_true_stride))
                else
                    y_estim
                end

                P_pre = copy(P[:, :, curr_step])
                S = H_correction * P_pre * H_correction' + Matrix(y_cov)

                if cov_update
                    dx[:, curr_step], P[:, :, curr_step] = measurement_update(
                        zeros(Float64, 9), P[:, :, curr_step],
                        y_meas, H_correction, Matrix(y_cov))
                else
                    dx[:, curr_step], _ = measurement_update(
                        zeros(Float64, 9), P[:, :, curr_step],
                        y_meas, H_correction, Matrix(y_cov))
                end

                record_step!(diag_rec;
                    k=curr_step, t=inertial.t[curr_step],
                    y_meas=y_meas, S=S,
                    y_hat=y_estim, y_cov=Matrix(y_cov),
                    y_true_stride=y_true_stride, e_state_true=e_state_true,
                    P_pre=P_pre, P_post=P[:, :, curr_step])

                push!(var_hist, diag(P[:, :, curr_step]))

                # dx is (nominal - true) -> SUBTRACT it
                x[:, curr_step], quat[:, curr_step] = comp_internal_states(
                    x[:, curr_step],
                    -dx[:, curr_step],
                    quat[:, curr_step]
                )
            end


        end

        # R_nb_final = euler_to_matrix(x[7:9, :])
        # zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])

        dx[:, seg_end] .= 0.0
        if zero_xcov
            P[1:2, 9, seg_end] .= 0.0
            P[9, 1:2, seg_end] .= 0.0
        end

        if seg_end != N
            seg_start = seg_end + 1
            seg_end = N
        else
            break
        end
    end

    R_nb_final = euler_to_matrix(x[7:9, :])
    zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])

    return zupt, zupt_ins_traj, step_seg, training_outputs, pred_outputs,
    beta_hist, training_inputs, var_hist, diag_rec.pred_nis_stride,
    diag_rec, quat, x, P
end