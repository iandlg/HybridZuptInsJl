function hybrid_zupt_aided_ins(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    params::HsgpParameters;
    x_init::Vector{Float64}=zeros(9),
    train_ratio::Float64=0.5,
    ref_frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
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

    true_outputs = Dict{String,Any}("yaw" => Float64[], "pos" => Vector{Float64}[], "t" => Float64[])
    pred_outputs = Dict{String,Any}("yaw" => Float64[], "pos" => Vector{Float64}[], "t" => Float64[])

    seg_start = 2
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[]

    # HSGP variables
    outputs_keys = ["yaw", "pos_1", "pos_2", "pos_3"]
    eigvals = calc_eigenvalues(params.LL, params.m, params.d)
    eigvecs_dx = zeros(params.d, params.m)
    omega = sqrt.(eigvals)
    psd = Dict(
        k => power_spectral_density(
            omega,
            getfield(params.hp, Symbol(k))[2],
            getfield(params.hp, Symbol(k))[1]
        )
        for k in outputs_keys
    )
    beta = Dict(k => zeros(params.m) for k in outputs_keys)
    P_beta = Dict(k => Matrix(Diagonal(psd[k])) for k in outputs_keys)

    # define parameters 
    sigma_gt_pos = 1e-3
    sigma_gt_yaw = 1e-3
    sigma_dt = 1e-1


    while true
        for n in seg_start:seg_end
            x[:, n], quat[:, n] = navigation_equations(
                x[:, n-1], u[:, n], quat[:, n-1], Ts, g_vec)
            F_store[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)

            dx[:, n] = F_store[:, :, n] * dx[:, n-1]
            P[:, :, n] = F_store[:, :, n] * P[:, :, n-1] * F_store[:, :, n]' + G * Q * G'
            dx_timeupd[:, n] = dx[:, n]
            P_timeupd[:, :, n] = P[:, :, n]

            if zupt[n]
                S = H * P[:, :, n] * H' + R_meas
                K = P[:, :, n] * H' / S
                dx[:, n] = dx[:, n] - K * (dx[4:6, n] - x[4:6, n])
                P[:, :, n] = (I9 - K * H) * P[:, :, n]
            end

            P[:, :, n] = (P[:, :, n] + P[:, :, n]') / 2

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

        R_nb = euler_to_matrix(x[7:9, :])

        # Correction
        if length(step_seg) > 1
            prev_step = step_seg[end-1]
            curr_step = step_seg[end]

            R_nb_ins_0 = R_nb[:, :, prev_step]
            pos_ins_seg = x[1:3, [prev_step, curr_step]]
            ins_step = R_nb_ins_0' * (pos_ins_seg[:, 2] - pos_ins_seg[:, 1])
            P_step = R_nb_ins_0' * P_smooth[1:3, 1:3, curr_step] * R_nb_ins_0
            P_step_gt = I(3) .* sigma_gt_pos^2

            # Time difference between consecutive step indices
            Δt = inertial.t[curr_step] - inertial.t[prev_step]
            # Feature selectors based on the enum
            feature_estimate_selector = Dict{FeatureType,Vector{Float64}}(
                THREED_STEP => () -> [ins_step[1], ins_step[2], ins_step[3]],                             # 3 × N
                TWOD_STEP_DT => () -> [ins_step[1], ins_step[2], Δt],
                THREED_STEP_DT => () -> [ins_step[1], ins_step[2], ins_step[3], Δt],
            )
            feature_covariance_selector = Dict{FeatureType,Matrix{Float64}}(
                THREED_STEP => () -> P_step,
                TWOD_STEP_DT => () -> [
                    P_step[1:2, 1:2] zeros(Float64, (2, 1));
                    zeros(Float64, (1, 2)) sigma_dt^2
                ],
                THREED_STEP_DT => () -> [
                    P_step zeros(Float64, (3, 1));
                    zeros(Float64, (1, 3)) sigma_dt^2
                ],
            )
            feature_estimate = feature_estimate_selector[feature_type]()
            feature_covariance = feature_covariance_selector[feature_type]()

            euler_ins_seg = matrix_to_euler(R_nb[:, :, prev_step:curr_step])
            yaw_ins_seg = euler_ins_seg[3, :]
            unwrap!(yaw_ins_seg)
            unwrapped_yaw_diff = yaw_ins_seg[end] - yaw_ins_seg[1]
            var_yaw_curr = P_smooth[9, 9, curr_step]

            # Compute training data 
            R_nb_gt_0 = gt_traj.R_nb[:, :, prev_step]
            pos_gt_seg = gt_traj.pos[:, [prev_step, curr_step]]
            gt_step = R_nb_gt_0' * (pos_gt_seg[:, 2] - pos_gt_seg[:, 1])

            euler_gt_seg = matrix_to_euler(gt_traj.R_nb[:, :, prev_step:curr_step])
            yaw_gt_seg = euler_gt_seg[3, :]
            unwrap!(yaw_gt_seg)
            unwrapped_yaw_diff_gt = yaw_gt_seg[end] - yaw_gt_seg[1]

            y_yaw = unwrapped_yaw_diff_gt - unwrapped_yaw_diff
            y_pos = gt_step - ins_step
            y = vcat([y_yaw], y_pos)

            push!(true_outputs["yaw"], y_yaw)
            push!(true_outputs["pos"], y_pos)
            push!(true_outputs["t"], inertial.t[prev_step])

            # Update with training data when gt is supposedly available
            if gt_available[prev_step] && gt_available[curr_step]
                # Normalise output
                y_norm = copy(y)
                y_norm[2:4] = (y[2:4] .- params.pos_stats[1]) ./ params.pos_stats[2]
                y_norm[1] = (y[1] - params.yaw_stats[1]) / params.yaw_stats[2]

                # Normalize uncertainty
                cov_y_pos_norm = Diagonal(1 ./ params.pos_stats[2]) *
                                 (P_step + P_step_gt) * Diagonal(1 ./ params.pos_stats[2])
                cov_y_yaw_norm = (var_yaw_curr + sigma_gt_yaw^2) / params.yaw_stats[2]^2

                # Normalize input
                feat_estim_norm = (feature_estimate .- params.input_stats[1]) ./ params.input_stats[2]
                feat_cov_norm = Diagonal(1 ./ params.input_stats[2]) *
                                (feature_covariance) * Diagonal(1 ./ params.input_stats[2])

                # Usefull quantities
                for d in 1:params.d
                    eigvecs_dx[d, :] = calc_eigenvectors_dx(feat_estim_norm, params.LL, eigvals, d)
                end

                # Compute input uncertainty

                eigvect = _build_eigvect(c, input)
                for (idx, outpt) in enumerate(c.outputs_keys)
                    sigma_n = getfield(c.params.hp, Symbol(outpt))[3]
                    c.beta[outpt], c.P_beta[outpt] = measurement_update(
                        c.beta[outpt],
                        c.P_beta[outpt],
                        [y_normalized[idx]],
                        eigvect,
                        fill(sigma_n^2, 1, 1)
                    )
                end

                # Update the Position and orientation using ground truth
                x[1:3, curr_step] = gt_traj.pos[:, curr_step]
                quat[:, curr_step] = matrix_to_quat(gt_traj.R_nb[:, :, curr_step])
            end

            if !gt_available[curr_step] && !isnothing(corrector)
                preds = predict_correction(corrector, input_feature)
                if feature_type == TWOD_STEP_DT
                    preds[4] = 0.0
                end
                push!(pred_outputs["yaw"], preds[1])
                push!(pred_outputs["pos"], preds[2:4])
                push!(pred_outputs["t"], inertial.t[prev_step])

                x[9, curr_step] = yaw_ins_seg[end] + preds[1]
                x[1:3, curr_step] = pos_ins_seg[:, 2] + R_nb_ins_0 * preds[2:4]
                quat[:, curr_step] = matrix_to_quat(euler_to_matrix(x[7:9, curr_step]))
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

    # Convert outputs to Dict{String,VecOrMat{Float64}}
    true_outputs_typed = Dict{String,VecOrMat{Float64}}(
        "yaw" => true_outputs["yaw"],
        "pos" => isempty(true_outputs["pos"]) ? Matrix{Float64}(undef, 3, 0) : hcat(true_outputs["pos"]...),
        "t" => true_outputs["t"]
    )
    pred_outputs_typed = Dict{String,VecOrMat{Float64}}(
        "yaw" => pred_outputs["yaw"],
        "pos" => isempty(pred_outputs["pos"]) ? Matrix{Float64}(undef, 3, 0) : hcat(pred_outputs["pos"]...),
        "t" => pred_outputs["t"]
    )

    return zupt, zupt_ins_traj, step_seg, true_outputs_typed, pred_outputs_typed
end