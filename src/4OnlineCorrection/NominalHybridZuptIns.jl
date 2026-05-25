function hybrid_nominal_zupt_aided_ins(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory;
    corrector::Union{Nothing,AbstractNominalCorrector}=nothing,
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

    true_outputs = Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}}(
        "output" => Vector{Float64}[], "t" => Float64[]
    )
    pred_outputs = Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}}(
        "output" => Vector{Float64}[], "t" => Float64[]
    )

    beta_hist = Vector{Float64}[]

    seg_start = 2
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[]

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

            # Feature selectors based on the enum
            feature_selectors = Dict(
                THREED_STEP => () -> [ins_step[1], ins_step[2], ins_step[3]],                             # 3 × N
                TWOD_STEP_DT => () -> [ins_step[1], ins_step[2], inertial.t[curr_step] - inertial.t[prev_step]],
                THREED_STEP_DT => () -> [ins_step[1], ins_step[2], ins_step[3], inertial.t[curr_step] - inertial.t[prev_step]],
            )
            input_feature = feature_selectors[feature_type]()

            euler_ins_seg = matrix_to_euler(R_nb[:, :, prev_step:curr_step])
            yaw_ins_seg = euler_ins_seg[3, :]
            unwrap!(yaw_ins_seg)
            unwrapped_yaw_diff = yaw_ins_seg[end] - yaw_ins_seg[1]

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
            y = vcat(y_pos, y_yaw)

            push!(true_outputs["output"], y)
            push!(true_outputs["t"], inertial.t[prev_step])
            # Update with training data when gt is supposedly available
            if gt_available[prev_step] && gt_available[curr_step]

                if !isnothing(corrector)
                    update_corrector!(corrector, input_feature, y)
                    beta = Float64[]
                    if typeof(corrector) == HsgpCorrector
                        for (idx, key) in enumerate(corrector.outputs_keys)
                            beta = vcat(beta, corrector.beta[key])
                        end
                        @info "saved β : " beta
                        push!(beta_hist, beta)
                    end
                end

                # Update the Position and orientation using ground truth
                x[1:3, curr_step] = gt_traj.pos[:, curr_step]
                quat[:, curr_step] = matrix_to_quat(gt_traj.R_nb[:, :, curr_step])
            end

            if !gt_available[curr_step] && !isnothing(corrector)
                preds, preds_std = predict_correction(corrector, input_feature)
                if feature_type == TWOD_STEP_DT
                    preds[3] = 0.0
                end
                if !haskey(pred_outputs, "output_std") && !isnothing(preds_std)
                    pred_outputs["output_std"] = Vector{Float64}[]
                end
                if !isnothing(preds_std)
                    push!(pred_outputs["output_std"], preds_std)
                end
                push!(pred_outputs["output"], preds)
                push!(pred_outputs["t"], inertial.t[prev_step])

                x[9, curr_step] = yaw_ins_seg[end] + preds[4]
                x[1:3, curr_step] = pos_ins_seg[:, 2] + R_nb_ins_0 * preds[1:3]
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

    # Remove last incorrect output
    true_outputs["output"] = true_outputs["output"][1:end-1]
    true_outputs["t"] = true_outputs["t"][1:end-1]
    pred_outputs["output"] = pred_outputs["output"][1:end-1]
    if haskey(pred_outputs, "output_std")
        pred_outputs["output_std"] = pred_outputs["output_std"][1:end-1]
    end
    pred_outputs["t"] = pred_outputs["t"][1:end-1]
    true_outputs = CorrectionOutput(true_outputs)
    pred_outputs = CorrectionOutput(pred_outputs)

    beta_hist = hcat(beta_hist...)

    return zupt, zupt_ins_traj, step_seg, true_outputs, pred_outputs, beta_hist
end