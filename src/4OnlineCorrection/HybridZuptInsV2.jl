function hybrid_zupt_aided_insv2(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    corrector::AbstractCorrector;
    step_detector::AbstractSegmentDetector=StepDetector(),
    x_init::Vector{Float64}=zeros(9),
    gt_available::Vector{Bool}=zeros(Bool, length(gt_traj)),
    ref_frame::ReferenceFrame=HEADING,
    feature_type::FeatureType=THREED_STEP
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
    ΔP = zeros(Float64, 9, 9)

    P[1:3, 1:3, 1] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
    P[4:6, 4:6, 1] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
    P[7:9, 7:9, 1] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

    x[:, 1] = x_init
    quat[:, 1] = matrix_to_quat(euler_to_matrix(x_init[7:9]))

    initialize_corrector!(corrector;
        t=inertial.t[1],
        pos_init=x_init[1:3],
        quat_init=quat[:, 1],
        Σ_init=P[[1:3; 7:9], [1:3; 7:9], 1],
    )

    # n_train_cutoff = floor(Int, train_ratio * N)
    # gt_available = [n <= n_train_cutoff for n in 1:N]
    # @show typeof(gt_available)

    seg_start = 2
    seg_end = N
    step_seg = Int[1]

    # training_inputs = CorrectionIO(d_feat, true)
    # training_outputs = CorrectionIO(p, true)
    # pred_outputs = CorrectionIO(p, true)

    while true
        # ------------------- Step Covariance Reset -------------------------
        ΔP = zeros(Float64, 9, 9)

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

        dx[:, seg_end] .= 0.0
        P[1:2, 9, seg_end] .= 0.0
        P[9, 1:2, seg_end] .= 0.0

        if seg_end != N
            seg_start = seg_end + 1
            seg_end = N
        else
            break # Break early as to not repeat next part on last index
        end

        prev_step = step_seg[end-1]
        curr_step = step_seg[end]

        @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------"

        # Compute yaw change from x
        yaw_ins_seg = unwrap(x[9, prev_step:curr_step])
        Δθ3 = yaw_ins_seg[end] - yaw_ins_seg[1]

        dynamic_update!(corrector;
            t=inertial.t[curr_step],
            Δp=x[1:3, curr_step] - x[1:3, prev_step],
            Δq=quat_multiply(quat_conjugate(quat[:, prev_step]), quat[:, curr_step]),
            Σpq=ΔP[[1:3; 7:9], [1:3; 7:9]]
        )

        if gt_available[curr_step] && gt_available[prev_step]

        elseif gt_available[curr_step]

        else

        end

        relinearize!(corrector)
    end

    R_nb_final = euler_to_matrix(x[7:9, :])
    zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])
    corr_traj = get_trajectory(corrector)

    return zupt, zupt_ins_traj, step_seg, corr_traj
end