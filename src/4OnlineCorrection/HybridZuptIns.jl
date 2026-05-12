"""
    hybrid_nominal_zupt_aided_ins(inertial, simdata, gt_traj, gp_params; x_init)

Run the open-loop zero-velocity aided INS Kalman filter with RTS smoothing,
augmented by an online HSGP-based GP correction when ground truth is available,
and GP prediction when it is not.

# Arguments
- `inertial`:  `InertialData` with fields `t` and `u` (6×N).
- `simdata`:   `InsConfig` object.
- `gt_traj`:   `Trajectory` aligned to the same timestamps as `inertial`.
- `gp_params`: `HsgpParameters` containing hyperparameters, basis-function count,
               feature scaling, and domain half-widths.
- `x_init`:    Optional initial state vector of length 9 (default: zeros).

# Returns
- `zupt`:               Boolean vector of zero-velocity flags (length N).
- `zupt_ins_traj`:      `Trajectory` with estimated positions, orientations, velocities.
- `step_seg`:           Vector of indices where step segments were terminated.
- `y_train`:            Vector of training observations collected during the run.
"""
function hybrid_nominal_zupt_aided_ins(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    gp_params::HsgpParameters;
    x_init::Vector{Float64}=zeros(9)
)
    is_compatible(inertial, gt_traj) ||
        throw(ArgumentError("TimeSeries need to be aligned."))

    u = inertial.u
    N = size(u, 2)
    Ts = simdata.Ts
    g_vec = simdata.g isa Float64 ? [0.0, 0.0, simdata.g] : collect(simdata.g)

    zupt, _ = detect_zupt(u, simdata)

    # ── filter matrices ──────────────────────────────────────────────────────
    Q, R_meas, H = init_filter(simdata)
    I9 = Matrix{Float64}(I, 9, 9)

    # ── allocations ──────────────────────────────────────────────────────────
    x = zeros(9, N)
    quat = zeros(4, N)
    dx = zeros(9, N)
    dx_timeupd = zeros(9, N)
    dx_smooth = zeros(9, N)
    P = zeros(9, 9, N)
    P_timeupd = zeros(9, 9, N)
    P_smooth = zeros(9, 9, N)
    F_store = zeros(9, 9, N)

    # ── initial covariance ───────────────────────────────────────────────────
    P[1:3, 1:3, 1] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
    P[4:6, 4:6, 1] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
    P[7:9, 7:9, 1] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

    # ── initial navigation state ─────────────────────────────────────────────
    x[:, 1] = x_init
    quat[:, 1] = matrix_to_quat(euler_to_matrix(x_init[7:9]))

    # ── HSGP initialisation ──────────────────────────────────────────────────
    outputs_keys = ["yaw", "pos_1", "pos_2", "pos_3"]

    eigvals = calc_eigenvalues(gp_params.LL, gp_params.m, gp_params.d)   # (m, d)
    omega = sqrt.(eigvals)                                               # (m, d)

    # Per-output PSD and weight vectors
    psd = Dict(
        k => power_spectral_density(
            omega,
            getfield(gp_params.hp, Symbol(k))[2],   # length scale
            getfield(gp_params.hp, Symbol(k))[1]    # sigma_f
        )
        for k in outputs_keys
    )
    beta = Dict(k => zeros(gp_params.m) for k in outputs_keys)
    P_beta = Dict(k => Matrix(Diagonal(psd[k])) for k in outputs_keys)

    # Ground truth availability: first 40 % of samples
    train_percent = 0.4
    n_train_cutoff = floor(Int, train_percent * N)
    gt_available = [n <= n_train_cutoff for n in 1:N]   # 1-based

    y_train = Vector{Vector{Float64}}()

    # ── segmentation bookkeeping ─────────────────────────────────────────────
    seg_start = 2          # first index to process (1-based, history is at n-1)
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[]

    # ═════════════════════════════════════════════════════════════════════════
    while true

        # ── forward Kalman filter ─────────────────────────────────────────────
        for n in seg_start:seg_end

            # time update
            x[:, n], quat[:, n] = navigation_equations(
                x[:, n-1], u[:, n], quat[:, n-1], Ts, g_vec
            )
            F_store[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)

            dx[:, n] = F_store[:, :, n] * dx[:, n-1]
            P[:, :, n] = F_store[:, :, n] * P[:, :, n-1] * F_store[:, :, n]' +
                         G * Q * G'

            dx_timeupd[:, n] = dx[:, n]
            P_timeupd[:, :, n] = P[:, :, n]

            # zero-velocity update
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

        # ── RTS smoothing ─────────────────────────────────────────────────────
        dx_smooth[:, seg_end] = dx[:, seg_end]
        P_smooth[:, :, seg_end] = P[:, :, seg_end]

        for n in seg_end-1:-1:seg_start
            A = P[:, :, n] * F_store[:, :, n]' / P_timeupd[:, :, n+1]

            dx_smooth[:, n] = dx[:, n] + A * (dx_smooth[:, n+1] - dx_timeupd[:, n+1])
            P_smooth[:, :, n] = P[:, :, n] +
                                A * (P_smooth[:, :, n+1] - P_timeupd[:, :, n+1]) * A'
            P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n]') / 2
        end

        # ── internal state compensation ───────────────────────────────────────
        idx_range = seg_start:seg_end
        compensate_internal_states!(
            view(x, :, idx_range),
            -dx_smooth[:, idx_range],
            view(quat, :, idx_range)
        )

        # Rotation matrices for the full trajectory (needed for GP features)
        R_nb = euler_to_matrix(x[7:9, :])   # (3, 3, N)

        # ── GP update (online measurement update of HSGP weights) ────────────
        if length(step_seg) > 1
            prev_step = step_seg[end-1]   # seg_start - 1
            curr_step = step_seg[end]     # seg_end

            # INS step vector in body frame
            R_nb_ins_0 = R_nb[:, :, prev_step]
            pos_ins_seg = x[1:3, [prev_step, curr_step]]
            ins_step = R_nb_ins_0' * (pos_ins_seg[:, 2] - pos_ins_seg[:, 1])

            # INS yaw difference over the segment
            euler_ins_seg = matrix_to_euler(R_nb[:, :, prev_step:curr_step])
            yaw_ins_seg = euler_ins_seg[3, :]
            unwrap!(yaw_ins_seg)
            unwrapped_yaw_diff = yaw_ins_seg[end] - yaw_ins_seg[1]

            # Scaled input feature for HSGP
            input_feature = reshape(
                (ins_step .- gp_params.feature_mean) ./ gp_params.feature_std,
                1, gp_params.d          # (1, d) row vector
            )
            eigvect = calc_eigenvectors(input_feature, gp_params.LL, eigvals)   # (1, m)

            # ── measurement update when GT is available ───────────────────────
            if gt_available[prev_step] && gt_available[curr_step]
                R_nb_gt_0 = gt_traj.R_nb[:, :, prev_step]
                pos_gt_seg = gt_traj.pos[:, [prev_step, curr_step]]
                gt_step = R_nb_gt_0' * (pos_gt_seg[:, 2] - pos_gt_seg[:, 1])

                euler_gt_seg = matrix_to_euler(gt_traj.R_nb[:, :, prev_step:curr_step])
                yaw_gt_seg = euler_gt_seg[3, :]
                unwrap!(yaw_gt_seg)
                unwrapped_yaw_diff_gt = yaw_gt_seg[end] - yaw_gt_seg[1]

                y_yaw = unwrapped_yaw_diff_gt - unwrapped_yaw_diff
                y_pos = gt_step - ins_step
                y = vcat([y_yaw], y_pos)   # length 4

                # Kalman update of HSGP weights for each output
                for (idx, outpt) in enumerate(outputs_keys)
                    sigma_n = getfield(gp_params.hp, Symbol(outpt))[3]
                    beta[outpt], P_beta[outpt] = measurement_update(
                        beta[outpt],
                        Matrix(P_beta[outpt]),   # convert Diagonal → dense for measurement_update
                        [y[idx]],
                        eigvect,                 # (1, m) observation matrix
                        fill(sigma_n^2, 1, 1)   # (1, 1) noise covariance
                    )
                end

                push!(y_train, y)

                # Update nominal position with ground truth position
                x[1:3, curr_step] = gt_traj.pos[:, curr_step]
                quat[:, curr_step] = matrix_to_quat(gt_traj.R_nb[:, :, curr_step])
            end

            # ── GP prediction when GT is NOT available ────────────────────────
            if !gt_available[curr_step]
                preds = Vector{Float64}(undef, 4)
                for (idx, outpt) in enumerate(outputs_keys)
                    preds[idx] = (eigvect*beta[outpt])[1]
                end

                # Yaw correction (unwrapped yaw at seg end + predicted offset)
                x[9, curr_step] = yaw_ins_seg[end] + preds[1]

                # Position correction (body-frame offset rotated to nav frame)
                x[1:3, curr_step] = pos_ins_seg[:, 2] + R_nb_ins_0 * preds[2:4]

                # Keep quaternion consistent with updated Euler angles
                quat[:, curr_step] = matrix_to_quat(euler_to_matrix(x[7:9, curr_step]))
            end
        end

        # ── build running trajectory (updated each segment) ───────────────────
        R_nb_final = euler_to_matrix(x[7:9, :])
        zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])

        # ── reset error state and partial covariance ──────────────────────────
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

    # ── final trajectory ──────────────────────────────────────────────────────
    R_nb_final = euler_to_matrix(x[7:9, :])
    zupt_ins_traj = Trajectory(inertial.t, x[1:3, :], R_nb_final, x[4:6, :])

    return zupt, zupt_ins_traj, step_seg, y_train
end