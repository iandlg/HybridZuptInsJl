# --------------------------------------------------------------------------
# Step detector – simplified without unused return value
# --------------------------------------------------------------------------
mutable struct StepDetector
    zupt_counter::Int
    no_zupt_counter::Int
    reached_no_zupt_counter::Bool
end
StepDetector() = StepDetector(0, 0, false)

"""
    update!(det::StepDetector, zupt::Bool) -> Bool

Update the detector state with a new zero-velocity flag.
Returns `true` when a segment end should be triggered.
"""
function update!(det::StepDetector, zupt::Bool)
    if zupt
        det.zupt_counter += 1
        det.no_zupt_counter = 0
    else
        det.no_zupt_counter += 1
        det.zupt_counter = 0
    end

    if det.zupt_counter == 10 && det.reached_no_zupt_counter
        det.reached_no_zupt_counter = false
        return true
    end

    if det.no_zupt_counter == 50
        det.reached_no_zupt_counter = true
    end
    return false
end

# --------------------------------------------------------------------------
# Main smoothed ZUPT-aided INS
# --------------------------------------------------------------------------
"""
    smoothed_zupt_aided_ins(inertial, simdata)

Run the open-loop zero-velocity aided INS Kalman filter with RTS smoothing.

# Arguments
- `inertial`: `InertialData` with fields `t::Vector{Float64}` and `u::Matrix{Float64}` (6×N).
- `simdata`: `InsConfig` object.

# Returns
- `zupt`: Boolean vector of zero-velocity flags.
- `traj`: `Trajectory` object containing estimated positions, orientations and velocities.
- `step_seg`: Vector of indices where step segments were terminated.
"""
function smoothed_zupt_aided_ins(
    inertial::InertialData,
    simdata::InsConfig
)
    u = inertial.u
    N = size(u, 2)
    Ts = simdata.Ts
    g_vec = if simdata.g isa Float64
        [0.0, 0.0, simdata.g]
    else
        collect(simdata.g)
    end
    display(g_vec)

    # Zero-velocity detection (assume external function)
    zupt, _ = detect_zupt(u, simdata)   # returns Vector{Bool} of length N

    # Initialise filter matrices
    Q, R, H = init_filter(simdata)
    I9 = Matrix{Float64}(I, 9, 9)

    # Allocate arrays
    x = zeros(9, N)
    quat = zeros(4, N)
    dx = zeros(9, N)
    dx_timeupd = zeros(9, N)
    dx_smooth = zeros(9, N)
    P = zeros(9, 9, N)
    P_timeupd = zeros(9, 9, N)
    P_smooth = zeros(9, 9, N)
    F = zeros(9, 9, N)

    # Initial covariance
    P[1:3, 1:3, 1] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
    P[4:6, 4:6, 1] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
    P[7:9, 7:9, 1] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

    # Initial navigation state
    x[:, 1], quat[:, 1] = initialize_nav(u, simdata.init_heading,
        init_pos_array(simdata))

    # Segmentation bookkeeping
    seg_start = 2          # first index to process (1-based, offset for n-1)
    seg_end = N
    step_detector = StepDetector()
    step_seg = Int[]

    while true
        # ------------------------------------------------------------------
        # Forward Kalman filter over the current segment
        # ------------------------------------------------------------------
        for n in seg_start:seg_end
            # Time update
            x[:, n], quat[:, n] = navigation_equations(
                x[:, n-1], u[:, n], quat[:, n-1], Ts, g_vec
            )
            F[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)


            dx[:, n] = F[:, :, n] * dx[:, n-1]
            P[:, :, n] = F[:, :, n] * P[:, :, n-1] * F[:, :, n]' +
                         G * Q * G'

            dx_timeupd[:, n] = dx[:, n]
            P_timeupd[:, :, n] = P[:, :, n]

            # Zero-velocity update
            if zupt[n]
                S = H * P[:, :, n] * H' + R
                K = P[:, :, n] * H' / S   # equivalent to K = P*H'*inv(S)
                dx[:, n] = dx[:, n] - K * (dx[4:6, n] - x[4:6, n])
                P[:, :, n] = (I9 - K * H) * P[:, :, n]
            end
            if n == 3
                display(dx[:, n])
            end
            # Enforce symmetry
            P[:, :, n] = (P[:, :, n] + P[:, :, n]') / 2

            # Segmentation decision
            if update!(step_detector, zupt[n])
                push!(step_seg, n)
                seg_end = n
                break
            end
        end

        # ------------------------------------------------------------------
        # RTS smoothing (backward pass) over the finished segment
        # ------------------------------------------------------------------
        dx_smooth[:, seg_end] = dx[:, seg_end]
        P_smooth[:, :, seg_end] = P[:, :, seg_end]

        for n in seg_end-1:-1:seg_start
            A = P[:, :, n] * F[:, :, n]' / P_timeupd[:, :, n+1]

            dx_smooth[:, n] = dx[:, n] + A * (dx_smooth[:, n+1] - dx_timeupd[:, n+1])
            P_smooth[:, :, n] = P[:, :, n] +
                                A * (P_smooth[:, :, n+1] - P_timeupd[:, :, n+1]) * A'
            P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n]') / 2
        end

        # ------------------------------------------------------------------
        # Compensate internal states using the smoothed errors
        # ------------------------------------------------------------------
        idx_range = seg_start:seg_end
        compensate_internal_states!(
            view(x, :, idx_range),           # view of columns idx_range
            -dx_smooth[:, idx_range],        # negative copy (or use view if needed)
            view(quat, :, idx_range)         # view of quat columns
        )

        # ------------------------------------------------------------------
        # Prepare for next segment (reset error state and covariance)
        # ------------------------------------------------------------------
        dx[:, seg_end] .= 0.0
        P[1:2, 9, seg_end] .= 0.0   # note: 1-based, element (9,?) – Python indices 0:2,8
        P[9, 1:2, seg_end] .= 0.0

        if seg_end != N
            seg_start = seg_end + 1
            seg_end = N
        else
            break
        end
    end

    # ----------------------------------------------------------------------
    # Build final trajectory output
    # ----------------------------------------------------------------------
    # Rotation matrices from Euler angles (3×3×N)
    R_nb = euler_to_matrix(x[7:9, :])   # assuming euler_to_matrix works on 3×N

    traj = Trajectory(inertial.t, x[1:3, :], R_nb, x[4:6, :])

    return zupt, traj, step_seg
end