"""
    navigation_equations(x, u, q, Ts, g_vec)

Mechanized navigation equations (single step).

# Arguments
- `x`: 9-element vector `[px, py, pz, vx, vy, vz, roll, pitch, yaw]` (old state)
- `u`: 6-element vector `[ax, ay, az, wx, wy, wz]` (IMU measurements)
- `q`: 4-element quaternion scalar-first `[q0, q1, q2, q3]` (old quaternion)
- `Ts`: sampling time (seconds)
- `g_vec`: 3-element gravity vector `[gx, gy, gz]` (e.g., `[0,0,9.81]`)

# Returns
- `y`: new state vector (9 elements)
- `q_new`: updated quaternion
"""
function navigation_equations(x::AbstractVector{T},
    u::AbstractVector{T},
    q::AbstractVector{T},
    Ts::Real,
    g_vec::AbstractVector{T}) where T<:Real

    # ------------------------------------------------------------------
    # 1. Quaternion update from angular rates
    # ------------------------------------------------------------------
    w_tb = u[4:6]          # angular rates [wx, wy, wz]
    P, Q, R = w_tb .* Ts   # scaled angular increments

    # Build OMEGA matrix (scalar-first convention)
    OMEGA = 0.5 * [
        0.0 R -Q P;
        -R 0.0 P Q;
        Q -P 0.0 R;
        -P -Q -R 0.0
    ]

    v = norm(w_tb) * Ts
    if v != 0.0
        q_new = (cos(v / 2) * I(4) + (2 / v) * sin(v / 2) * OMEGA) * q
        q_new ./= norm(q_new)
    else
        q_new = copy(q)
    end

    # ------------------------------------------------------------------
    # 2. Extract Euler angles from the updated quaternion
    # ------------------------------------------------------------------
    R_nb = quat_to_matrix(q_new)   # 3x3 rotation matrix (body-to-navigation)
    attitude = matrix_to_euler(R_nb)

    # ------------------------------------------------------------------
    # 3. Position and velocity update
    # ------------------------------------------------------------------
    # Specific force in navigation frame
    f_t = R_nb * u[1:3]

    # Acceleration in navigation frame (gravity added)
    acc_t = f_t + g_vec

    # State transition matrix A (6x6)
    A = Matrix{T}(I, 6, 6)
    A[1, 4] = Ts
    A[2, 5] = Ts
    A[3, 6] = Ts

    # Input matrix B (6x3)
    B = vcat((Ts^2 / 2) * I(3), Ts * I(3))

    # Update position and velocity (first 6 elements)
    y_pv = A * x[1:6] + B * acc_t

    # Assemble full new state
    y = vcat(y_pv, attitude)

    return y, q_new
end

"""
    state_matrix(q, u, Ts)

Calculate the state transition matrix F and the process noise gain matrix G.

# Arguments
- `q`: 4-element vector `[qx, qy, qz, qw]`
- `u`: 6-element vector `[specific_force (3), angular_rates (3)]`
- `Ts`: sampling time (seconds)

# Returns
- `F`: 9x9 discrete-time state transition matrix
- `G`: 9x6 discrete-time process noise gain matrix
"""
function state_matrix(
    q::AbstractVector{T},
    u::AbstractVector{T},
    Ts::Real
) where T<:Real

    length(q) == 4 || throw(DimensionMismatch("q must have length 4"))
    length(u) == 6 || throw(DimensionMismatch("u must have length 6"))

    # Rotation matrix (body-to-navigation) from quaternion
    R_nb = quat_to_matrix(q)   # 3x3

    # Specific force in navigation frame
    f_n = R_nb * u[1:3]        # 3-element vector

    # Skew-symmetric matrix of f_t
    St = [0.0 -f_n[3] f_n[2];
        f_n[3] 0.0 -f_n[1];
        -f_n[2] f_n[1] 0.0]

    Omat = zeros(T, 3, 3)
    Imat = Matrix{T}(I, 3, 3)

    # Continuous-time state transition matrix (9x9)
    Fc = [Omat Imat Omat;
        Omat Omat St;
        Omat Omat Omat]

    # Continuous-time process noise gain matrix (9x6)
    Gc = [Omat Omat;
        R_nb Omat;
        Omat -R_nb]

    # First-order discrete approximation
    F = I(9) + Ts * Fc
    G = Ts * Gc

    return F, G
end


"""
    comp_internal_states(x_in, dx, q_in)

Correct estimated navigation states with Kalman filter perturbations.

# Arguments
- `x_in`: A priori state vector, length 9: [position (3), velocity (3), Euler angles (3)]
- `dx`:   Error state vector, length 9: [position errors, velocity errors, orientation errors]
- `q_in`: Quaternion vector, `[qx, qy, qz, qw]`

# Returns
- `x_out`: Corrected (posterior) state vector
- `q_out`: Corrected quaternion
"""
function comp_internal_states(
    x_in::AbstractVector{T},
    dx::AbstractVector{T},
    q_in::AbstractVector{T}
) where T<:Real

    # Convert quaternion to rotation matrix (body‑to‑navigation)
    R = quat_to_matrix(q_in)

    # Simple additive correction for position and velocity
    x_out = x_in + dx

    # Orientation errors (small angles)
    ϵ = dx[7:9]

    # Skew‑symmetric matrix of orientation errors
    Ω = skew(ϵ)

    # Correct rotation matrix: R_corrected = (I - Ω) * R
    R_corr = (I(3) - Ω) * R

    # Extract corrected Euler angles [roll, pitch, yaw]
    roll = atan(R_corr[3, 2], R_corr[3, 3])
    pitch = atan(-R_corr[3, 1] / sqrt(R_corr[3, 2]^2 + R_corr[3, 3]^2))
    yaw = atan(R_corr[2, 1], R_corr[1, 1])

    # Overwrite the Euler‑angle part of the state vector
    x_out[7] = roll
    x_out[8] = pitch
    x_out[9] = yaw

    # Compute corrected quaternion (scalar‑first)
    q_out = matrix_to_quat(R_corr)

    return x_out, q_out
end
"""
    compensate_internal_states!(x, dx, quat)

Apply corrections `dx` (negative of the smoothing error) to the state vectors
`x` (9xN) and quaternions `quat` (4xN) in-place.
"""
function compensate_internal_states!(x::AbstractMatrix{T},
    dx::AbstractMatrix{T},
    quat::AbstractMatrix{T}) where T
    for k in axes(x, 2)
        x_corr, q_corr = comp_internal_states(x[:, k], dx[:, k], quat[:, k])
        x[:, k] .= x_corr
        quat[:, k] .= q_corr
    end
    return x, quat
end

"""
    initialize_nav(u, init_heading, init_pos)

Calculate the initial state of the navigation equations.

# Arguments
- `u`: IMU data matrix, size `(6, N)`, rows: `[ax, ay, az, wx, wy, wz]`.
- `init_heading`: initial heading (yaw) in radians.
- `init_pos`: initial position `[x, y, z]` (3-element vector).

# Returns
- `x`: initial navigation state vector `[position (3), velocity (3), Euler angles (3)]`.
- `quat`: quaternion `[qx, qy, qz, qw]` representing the initial attitude.
"""
function initialize_nav(
    u::AbstractMatrix{T},
    init_heading::Real,
    init_pos::AbstractVector{T}
) where T<:Real
    # Use first 20 samples to estimate roll & pitch (assume stationary)
    n_samples = min(20, size(u, 2))
    f_u = mean(u[1, 1:n_samples])
    f_v = mean(u[2, 1:n_samples])
    f_w = mean(u[3, 1:n_samples])

    roll = atan(-f_v, -f_w)
    pitch = atan(f_u, sqrt(f_v^2 + f_w^2))

    attitude = [roll, pitch, init_heading]

    # Rotation matrix from Euler angles, then quaternion
    R_bn = euler_to_matrix(attitude)          # 3x3 matrix
    quat = matrix_to_quat(R_bn)              # returns [q0, q1, q2, q3]

    x = zeros(T, 9)
    x[1:3] = init_pos
    x[7:9] = attitude

    return x, quat
end

# Convenience for integer input
initialize_nav(u::AbstractMatrix{<:Integer}, init_heading, init_pos) =
    initialize_nav(float.(u), init_heading, float.(init_pos))

"""
    init_filter(simdata)

Initialize the Kalman filter matrices from an `InsConfig` object.

# Arguments
- `simdata`: `INSConfig` instance (provides noise standard deviations).

# Returns
- `Q`: 6x6 process noise covariance matrix.
- `R`: 3x3 measurement noise covariance matrix.
- `H`: 3x9 observation matrix (velocity measurement).
"""
function init_filter(simdata::InsConfig)
    sigma_acc = sigma_acc_array(simdata)   # Vector{Float64} length 3
    sigma_gyro = sigma_gyro_array(simdata)
    sigma_vel = sigma_vel_array(simdata)

    # Process noise covariance (6x6)
    Q = zeros(6, 6)
    Q[1:3, 1:3] = Diagonal(sigma_acc .^ 2)
    Q[4:6, 4:6] = Diagonal(sigma_gyro .^ 2)

    # Observation matrix H (3x9): measures velocity states (indices 4:6)
    H = zeros(3, 9)
    H[1:3, 4:6] = Matrix{Float64}(I, 3, 3)

    # Measurement noise covariance (3x3)
    R = Diagonal(sigma_vel .^ 2)

    return Q, R, H
end