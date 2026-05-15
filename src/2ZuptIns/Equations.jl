function integrate_quaternion(q::Vector{Float64}, w::Vector{Float64}, Ts::Float64)::Vector{Float64}
    v = norm(w) * Ts
    v ≈ 0.0 && return q

    dq = [cos(v / 2); (w / norm(w)) .* sin(v / 2)]   # incremental rotation quaternion
    q_new = quat_multiply(q, dq)                   # q ⊗ dq  (body-frame, Hamilton)
    return q_new ./ norm(q_new)
end

"""
    navigation_equations(x, u, q, Ts, g_vec)

Mechanized navigation equations of the inertial navigation system.

# Arguments
- `x`: 9-element vector `[position (3), velocity (3), euler angles (3)]`
- `u`: 6-element vector `[specific_force (3), angular_rates (3)]`
- `q`: 4-element quaternion **scalar-first** `[q0, q1, q2, q3]`
- `Ts`: sampling time (seconds)
- `g_vec`: 3-element gravity vector `[gx, gy, gz]` (e.g., `[0,0,9.81]`)

# Returns
- `y`: new navigation state `[position (3), velocity (3), euler angles (3)]`
- `q`: updated quaternion (scalar-first)
"""
function navigation_equations(
    x::AbstractVector{T},
    u::AbstractVector{T},
    q::AbstractVector{T},
    Ts::Real,
    g_vec::AbstractVector{T}) where T<:Real

    # Ensure floats and copy quaternion for in-place update
    xf = float.(x)
    uf = float.(u)
    qf = float.(q)   # scalar-first

    # -------- 1. Quaternion update from angular rates -----------------
    qf = integrate_quaternion(qf, uf[4:6], Ts)

    # -------- 2. Euler angles from the updated quaternion ------------
    # Convert scalar-first -> vector-first for quat_to_dcm
    R_nb = quat_to_matrix(qf)               # 3x3 rotation matrix
    euler = matrix_to_euler(R_nb)          # [roll, pitch, yaw]

    # -------- 3. Position & velocity update --------------------------
    f_t = R_nb * uf[1:3]     # specific force in navigation frame
    acc_t = f_t + g_vec         # add gravity

    # Build state transition matrix A (6x6)
    A = Matrix{Float64}(I, 6, 6)
    A[1, 4] = Ts
    A[2, 5] = Ts
    A[3, 6] = Ts

    # Build input matrix B (6x3)
    B = vcat((
            Ts^2 / 2) * I(3),
        Ts * I(3)
    )

    # New position/velocity (first 6 elements)
    y_pv = A * xf[1:6] + B * acc_t

    # Assemble full output state
    y = vcat(y_pv, euler)

    return y, qf
end


"""
    state_matrix(q, u, Ts)

Calculate the state transition matrix F and the process noise gain matrix G.

# Arguments
- `q`: 4-element vector `[qw, qx, qy, qz]` - quaternion (scalar first)
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
    Rb2t = quat_to_matrix(q)   # 3x3

    # Specific force in navigation frame
    f_t = Rb2t * u[1:3]        # 3-element vector

    # Skew-symmetric matrix of f_t
    St = -Rb2t * skew(u[1:3])

    Omat = zeros(T, 3, 3)
    Imat = Matrix{T}(I, 3, 3)

    # Continuous-time state transition matrix (9x9)
    Fc = [Omat Imat Omat;
        Omat Omat St;
        Omat Omat Omat]

    # Continuous-time process noise gain matrix (9x6)
    Gc = [Omat Omat;
        Imat Omat;
        Omat Imat]

    # First-order discrete approximation
    F = I(9) + Ts * Fc
    F[7:9, 7:9] = exp(skew(u[4:6] * Ts))'
    G = Ts * Gc

    return F, G
end

"""
    comp_internal_states(x_in, dx, q_in)

Correct estimated navigation states with Kalman filter perturbations.

# Arguments
- `x_in`: A priori state vector, length 9: [position (3), velocity (3), Euler angles (3)]
- `dx`:   Error state vector, length 9: [position errors, velocity errors, orientation errors]
- `q_in`: Quaternion vector, `[qw, qx, qy, qz]`

# Returns
- `x_out`: Corrected (posterior) state vector
- `q_out`: Corrected quaternion
"""
function comp_internal_states(
    x_in::AbstractVector{T},
    dx::AbstractVector{T},
    q_in::AbstractVector{T}
) where T<:Real

    # Convert quaternion to rotation matrix (body-to-navigation)
    R = quat_to_matrix(q_in)

    # Simple additive correction for position and velocity
    x_out = x_in + dx

    # Orientation errors (small angles)
    dq = matrix_to_quat(euler_to_matrix(dx[7:9]))

    # Skew-symmetric matrix of orientation errors
    # Ω = skew(dx[7:9])

    # Correct rotation matrix: R_corrected = (I - Ω) * R
    # R_corr = (I(3) - Ω) * R

    # Overwrite the Euler-angle part of the state vector
    # x_out[7:9] = matrix_to_euler(R_corr)

    # Compute corrected quaternion (scalar-first)
    q_out = quat_multiply(q_in, dq)

    x_out[7:9] = matrix_to_euler(quat_to_matrix(q_out))

    return x_out, q_out / norm(q_out)
end

"""
    compensate_internal_states!(x, dx, quat)

Apply corrections `dx` (negative of the smoothing error) to the state vectors
`x` (9xN) and quaternions `quat` (4xN) in‑place.
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
- `quat`: quaternion `[qw, qx, qy, qz]` representing the initial attitude.
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