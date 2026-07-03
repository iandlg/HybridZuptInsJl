"""
    matrix_to_euler(R)

Convert a rotation matrix or a stack of rotation matrices to Euler angles using a loop.

# Arguments
- `R`: AbstractArray of size `(3,3)` or `(3,3,N)`

# Returns
- If `R` is `(3,3)`: 3-element vector `[roll, pitch, yaw]`
- If `R` is `(3,3,N)`: `(3,N)` matrix where each column is `[roll, pitch, yaw]` for one matrix
"""
function matrix_to_euler(R::AbstractArray{T,3}) where T
    size(R, 1) == 3 && size(R, 2) == 3 || throw(DimensionMismatch("First two dimensions must be 3"))
    N = size(R, 3)

    # Pre-allocate result matrix: 3 rows (roll, pitch, yaw), N columns
    result = Matrix{Float64}(undef, 3, N)

    # Loop over each rotation matrix in the stack
    for k in 1:N
        # Extract the 3x3 matrix at index k
        # Using views or direct indexing; here direct access for clarity
        R11 = R[1, 1, k]
        R21 = R[2, 1, k]
        R31 = R[3, 1, k]
        R32 = R[3, 2, k]
        R33 = R[3, 3, k]

        # Compute Euler angles
        roll = atan(R32, R33)
        pitch = -asin(clamp(R31, -1.0, 1.0))
        yaw = atan(R21, R11)

        # Store in result
        result[1, k] = roll
        result[2, k] = pitch
        result[3, k] = yaw
    end

    return result
end

function matrix_to_euler(R::AbstractArray{T,2}) where T
    size(R) == (3, 3) || throw(DimensionMismatch("Expected a 3x3 matrix"))
    # Convert to 3x3x1 and call the 3D version, then squeeze the column
    result_3d = matrix_to_euler(reshape(R, 3, 3, 1))
    return result_3d[:, 1]   # returns a 3-element vector
end


"""
    euler_to_matrix(ang)

Compute rotation matrices from Euler angles [roll, pitch, yaw] (in radians).

# Arguments
- `ang`: AbstractVector of length 3, or AbstractMatrix of size (3, N)
         Each column contains [roll; pitch; yaw].

# Returns
- If `ang` is a vector: a 3x3 rotation matrix.
- If `ang` is a matrix: a 3x3xN array, where `R[:,:,k]` is the k-th rotation matrix.
"""
function euler_to_matrix(ang::AbstractMatrix{T}) where T<:AbstractFloat
    size(ang, 1) == 3 || throw(DimensionMismatch("Expected 3 rows, got $(size(ang,1))"))
    N = size(ang, 2)

    # Pre-allocate result: (3,3,N) with the same element type
    R = Array{T,3}(undef, 3, 3, N)

    for k in 1:N
        r, p, y = ang[1, k], ang[2, k], ang[3, k]
        cr, sr = cos(r), sin(r)
        cp, sp = cos(p), sin(p)
        cy, sy = cos(y), sin(y)

        # Build intermediate matrix M (the one before transposition)
        M = zeros(T, 3, 3)
        M[1, 1] = cy * cp
        M[1, 2] = sy * cp
        M[1, 3] = -sp
        M[2, 1] = -sy * cr + cy * sp * sr
        M[2, 2] = cy * cr + sy * sp * sr
        M[2, 3] = cp * sr
        M[3, 1] = sy * sr + cy * sp * cr
        M[3, 2] = -cy * sr + sy * sp * cr
        M[3, 3] = cp * cr

        # Transpose to match Python's output (swaps axes 0 and 1)
        R[:, :, k] = M'
    end
    return R
end

# Single vector case
function euler_to_matrix(ang::AbstractVector{T}) where T<:AbstractFloat
    length(ang) == 3 || throw(DimensionMismatch("Expected 3 elements, got $(length(ang))"))
    # Convert to 3x1 matrix, call the matrix version, then drop the third dimension
    Mstack = euler_to_matrix(reshape(ang, 3, 1))
    return Mstack[:, :, 1]   # return a 3x3 matrix
end

# Convenience for integer input (promote to Float64)
euler_to_matrix(ang::AbstractVector{<:Integer}) = euler_to_matrix(float.(ang))
euler_to_matrix(ang::AbstractMatrix{<:Integer}) = euler_to_matrix(float.(ang))


"""
    quat_to_matrix(q)

Convert quaternion(s) to Direction Cosine Matrix (rotation matrix).

# Arguments
- `q`: AbstractVector of length 4 (single quaternion), or AbstractMatrix of size (4, N)
       Quaternion convention: `[qw, qx, qy, qz]` (scalar first).

# Returns
- If `q` is a vector: a 3x3 rotation matrix.
- If `q` is a matrix: a 3x3xN array, where `R[:,:,k]` is the k-th rotation matrix.
"""
function quat_to_matrix(q::AbstractMatrix{T}) where T<:Real
    size(q, 1) == 4 || throw(DimensionMismatch("Expected 4 rows, got $(size(q,1))"))
    N = size(q, 2)
    R = Array{T,3}(undef, 3, 3, N)

    for k in 1:N
        q0, q1, q2, q3 = q[1, k], q[2, k], q[3, k], q[4, k]

        p1 = q1^2
        p2 = q2^2
        p3 = q3^2
        p4 = q0^2
        denom = p1 + p2 + p3 + p4
        p6 = denom == 0 ? 0.0 : 2.0 / denom

        # Diagonal elements
        R[1, 1, k] = 1 - p6 * (p2 + p3)
        R[2, 2, k] = 1 - p6 * (p1 + p3)
        R[3, 3, k] = 1 - p6 * (p1 + p2)

        # Off-diagonal elements
        a1 = p6 * q1
        a2 = p6 * q2
        t = p6 * q3 * q0
        u = a1 * q2
        R[1, 2, k] = u - t
        R[2, 1, k] = u + t

        t = a2 * q0
        u = a1 * q3
        R[1, 3, k] = u + t
        R[3, 1, k] = u - t

        t = a1 * q0
        u = a2 * q3
        R[2, 3, k] = u - t
        R[3, 2, k] = u + t
    end

    return R
end

# Single quaternion (vector) case
function quat_to_matrix(q::AbstractVector{T}) where T<:Real
    length(q) == 4 || throw(DimensionMismatch("Expected 4 elements, got $(length(q))"))
    Rstack = quat_to_matrix(reshape(q, 4, 1))
    return Rstack[:, :, 1]
end

# Convenience for integer input
quat_to_matrix(q::AbstractVector{<:Integer}) = quat_to_matrix(float.(q))
quat_to_matrix(q::AbstractMatrix{<:Integer}) = quat_to_matrix(float.(q))

# ----------------------------------------------------------------------

"""
    matrix_to_quat(R)

Convert Direction Cosine Matrix (rotation matrix) to quaternion.

# Arguments
- `R`: 3x3 matrix (single), or 3x3xN array (batch)
       Rotation matrix/matrices.

# Returns
- If `R` is 3x3: a 4-element vector `[qw, qx, qy, qz]` (scalar first).
- If `R` is 3x3xN: a 4xN matrix, each column a quaternion `[qw, qx, qy, qz]`.
"""
function matrix_to_quat(R::AbstractArray{T,3}) where T<:Real
    size(R, 1) == 3 && size(R, 2) == 3 || throw(DimensionMismatch("First two dimensions must be 3"))
    N = size(R, 3)
    q = Array{T,2}(undef, 4, N)

    for k in 1:N
        # Extract the 3x3 matrix
        R11 = R[1, 1, k]
        R12 = R[1, 2, k]
        R13 = R[1, 3, k]
        R21 = R[2, 1, k]
        R22 = R[2, 2, k]
        R23 = R[2, 3, k]
        R31 = R[3, 1, k]
        R32 = R[3, 2, k]
        R33 = R[3, 3, k]

        T_trace = 1 + R11 + R22 + R33

        if T_trace > 1e-8
            S = 0.5 / sqrt(T_trace)
            q[1, k] = 0.25 / S            # qw (scalar)
            q[2, k] = (R32 - R23) * S      # qx
            q[3, k] = (R13 - R31) * S      # qy
            q[4, k] = (R21 - R12) * S      # qz
        else
            # Find the largest diagonal element
            if R11 > R22 && R11 > R33
                S = sqrt(1 + R11 - R22 - R33) * 2
                q[1, k] = (R32 - R23) / S
                q[2, k] = 0.25 * S
                q[3, k] = (R12 + R21) / S
                q[4, k] = (R13 + R31) / S
            elseif R22 > R33
                S = sqrt(1 + R22 - R11 - R33) * 2
                q[1, k] = (R13 - R31) / S
                q[2, k] = (R12 + R21) / S
                q[3, k] = 0.25 * S
                q[4, k] = (R23 + R32) / S
            else
                S = sqrt(1 + R33 - R11 - R22) * 2
                q[1, k] = (R21 - R12) / S
                q[2, k] = (R13 + R31) / S
                q[3, k] = (R23 + R32) / S
                q[4, k] = 0.25 * S
            end
        end
    end

    return q
end

# Single matrix case
function matrix_to_quat(R::AbstractMatrix{T}) where T<:Real
    size(R) == (3, 3) || throw(DimensionMismatch("Expected 3x3 matrix"))
    qstack = matrix_to_quat(reshape(R, 3, 3, 1))
    return qstack[:, 1]
end

# Convenience for integer input
matrix_to_quat(R::AbstractMatrix{<:Integer}) = matrix_to_quat(float.(R))
matrix_to_quat(R::AbstractArray{<:Integer,3}) = matrix_to_quat(float.(R))

# Integer input
matrix_to_quat(R::AbstractMatrix{<:Integer}) = matrix_to_quat(float.(R))
matrix_to_quat(R::AbstractArray{<:Integer,3}) = matrix_to_quat(float.(R))


"""
    normalize_quat(q)

Normalise quaternion(s) to unit length.

# Arguments
- `q`: AbstractVector of length 4 (single quaternion), or AbstractMatrix of size (4, N)

# Returns
- Normalised quaternion(s) with same shape.
- If a quaternion is zero (norm ≈ 0), returns the same zero quaternion to avoid division by zero.
"""
function normalize_quat(q::AbstractMatrix{T}) where T<:Real
    size(q, 1) == 4 || throw(DimensionMismatch("Expected 4 rows, got $(size(q,1))"))
    N = size(q, 2)
    q_norm = similar(q)
    for k in 1:N
        n = sqrt(q[1, k]^2 + q[2, k]^2 + q[3, k]^2 + q[4, k]^2)
        if n > eps(T)   # avoid division by ~zero
            q_norm[:, k] = q[:, k] / n
        else
            q_norm[:, k] = q[:, k]   # already zero
        end
    end
    return q_norm
end

# Single quaternion (vector) case
function normalize_quat(q::AbstractVector{T}) where T<:Real
    length(q) == 4 || throw(DimensionMismatch("Expected 4 elements, got $(length(q))"))
    n = sqrt(q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)
    if n > eps(T)
        return q / n
    else
        return copy(q)
    end
end

# Convenience for integers
normalize_quat(q::AbstractVector{<:Integer}) = normalize_quat(float.(q))
normalize_quat(q::AbstractMatrix{<:Integer}) = normalize_quat(float.(q))


function quat_multiply(p::AbstractVector{T}, q::AbstractVector{T})::AbstractVector{T} where T<:Real
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return [
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw
    ]
end


"""
    skew(v)

Return the 3x3 skew-symmetric matrix of a 3-element vector `v`.

# Arguments
- `v`: AbstractVector of length 3.

# Returns
- A 3x3 matrix S such that S * w = cross(v, w) for any vector w.
"""
function skew(v::AbstractVector{T}) where T<:Real
    length(v) == 3 || throw(DimensionMismatch("Expected vector of length 3, got $(length(v))"))
    # Unpack components
    v1, v2, v3 = v[1], v[2], v[3]
    return T[0 -v3 v2;
        v3 0 -v1;
        -v2 v1 0]
end

"""
    skew(v)

Compute skew-symmetric matrices for a batch of 3-element vectors.

# Arguments
- `v`: AbstractMatrix of size (3, N) – each column is a vector.

# Returns
- A 3x3xN array S_batch where S_batch[:,:,k] = skew(V[:,k]).
"""
function skew(v::AbstractMatrix{T}) where T<:Real
    size(v, 1) == 3 || throw(DimensionMismatch("Expected 3 rows, got $(size(v,1))"))
    N = size(v, 2)
    S_batch = Array{T,3}(undef, 3, 3, N)
    for k in 1:N
        v1, v2, v3 = v[1, k], v[2, k], v[3, k]
        S_batch[:, :, k] = T[0 -v3 v2;
            v3 0 -v1;
            -v2 v1 0]
    end
    return S_batch
end


function quat_conjugate(q::AbstractVector{T}) where T<:Real
    length(q) == 4 || throw(DimensionMismatch("Expected vector of length 4, got $(length(q))"))
    qw, qx, qy, qz = q
    return [qw, -qx, -qy, -qz]
end

function quat_conjugate(q::AbstractMatrix{T}) where T<:Real
    size(q, 1) == 4 || throw(DimensionMismatch("Expected 4 rows, got $(size(q,1))"))
    q_out = similar(q)
    q_out[1, :] .= q[1, :]   # scalar part stays unchanged
    q_out[2, :] .= -q[2, :]   # negate x
    q_out[3, :] .= -q[3, :]   # negate y
    q_out[4, :] .= -q[4, :]   # negate z
    return q_out
end

function quat_exp(v::Vector{T}) where T<:Real
    length(v) == 3 || throw(DimensionMismatch("Expected vector of length 3, got $(length(v))"))
    ϕ = LinearAlgebra.norm(v)
    ϕ < 1e-9 && return [1.0, 0.0, 0.0, 0.0]
    return vcat(cos(ϕ / 2), v ./ ϕ .* sin(ϕ / 2))
end
