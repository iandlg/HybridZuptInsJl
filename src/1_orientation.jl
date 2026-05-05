"""
	matrix_to_euler(R)

Convert rotation matrix or stack of rotation matrices to Euler angles.

Convention:
	roll  = atan(R[3,2], R[3,3])
	pitch = -asin(R[3,1])
	yaw   = atan(R[2,1], R[1,1])

# Arguments
- `R`: either (3, 3) for a single matrix, or (3, 3, N) for a stack

# Returns
- (3,) vector for a single matrix, (3, N) matrix for a stack  →  [roll, pitch, yaw]
"""
function matrix_to_euler(R::AbstractArray{<:Real})
    if ndims(R) == 2
        # single matrix — return a plain 3-vector
        return _matrix_to_euler_single(R)
    end

    # stack (3, 3, N)
    N = size(R, 3)
    out = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        out[:, i] = _matrix_to_euler_single(view(R, :, :, i))
    end
    return out
end

"""
    _matrix_to_euler_single(R::AbstractMatrix{<:Real})

Convert a single 3×3 rotation matrix to Euler angles.

This is an internal helper function used by `matrix_to_euler`. It extracts the roll, pitch,
and yaw angles from a rotation matrix using the convention:
- roll  = atan(R[3,2], R[3,3])
- pitch = -asin(R[3,1])
- yaw   = atan(R[2,1], R[1,1])

Note: The pitch computation clamps the input to [-1, 1] to guard against floating-point errors
that could cause `asin` to fail.

# Arguments
- `R`: A 3×3 rotation matrix

# Returns
- A 3-element vector [roll, pitch, yaw]
"""
function _matrix_to_euler_single(R::AbstractMatrix{<:Real})
    roll = atan(R[3, 2], R[3, 3])
    pitch = -asin(clamp(R[3, 1], -1.0, 1.0))   # clamp guards against float noise
    yaw = atan(R[2, 1], R[1, 1])
    return [roll, pitch, yaw]
end
