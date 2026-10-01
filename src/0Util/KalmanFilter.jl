"""
    measurement_update(state::AbstractVector{T}, stateCov::AbstractMatrix{T},
                       measurement::AbstractVector{T}, H::AbstractMatrix{T},
                       R::AbstractMatrix{T}) where T<:Real

Kalman measurement update for a single state.

# Arguments
- `state`: State vector, shape `(dim_x,)`.
- `stateCov`: State covariance matrix, shape `(dim_x, dim_x)`.
- `measurement`: Measurement vector, shape `(dim_z,)`.
- `H`: Observation matrix, shape `(dim_z, dim_x)`.
- `R`: Measurement noise covariance, shape `(dim_z, dim_z)`.

# Returns
- `updated_state`: Updated state vector, shape `(dim_x,)`.
- `updated_cov`: Updated covariance matrix, shape `(dim_x, dim_x)`.
"""
function measurement_update(state::AbstractVector{T}, stateCov::AbstractMatrix{T},
    measurement::AbstractVector{T}, H::AbstractMatrix{T},
    R::AbstractMatrix{T}) where T<:Real
    # Innovation
    innovation = measurement - H * state          # (dim_z,)

    # Innovation covariance
    S = H * stateCov * H' + R                    # (dim_z, dim_z)

    # Kalman gain
    K = stateCov * H' / S

    # Updated state
    updated_state = state + K * innovation        # (dim_x,)

    # Joseph-form covariance update
    # Idimx = Matrix{Float64}(I, dim_x, dim_x)
    A = I - K * H

    updated_cov = A * stateCov * A' + K * R * K'

    return updated_state, updated_cov
end

