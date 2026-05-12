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
    dim_x = length(state)
    dim_z = length(measurement)
    @assert size(stateCov) == (dim_x, dim_x)
    @assert size(H) == (dim_z, dim_x)
    @assert size(R) == (dim_z, dim_z)

    # Innovation
    innovation = measurement - H * state          # (dim_z,)

    # Innovation covariance
    S = H * stateCov * H' + R                    # (dim_z, dim_z)

    # Kalman gain
    K = stateCov * H' * inv(S)                   # (dim_x, dim_z)

    # Updated state
    updated_state = state + K * innovation        # (dim_x,)

    # Updated covariance (Joseph form is stable, but here simple form)
    I = Matrix{T}(I, dim_x, dim_x)
    updated_cov = (I - K * H) * stateCov          # (dim_x, dim_x)

    return updated_state, updated_cov
end


"""
    dynamic_update(state::AbstractVector{T}, stateCov::AbstractMatrix{T},
                   F::AbstractMatrix{T}, Q::AbstractMatrix{T}) where T<:Real

Kalman prediction step for a single state.

# Arguments
- `state`: State vector, shape `(dim_x,)`.
- `stateCov`: State covariance matrix, shape `(dim_x, dim_x)`.
- `F`: State transition matrix, shape `(dim_x, dim_x)`.
- `Q`: Process noise covariance, shape `(dim_x, dim_x)`.

# Returns
- `updated_state`: Predicted state vector, shape `(dim_x,)`.
- `updated_cov`: Predicted covariance matrix, shape `(dim_x, dim_x)`.
"""
function dynamic_update(state::AbstractVector{T}, stateCov::AbstractMatrix{T},
    F::AbstractMatrix{T}, Q::AbstractMatrix{T}) where T<:Real
    dim = length(state)
    @assert size(stateCov) == (dim, dim)
    @assert size(F) == (dim, dim)
    @assert size(Q) == (dim, dim)

    # Predict state: x̂ = F * x
    updated_state = F * state

    # Predict covariance: P̂ = F * P * F' + Q
    updated_cov = F * stateCov * F' + Q

    return updated_state, updated_cov
end