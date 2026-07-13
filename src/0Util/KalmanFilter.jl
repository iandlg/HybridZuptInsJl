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
    # dim_x = length(state)
    # dim_z = length(measurement)
    # @assert size(stateCov) == (dim_x, dim_x)
    # @assert size(H) == (dim_z, dim_x) "Expected $((dim_z, dim_x)) got $(size(H))"
    # @assert size(R) == (dim_z, dim_z) "Expected $((dim_z, dim_z)) got $(size(R))"

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

struct KalmanWorkspace{T<:Real}
    innovation::Vector{T}     # dim_z
    HP::Matrix{T}             # dim_z x dim_x  (H * P)
    S::Matrix{T}              # dim_z x dim_z
    Kt::Matrix{T}             # dim_z x dim_x  (Kᵀ, solved from S * Kᵀ = H*P)
    A::Matrix{T}              # dim_x x dim_x  (I - K*H)
    AP::Matrix{T}             # dim_x x dim_x  (A * P, scratch)
    RKt::Matrix{T}            # dim_z x dim_x  (R * Kᵀ, scratch)
end

function KalmanWorkspace{T}(dim_x::Int, dim_z::Int) where T<:Real
    KalmanWorkspace{T}(
        Vector{T}(undef, dim_z),
        Matrix{T}(undef, dim_z, dim_x),
        Matrix{T}(undef, dim_z, dim_z),
        Matrix{T}(undef, dim_z, dim_x),
        Matrix{T}(undef, dim_x, dim_x),
        Matrix{T}(undef, dim_x, dim_x),
        Matrix{T}(undef, dim_z, dim_x),
    )
end

"""
    measurement_update!(state, stateCov, measurement, H, R, ws)

In-place Kalman measurement update (Joseph form), no heap allocation.
`state` and `stateCov` are updated in place. `ws` is a pre-allocated
`KalmanWorkspace` sized for (dim_x, dim_z) and must be reused across calls.
"""
function measurement_update!(state::AbstractVector{T}, stateCov::AbstractMatrix{T},
    measurement::AbstractVector{T}, H::AbstractMatrix{T},
    R::AbstractMatrix{T}, ws::KalmanWorkspace{T}) where T<:Real

    dim_x = length(state)
    dim_z = length(measurement)
    # @assert size(stateCov) == (dim_x, dim_x)
    # @assert size(H) == (dim_z, dim_x) "Expected $((dim_z, dim_x)) got $(size(H))"
    # @assert size(R) == (dim_z, dim_z) "Expected $((dim_z, dim_z)) got $(size(R))"

    innovation, HP, S, Kt, A, AP, RKt = ws.innovation, ws.HP, ws.S, ws.Kt, ws.A, ws.AP, ws.RKt

    # innovation = measurement - H*state
    mul!(innovation, H, state)
    @. innovation = measurement - innovation

    # HP = H * P
    mul!(HP, H, stateCov)

    # S = H*P*H' + R  (= HP * H' + R)
    mul!(S, HP, H')
    @. S += R

    # Solve S * Kt = HP   (Kt = K')   via Cholesky, in-place, no alloc
    Sfac = cholesky!(Symmetric(S))
    copyto!(Kt, HP)
    ldiv!(Sfac, Kt)          # Kt now holds K' (dim_z x dim_x)

    # state += K*innovation = Kt' * innovation
    mul!(state, Kt', innovation, one(T), one(T))

    # A = I - K*H = I - Kt'*H
    mul!(A, Kt', H)
    @. A = -A
    @inbounds for i in 1:dim_x
        A[i, i] += one(T)
    end

    # stateCov = A*P*A' + K*R*K' = A*P*A' + Kt'*R*Kt
    mul!(AP, A, stateCov)
    mul!(stateCov, AP, A')

    mul!(RKt, R, Kt)
    mul!(stateCov, Kt', RKt, one(T), one(T))

    return state, stateCov
end