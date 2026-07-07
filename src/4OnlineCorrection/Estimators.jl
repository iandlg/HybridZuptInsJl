# ── AbstractEstimator.jl ────────────────────────────────────────────────────
abstract type AbstractEstimator end

function initialize_estimator!(c::AbstractEstimator;
    t::Float64, x_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64},
    P_init::AbstractMatrix{Float64}, kwargs...)
    error("initialize_estimator! not implemented for $(typeof(c))")
end

function propagate_nominal!(c::AbstractEstimator, t::Float64, u::AbstractVector{Float64},
    Ts::Float64, g_vec::AbstractVector{Float64})
    error("propagate_nominal! not implemented for $(typeof(c))")
end

function propagate_covariance!(c::AbstractEstimator, Ts::Float64, Q::AbstractMatrix{Float64})
    error("propagate_covariance! not implemented for $(typeof(c))")
end

function time_update!(c::AbstractEstimator; t::Float64, u::AbstractVector{Float64},
    Ts::Float64, g_vec::AbstractVector{Float64}, Q::AbstractMatrix{Float64}, kwargs...)
    propagate_nominal!(c, t, u, Ts, g_vec)
    propagate_covariance!(c, Ts, Q)
    c.i += 1
end

function zupt_measurement_update!(c::AbstractEstimator; R_meas::AbstractMatrix{Float64}, kwargs...)
    error("zupt_measurement_update! not implemented for $(typeof(c))")
end

function smooth_segment!(c::AbstractEstimator; seg_start::Int, seg_end::Int, kwargs...)
    error("smooth_segment! not implemented for $(typeof(c))")
end

function inject_nominal_state!(c::AbstractEstimator)
    error("inject_nominal_state! not implemented for $(typeof(c))")
end

function reset_error_state!(c::AbstractEstimator)
    error("reset_error_state! not implemented for $(typeof(c))")
end

relinearize_extra_state!(::AbstractEstimator) = nothing

function relinearize!(c::AbstractEstimator)
    inject_nominal_state!(c)
    reset_error_state!(c)
    relinearize_extra_state!(c)
end

function stride_measurement_update!(c::AbstractEstimator;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64},
    R_aug_wl::AbstractMatrix{Float64}, kwargs...)
    error("stride_measurement_update! not implemented for $(typeof(c))")
end

function posyaw_measurement_update!(c::AbstractEstimator;
    curr_pos::AbstractVector{Float64}, curr_θ3::Float64,
    Σy::AbstractMatrix{Float64}, kwargs...)
    error("posyaw_measurement_update! not implemented for $(typeof(c))")
end

function learned_measurement_update!(c::AbstractEstimator;
    kwargs...)::NTuple{2,Optional{AbstractVector{Float64}}}
    error("learned_measurement_update! not implemented for $(typeof(c))")
end

# ── Accessors ────────────────────────────────────────────────────────────
get_time(c::AbstractEstimator) = c.t[1:c.i]
get_pos(c::AbstractEstimator) = c.pos[:, 1:c.i]
get_pos(c::AbstractEstimator, k::Int) = c.pos[:, k]
get_vel(c::AbstractEstimator) = c.vel[:, 1:c.i]
get_vel(c::AbstractEstimator, k::Int) = c.vel[:, k]
get_quat(c::AbstractEstimator) = c.quat[:, 1:c.i]
get_quat(c::AbstractEstimator, k::Int) = c.quat[:, k]
get_quat_matrix(c::AbstractEstimator, k::Int) = quat_to_matrix(c.quat[:, k])

function get_posatt_cov(c::AbstractEstimator)
    idx = [1, 2, 3, 9]
    return c.Σ[idx, idx]
end

function get_trajectory(c::AbstractEstimator)::Trajectory
    Trajectory(get_time(c), get_pos(c), quat_to_matrix(get_quat(c)), get_vel(c))
end

# ── BasicEstimator.jl ─────────────────────────────────────────────────────
mutable struct BasicEstimator <: AbstractEstimator
    t::Vector{Float64}
    pos::Matrix{Float64}
    vel::Matrix{Float64}
    quat::Matrix{Float64}
    δx::Matrix{Float64}
    Σ::Matrix{Float64}
    F_store::Array{Float64,3}
    P_store::Array{Float64,3}
    P_timeupd::Array{Float64,3}
    i::Int
end

function BasicEstimator(N::Int)::BasicEstimator
    @assert N > 1 "Invalid number of samples"
    return BasicEstimator(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 3, N), zeros(Float64, 4, N),
        zeros(Float64, 9, N), zeros(Float64, 9, 9),
        zeros(Float64, 9, 9, N), zeros(Float64, 9, 9, N), zeros(Float64, 9, 9, N), 1
    )
end

function initialize_estimator!(c::BasicEstimator;
    t::Float64, x_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64},
    P_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = x_init[1:3]
    c.vel[:, 1] = x_init[4:6]
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ .= P_init
    c.P_store[:, :, 1] = P_init
    c.i = 1
end

function propagate_nominal!(c::BasicEstimator, t::Float64, u::AbstractVector{Float64},
    Ts::Float64, g_vec::AbstractVector{Float64})
    n = c.i + 1
    x_prev = vcat(c.pos[:, c.i], c.vel[:, c.i], matrix_to_euler(quat_to_matrix(c.quat[:, c.i])))
    y, q_new = navigation_equations(x_prev, u, c.quat[:, c.i], Ts, g_vec)
    c.t[n] = t
    c.pos[:, n] = y[1:3]
    c.vel[:, n] = y[4:6]
    c.quat[:, n] = q_new
end

function propagate_covariance!(c::BasicEstimator, Ts::Float64, Q::AbstractMatrix{Float64})
    n = c.i + 1
    F, G = state_matrix(c.quat[:, n], zeros(6), Ts)  # placeholder overwritten below if u needed
    c.δx[:, n] = F * c.δx[:, c.i]
    c.Σ .= F * c.Σ * F' + G * Q * G'
    c.Σ .= (c.Σ + c.Σ') / 2
    c.F_store[:, :, n] = F
    c.P_store[:, :, n] = c.Σ
    c.P_timeupd[:, :, n] = c.Σ
end

function zupt_measurement_update!(c::BasicEstimator; R_meas::AbstractMatrix{Float64}, kwargs...)
    n = c.i
    H = zeros(3, 9)
    H[1:3, 4:6] = Matrix{Float64}(I, 3, 3)
    c.δx[:, n], c.Σ = measurement_update(
        c.δx[:, n], c.Σ, -H * c.δx[:, n] .+ H * c.δx[:, n] .- (c.vel[:, n] .- c.vel[:, n]), H, R_meas
    )
    # δx correction toward zero velocity: measurement = 0 = vel + δv
    c.δx[:, n], c.Σ = measurement_update(
        c.δx[:, n], c.Σ, -c.vel[:, n], H, R_meas
    )
    c.Σ .= (c.Σ + c.Σ') / 2
    c.P_store[:, :, n] = c.Σ
end

function smooth_segment!(c::BasicEstimator; seg_start::Int, seg_end::Int, kwargs...)
    δx_smooth = zeros(Float64, 9, seg_end)
    P_smooth = zeros(Float64, 9, 9, seg_end)
    δx_smooth[:, seg_end] = c.δx[:, seg_end]
    P_smooth[:, :, seg_end] = c.P_store[:, :, seg_end]

    for n in (seg_end-1):-1:seg_start
        A = c.P_store[:, :, n] * c.F_store[:, :, n]' / c.P_timeupd[:, :, n+1]
        δx_smooth[:, n] = c.δx[:, n] + A * (δx_smooth[:, n+1] - c.δx[:, n+1])
        P_smooth[:, :, n] = c.P_store[:, :, n] +
                            A * (P_smooth[:, :, n+1] - c.P_timeupd[:, :, n+1]) * A'
        P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n]') / 2
    end

    for n in seg_start:seg_end
        x_in = vcat(c.pos[:, n], c.vel[:, n], matrix_to_euler(quat_to_matrix(c.quat[:, n])))
        x_out, q_out = comp_internal_states(x_in, -δx_smooth[:, n], c.quat[:, n])
        c.pos[:, n] = x_out[1:3]
        c.vel[:, n] = x_out[4:6]
        c.quat[:, n] = q_out
    end

    c.δx[:, seg_end] .= 0.0
    c.Σ .= c.P_store[:, :, seg_end]
    c.Σ[1:2, 9] .= 0.0
    c.Σ[9, 1:2] .= 0.0
end

function inject_nominal_state!(c::BasicEstimator)
    n = c.i
    x_in = vcat(c.pos[:, n], c.vel[:, n], matrix_to_euler(quat_to_matrix(c.quat[:, n])))
    x_out, q_out = comp_internal_states(x_in, c.δx[:, n], c.quat[:, n])
    c.pos[:, n] = x_out[1:3]
    c.vel[:, n] = x_out[4:6]
    c.quat[:, n] = q_out
end

function reset_error_state!(c::BasicEstimator)
    c.δx[:, c.i] .= 0.0
end

function stride_measurement_update!(c::BasicEstimator;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64},
    R_aug_wl::AbstractMatrix{Float64}, kwargs...)
    return nothing
end

function posyaw_measurement_update!(c::BasicEstimator;
    curr_pos::AbstractVector{Float64}, curr_θ3::Float64,
    Σy::AbstractMatrix{Float64}, kwargs...)
    n = c.i
    H = zeros(4, 9)
    H[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    H[4, 7:9] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, n]))[3]
    meas = vcat(curr_pos .- c.pos[:, n],
        atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim)))
    c.δx[:, n], c.Σ = measurement_update(c.δx[:, n], c.Σ, meas, H, Σy)
    c.Σ .= (c.Σ + c.Σ') / 2
    c.P_store[:, :, n] = c.Σ
end

function learned_measurement_update!(c::BasicEstimator;
    kwargs...)::NTuple{2,Optional{AbstractVector{Float64}}}
    return nothing, nothing
end