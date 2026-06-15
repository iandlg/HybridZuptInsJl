function stride_heading(; R_wb::AbstractMatrix{T}, Δp_w::AbstractVector{T})::AbstractVector{T} where T<:Real
    euler = zeros(Float64, 3)
    euler[3] = matrix_to_euler(R_wb')[3]
    return euler_to_matrix(euler) * Δp_w
end

function stride_body(; R_wb::AbstractMatrix{T}, Δp_w::AbstractVector{T})::AbstractVector{T} where T<:Real
    return R_wb' * Δp_w
end

function stride_local(ref_frame::ReferenceFrame; R_wb::AbstractMatrix{T}, Δp_w::AbstractVector{T}) where T<:Real
    return Dict{ReferenceFrame,AbstractVector{T}}(
        HEADING => stride_heading(R_wb=R_wb, Δp_w=Δp_w),
        BODY => stride_body(R_wb=R_wb, Δp_w=Δp_w),
    )[ref_frame]
end

abstract type AbstractCorrector end

function initialize_corrector!(
    c::AbstractCorrector;
    t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwarg...)
    error("initialize_corrector! not implemented for $(typeof(c))")
end

function dynamic_update!(c::AbstractCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwarg...)
    error("dynamic_update! not implemented for $(typeof(c))")
end

function stride_measurement_update!(c::AbstractCorrector; ref_frame::ReferenceFrame, feature_type::FeatureType, stride_aug::AbstractVector{Float64}, Σstride::AbstractMatrix{Float64}, kwargs...)
    error("stride_measurement_update! not implemented for $(typeof(c))")
end

function posyaw_measurement_update!(c::AbstractCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    error("posyaw_measurement_update! not implemented for $(typeof(c))")
end

function learned_measurement_update!(c::AbstractCorrector; kwargs...)
    error("learned_measurement_update! not implemented for $(typeof(c))")
end

function relinearize!(c::AbstractCorrector; kwarg...)
    error("relinearize! not implemented for $(typeof(c))")
end

# Accessors
function get_time(c::AbstractCorrector)::AbstractVector{Float64}
    return c.t[1:c.i]
end

function get_pos(c::AbstractCorrector)::AbstractMatrix{Float64}
    return c.pos[:, 1:c.i]
end

function get_quat(c::AbstractCorrector)::AbstractMatrix{Float64}
    return c.quat[:, 1:c.i]
end

function get_trajectory(c::AbstractCorrector)::Trajectory
    Trajectory(
        get_time(c),
        get_pos(c),
        quat_to_matrix(get_quat(c))
    )
end

# ── Concrete correctors ───────────────────────────────────────────────────
mutable struct DefaultCorrector <: AbstractCorrector
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractMatrix{Float64}
    Σ::AbstractArray{Float64,3}
    G::AbstractArray{Float64,3}
    H::AbstractArray{Float64,3}
    i::Int
end

function DefaultCorrector(N::Int)::DefaultCorrector
    @assert N > 1 "Invalid number of "
    return DefaultCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 6, N),
        zeros(Float64, 6, 6, N), repeat(Matrix{Float64}(I, 6, 6), 1, 1, N), zeros(Float64, 4, 6, N), 1)
end

function initialize_corrector!(c::DefaultCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[:, :, 1] = Σpq_init
    c.G[:, :, 1] = Matrix{Float64}(I, 6, 6)
    c.i = 1
end

function dynamic_update!(c::DefaultCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    c.G[1:3, 1:3, c.i] = quat_to_matrix(c.quat[:, c.i-1])
    c.G[4:6, 4:6, c.i] = quat_to_matrix(c.quat[:, c.i])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + c.G[1:3, 1:3, c.i] * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)

    # Covariance update
    c.δx[:, c.i] .= 0.0 # c.F[:, :, c.i] * c.δx[:, c.i-1]
    c.Σ[:, :, c.i] = c.Σ[:, :, c.i-1] + c.G[:, :, c.i] * Σpq * c.G[:, :, c.i]'

end

function stride_measurement_update!(c::DefaultCorrector; ref_frame::ReferenceFrame, stride_aug::AbstractVector{Float64}, Σstride::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = quat_to_matrix(c.quat[:, c.i-1])' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # c.H[4, 4:6, c.i] = jacobian_∂θ3_∂δθ_left(c.quat[:, c.i])
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Δp ins : " (c.pos[:, c.i] - c.pos[:, c.i-1]) maxlog = 10
    # @info "prev rot mat INS : " matrix_to_quat(c.H[1:3, 1:3, c.i]) maxlog = 10


    stride_aug[1:3] -= stride_local(ref_frame; R_wb=c.H[1:3, 1:3, c.i]', Δp_w=c.pos[:, c.i] - c.pos[:, c.i-1])
    Δθ3 = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3] - matrix_to_euler(quat_to_matrix(c.quat[:, c.i-1]))[3]
    Δθ3 = atan(sin(Δθ3), cos(Δθ3))

    augstrid_ins = [stride_local(ref_frame; R_wb=c.H[1:3, 1:3, c.i]', Δp_w=c.pos[:, c.i] - c.pos[:, c.i-1]); Δθ3]
    @info "INS aug stride : " augstrid_ins maxlog = 10

    stride_aug[4] = atan(sin(stride_aug[4] - Δθ3), cos(stride_aug[4] - Δθ3))

    @info "Augmented stride difference : " stride_aug maxlog = 10

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        stride_aug,
        c.H[:, :, c.i],
        Σstride
    )
end

function posyaw_measurement_update!(c::DefaultCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::DefaultCorrector; kwargs...)
    # Do nothing
end

function relinearize!(c::DefaultCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.δx[:, c.i] .= 0.0
end

mutable struct StaticCorrector <: AbstractCorrector
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractMatrix{Float64}
    Σ::AbstractArray{Float64,3}
    G::AbstractArray{Float64,3}
    H::AbstractArray{Float64,3}
    i::Int

    # Corrector specific Arguments
    stride_err_mean::AbstractMatrix{Float64}
end

function StaticCorrector(N::Int)::StaticCorrector
    @assert N > 1 "Invalid number of "
    return StaticCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 10, N),
        zeros(Float64, 10, 10, N), repeat(Matrix{Float64}(I, 10, 10), 1, 1, N), zeros(Float64, 4, 10, N), 1,
        zeros(Float64, 4, N)
    )
end

function initialize_corrector!(c::StaticCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6, 1] = Σpq_init
    c.Σ[7:10, 7:10, 1] = Matrix{Float64}(I, 4, 4) * 1e-1
    c.G[:, :, 1] = Matrix{Float64}(I, 10, 10)
    c.i = 1
end

function dynamic_update!(c::StaticCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    c.G[1:3, 1:3, c.i] = quat_to_matrix(c.quat[:, c.i-1])
    c.G[4:6, 4:6, c.i] = quat_to_matrix(c.quat[:, c.i])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + c.G[1:3, 1:3, c.i] * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    c.stride_err_mean[:, c.i] = c.stride_err_mean[:, c.i-1]

    temp = zeros(Float64, 10, 10)


    # Covariance update
    c.δx[:, c.i] .= 0.0 # c.F[:, :, c.i] * c.δx[:, c.i-1]
    c.Σ[1:6, 1:6, c.i] = c.Σ[1:6, 1:6, c.i-1] + c.G[1:6, 1:6, c.i] * Σpq * c.G[1:6, 1:6, c.i]'
    c.Σ[7:10, 7:10, c.i] = c.Σ[7:10, 7:10, c.i-1]
    c.Σ[7:10, 1:6, c.i] = c.Σ[7:10, 1:6, c.i-1]
    c.Σ[1:6, 7:10, c.i] = c.Σ[1:6, 7:10, c.i-1]
end

function stride_measurement_update!(c::StaticCorrector; ref_frame::ReferenceFrame, stride_aug::AbstractVector{Float64}, Σstride::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = quat_to_matrix(c.quat[:, c.i-1])' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # c.H[4, 4:6, c.i] = jacobian_∂θ3_∂δθ_left(c.quat[:, c.i])
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    c.H[1:4, 7:10, c.i] = Matrix{Float64}(I, 4, 4)

    stride_aug[1:3] -= (
        stride_local(ref_frame; R_wb=c.H[1:3, 1:3, c.i]', Δp_w=c.pos[:, c.i] - c.pos[:, c.i-1])
        +
        c.stride_err_mean[1:3, c.i]
    )

    Δθ3 = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3] - matrix_to_euler(quat_to_matrix(c.quat[:, c.i-1]))[3] + c.stride_err_mean[4, c.i]
    Δθ3 = atan(sin(Δθ3), cos(Δθ3))
    stride_aug[4] = atan(sin(stride_aug[4] - Δθ3), cos(stride_aug[4] - Δθ3))

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        stride_aug,
        c.H[:, :, c.i],
        Σstride
    )
end


function posyaw_measurement_update!(c::StaticCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    c.H[1:4, 7:10, c.i] .= 0.0
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::StaticCorrector; kwargs...)
    c.H[1:3, 1:3, c.i] = quat_to_matrix(c.quat[:, c.i-1])' # H[1:3, 1:3] = R_bw = R_wb^⊤
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    c.H[1:4, 7:10, c.i] .= 0.0

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        c.stride_err_mean[:, c.i],
        c.H[:, :, c.i],
        c.Σ[7:10, 7:10, c.i] .* 1e-2
    )
end

function relinearize!(c::StaticCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.stride_err_mean[:, c.i] += c.δx[7:10, c.i]
    c.δx[:, c.i] .= 0.0
end