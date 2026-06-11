abstract type AbstractCorrector end

function initialize_corrector!(c::AbstractCorrector; t::Float64, pos_init::Vector{Float64}, quat_init::Vector{Float64}, Σ_init::Matrix{Float64}, kwarg...)
    error("initialize_corrector! not implemented for $(typeof(c))")
end

function dynamic_update!(c::AbstractCorrector; t::Float64, Δp::Vector{Float64}, Δq::Vector{Float64}, Σpq::Matrix{Float64}, kwarg...)
    error("dynamic_update! not implemented for $(typeof(c))")
end

function relinearize!(c::AbstractCorrector; kwarg...)
    error("relinearize! not implemented for $(typeof(c))")
end
# Accessors
function get_time(c::AbstractCorrector)::Vector{Float64}
    return c.t[1:c.i]
end

function get_pos(c::AbstractCorrector)::Matrix{Float64}
    return c.pos[:, 1:c.i]
end

function get_quat(c::AbstractCorrector)::Matrix{Float64}
    return c.quat[:, 1:c.i]
end

function get_trajectory(c::AbstractCorrector)::Trajectory
    Trajectory(
        get_time(c),
        get_pos(c),
        quat_to_matrix(get_quat(c))
    )
end



# function update_corrector!(c::AbstractCorrector, Δp::Vector{Float64}, Δq::Vector{Float64}, Σpq::Matrix{Float64})
#     error("update_corrector! not implemented for $(typeof(c))")
# end

# function predict_correction(c::AbstractCorrector, input)
#     error("predict_correction not implemented for $(typeof(c))")
# end

# ── Concrete correctors ───────────────────────────────────────────────────
mutable struct DefaultCorrector <: AbstractCorrector
    t::Vector{Float64}
    pos::Matrix{Float64}
    quat::Matrix{Float64}
    δx::Matrix{Float64}
    Σ::Array{Float64,3}
    F::Array{Float64,3}
    i::Int
end

function DefaultCorrector(N::Int)::DefaultCorrector
    @assert N > 1 "Invalid number of "
    return DefaultCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 6, N),
        zeros(Float64, 6, 6, N), repeat(Matrix{Float64}(I, 6, 6), 1, 1, N), 1)
end

function initialize_corrector!(c::DefaultCorrector; t::Float64, pos_init::Vector{Float64}, quat_init::Vector{Float64}, Σ_init::Matrix{Float64})
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[:, :, 1] = Σ_init
    c.F[:, :, 1] = Matrix{Float64}(I, 6, 6)
    c.i = 1
end

function dynamic_update!(c::DefaultCorrector; t::Float64, Δp::Vector{Float64}, Δq::Vector{Float64}, Σpq::Matrix{Float64})
    c.i += 1
    c.t[c.i] = t
    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)

    # Covariance update
    c.F[4:6, 4:6, c.i] = quat_to_matrix(Δq)'
    c.δx[:, c.i] .= 0.0 # c.F[:, :, c.i] * c.δx[:, c.i-1]
    c.Σ[:, :, c.i] = c.F[:, :, c.i] * c.Σ[:, :, c.i-1] * c.F[:, :, c.i]' + Σpq
end

function measurement_update!(c::DefaultCorrector;)

end

function relinearize!(c::DefaultCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i], quat_exp(c.δx[4:6, c.i]))
end