abstract type AbstractNominalCorrector end

function update_corrector!(::AbstractNominalCorrector, input, y)
    error("update_corrector! not implemented for $(typeof(corrector))")
end

function predict_correction(::AbstractNominalCorrector, input)
    error("predict_correction not implemented for $(typeof(corrector))")
end


# ── Concrete HSGP corrector ───────────────────────────────────────────────────
mutable struct HsgpCorrector <: AbstractNominalCorrector
    params::HsgpParameters
    eigvals::Matrix{Float64}
    beta::Dict{String,Vector{Float64}}
    P_beta::Dict{String,Matrix{Float64}}
    outputs_keys::Vector{String}
end

function HsgpCorrector(params::HsgpParameters)
    outputs_keys = ["yaw", "pos_1", "pos_2", "pos_3"]
    eigvals = calc_eigenvalues(params.LL, params.m, params.d)
    omega = sqrt.(eigvals)
    psd = Dict(
        k => power_spectral_density(
            omega,
            getfield(params.hp, Symbol(k))[2],
            getfield(params.hp, Symbol(k))[1]
        )
        for k in outputs_keys
    )
    beta = Dict(k => zeros(params.m) for k in outputs_keys)
    P_beta = Dict(k => Matrix(Diagonal(psd[k])) for k in outputs_keys)
    return HsgpCorrector(params, eigvals, beta, P_beta, outputs_keys)
end

function _normalize_input(c::HsgpCorrector, input::Vector{Float64})
    return reshape(
        (input .- c.params.input_stats[1]) ./ c.params.input_stats[2],
        1, c.params.d
    )
end

function _build_eigvect(c::HsgpCorrector, input::Vector{Float64})
    return calc_eigenvectors(_normalize_input(c, input), c.params.LL, c.eigvals)
end

function _normalize_output(c::HsgpCorrector, y::Vector{Float64})
    y_norm = copy(y)
    y_norm[1] = (y[1] - c.params.yaw_stats[1]) / c.params.yaw_stats[2]
    y_norm[2:4] = (y[2:4] .- c.params.pos_stats[1]) ./ c.params.pos_stats[2]
    return y_norm
end

function update_corrector!(
    c::HsgpCorrector,
    input::Vector{Float64},
    y::Vector{Float64}
)
    y_normalized = _normalize_output(c, y)
    eigvect = _build_eigvect(c, input)
    for (idx, outpt) in enumerate(c.outputs_keys)
        sigma_n = getfield(c.params.hp, Symbol(outpt))[3]
        c.beta[outpt], c.P_beta[outpt] = measurement_update(
            c.beta[outpt],
            c.P_beta[outpt],
            [y_normalized[idx]],
            eigvect,
            fill(sigma_n^2, 1, 1)
        )
    end
end

function predict_correction(c::HsgpCorrector, input::Vector{Float64})
    eigvect = _build_eigvect(c, input)
    preds = Vector{Float64}(undef, 4)
    for (idx, outpt) in enumerate(c.outputs_keys)
        preds[idx] = (eigvect*c.beta[outpt])[1]
    end
    # Denormalize
    preds[1] = preds[1] * c.params.yaw_stats[2] + c.params.yaw_stats[1]
    preds[2:4] = preds[2:4] .* c.params.pos_stats[2] .+ c.params.pos_stats[1]
    return preds
end


# ── Concrete static mean corrector ────────────────────────────────────────────
mutable struct StaticMeanCorrector <: AbstractNominalCorrector
    sum::Vector{Float64}   # running sum for each of the 4 outputs (yaw, x, y, z)
    count::Int             # number of updates performed
end

function StaticMeanCorrector()
    return StaticMeanCorrector(zeros(4), 0)
end

function update_corrector!(c::StaticMeanCorrector, input::Vector{Float64}, y::Vector{Float64})
    @assert length(y) == 4 "y must have length 4 (yaw, pos_x, pos_y, pos_z)"
    c.sum .+= y
    c.count += 1
    return nothing
end

function predict_correction(c::StaticMeanCorrector, input::Vector{Float64})
    if c.count == 0
        return zeros(4)   # no data yet, return zero correction
    else
        return c.sum ./ c.count
    end
end