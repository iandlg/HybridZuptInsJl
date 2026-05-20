abstract type AbstractNominalCorrector end

function update_corrector!(c::AbstractNominalCorrector, input, y)
    error("update_corrector! not implemented for $(typeof(c))")
end

function predict_correction(c::AbstractNominalCorrector, input)
    error("predict_correction not implemented for $(typeof(c))")
end


# ── Concrete HSGP corrector ───────────────────────────────────────────────────
mutable struct HsgpCorrector <: AbstractNominalCorrector
    params::HsgpParameters
    eigvals::Matrix{Float64}
    beta::Dict{String,Vector{Float64}}
    P_beta::Dict{String,Matrix{Float64}}
    outputs_keys::Vector{String}
end

function HsgpCorrector(params::HsgpParameters)::HsgpCorrector
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

"""
    ExactGpCorrector(params::HsgpParameters)

Mutable corrector that stores all training inputs and outputs, then on prediction
fits an exact GP for each of the 4 outputs (yaw, x, y, z) using the hyperparameters
and normalization statistics from `params`. The GP is built from scratch on every
prediction using all stored data.

# Fields
- `params`: `HsgpParameters` containing hyperparameters (`SeHyperparams`),
  input dimension, number of basis functions (unused here), normalization statistics
  for inputs (`input_stats`), yaw (`yaw_stats`), and positions (`pos_stats`), and
  domain bounds `LL` (unused).
- `X_train`: Vector of stored input vectors (each length `d`).
- `Y_train`: Vector of stored output 4‑vectors (yaw, x, y, z).
"""
mutable struct ExactGpCorrector <: AbstractNominalCorrector
    params::HsgpParameters
    X_train::Vector{Vector{Float64}}
    Y_train::Vector{Vector{Float64}}
end

function ExactGpCorrector(params::HsgpParameters)
    return ExactGpCorrector(params, Vector{Vector{Float64}}[], Vector{Vector{Float64}}[])
end

"""
    update_corrector!(c::ExactGpCorrector, input::Vector{Float64}, y::Vector{Float64})

Add a new training pair (input, output) to the corrector. Normalization is not
applied here; it will be done lazily during prediction using the pre‑computed
statistics in `c.params`.
"""
function update_corrector!(c::ExactGpCorrector, input::Vector{Float64}, y::Vector{Float64})
    push!(c.X_train, input)
    push!(c.Y_train, y)
    return nothing
end

"""
    predict_correction(c::ExactGpCorrector, input::Vector{Float64}) -> Vector{Float64}

Predict the correction (yaw, x, y, z) for the given input. If no training data
exists, returns zero correction. Otherwise:
- Normalizes the test input using `c.params.input_stats`.
- Normalizes all stored training inputs and outputs using the same statistics.
- For each of the 4 outputs, builds a GP with fixed hyperparameters (`σ_f`, `ℓ`, `σ_n`)
  from `c.params.hp` and computes the posterior mean at the test point.
- Denormalises the predictions using `c.params.yaw_stats` and `c.params.pos_stats`.
"""
function predict_correction(c::ExactGpCorrector, input::Vector{Float64})
    if isempty(c.X_train)
        @warn "No training data for ExactGpCorrector, returning zero correction"
        return zeros(4)
    end

    # Unpack statistics
    input_mean, input_std = c.params.input_stats[1], c.params.input_stats[2]
    yaw_mean, yaw_std = c.params.yaw_stats[1], c.params.yaw_stats[2]
    pos_mean, pos_std = c.params.pos_stats[1], c.params.pos_stats[2]
    d = c.params.d

    # Normalise test input
    x_norm = reshape((input .- input_mean) ./ input_std, (d, 1))

    # Normalise stored training inputs
    n_train = length(c.X_train)
    X_norm = hcat(c.X_train...)
    X_norm = (X_norm .- input_mean) ./ input_std

    # Normalise stored outputs
    Y_norm = hcat(c.Y_train...)
    Y_norm[1, :] = (Y_norm[1, :] .- yaw_mean) ./ yaw_std
    Y_norm[2:4, :] = (Y_norm[2:4, :] .- pos_mean) ./ pos_std

    # Predict each output
    preds = Vector{Float64}(undef, 4)
    output_keys = ["yaw", "pos_1", "pos_2", "pos_3"]

    for (idx, key) in enumerate(output_keys)
        hp_vec = getfield(c.params.hp, Symbol(key))
        σ_f, ℓ, σ_n = hp_vec

        kernel = GaussianProcesses.SE(log(ℓ), log(σ_f))
        y_out = Y_norm[idx, :]

        # Build GP (training inputs: X_norm', size (n_train, d))
        gp = GaussianProcesses.GP(X_norm, y_out, GaussianProcesses.MeanZero(), kernel, log(σ_n))

        μ, _ = GaussianProcesses.predict_y(gp, x_norm)
        preds[idx] = μ[1]
    end

    # Denormalise predictions
    preds[1] = preds[1] * yaw_std + yaw_mean
    preds[2:4] = preds[2:4] .* pos_std .+ pos_mean
    return preds
end