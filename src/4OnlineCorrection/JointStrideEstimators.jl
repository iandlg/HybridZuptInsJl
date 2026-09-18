"""
Correctors for `hybrid_zupt_aided_insv4`: the stride model's coefficients `β`
live in the corrector's error state, `δx = [δp^w; δθ^w; δβ]`, and enter the
stride propagation itself (notes/015):

    p_{i+1} = p_i + R_i (Δp^b + R^{bh} S_p y_i + ε_p)
    q_{i+1} = exp(e₃ s_ψ' y_i) ⊗ q_i ⊗ Δq ⊗ exp(ε_q)
    β_{i+1} = β_i,              y_i = y₀ + Φ_i β   (physical units)

so a mocap position/yaw update learns `β` through the cross-covariance that the
propagation builds, and in the test half `β`'s uncertainty accumulates as the
correlated bias it is rather than as white per-stride noise.

The two concrete types differ only in `stride_model` (what `y₀`, `Φ` are) and
the prior on `β`.
"""
abstract type AbstractJointStrideEstimator <: AbstractEstimator end

_channel_mask(corrected_channels::Vector{Symbol})::Vector{Int} =
    [idx for (idx, sym) in enumerate([:pos_1, :pos_2, :pos_3, :yaw]) if sym in corrected_channels]

"""
The stride residual the model does not explain, per channel of the 4-channel
convention (corrected or not), as

    r_i = w_i + (j_{i+1} − j_i)

a per-stride process noise `w` (σ_w²) and a per-footfall jitter `j` (σ_j²) that
telescopes: whatever puts the mocap footfall pose off the INS footfall pose,
including mocap noise. `j` is noise on each mocap fix, not on the stride, so it
does not accumulate. Its moments `γ₀ = σ_w² + 2σ_j²`, `γ₁ = −σ_j²` are estimated
online from the stride errors of the ground-truth half, around their running
mean, with the GP noise hyperparameter as a prior worth `n₀` strides: `γ₀ = σ_n²`,
and `γ₁ = ρ₀σ_n²` with the lag-1 autocorrelation measured across trials (yaw
−0.35 ANG2 / −0.30 DCSC; position ≈0 or positive, so 0).

The hyperparameter alone is not enough: key 42's σ_n is about twice the DCSC
residual, and under mocap noise the jitter is the mocap noise, which nothing
else tells the filter.
"""
mutable struct StrideNoise
    γ₀_prior::Vector{Float64}
    γ₁_prior::Vector{Float64}
    n::Int
    S₁::Vector{Float64}
    S₂::Vector{Float64}
    S₁₂::Vector{Float64}
    last::Vector{Float64}
end

const NOISE_PRIOR_LAG1 = [0.0, 0.0, 0.0, -0.33]
const NOISE_PRIOR_STRIDES = 10

function StrideNoise(params::HsgpParameters)
    σ_n² = [(getfield(params.hp, Symbol(_OUTPUT_NAMES[j]))[1] * params.output_stats[2][j])^2 for j in 1:4]
    return StrideNoise(σ_n², NOISE_PRIOR_LAG1 .* σ_n², 0, zeros(4), zeros(4), zeros(4), zeros(4))
end

function update!(ν::StrideNoise, r::AbstractVector{Float64})
    ν.n > 0 && (ν.S₁₂ .+= ν.last .* r)
    ν.n += 1
    ν.S₁ .+= r
    ν.S₂ .+= r .^ 2
    ν.last .= r
    return ν
end

"`(σ_w, σ_j)` per channel."
function noise_split(ν::StrideNoise)
    n₀, n = NOISE_PRIOR_STRIDES, ν.n
    μ = n > 0 ? ν.S₁ ./ n : zeros(4)
    γ₀ = (n₀ .* ν.γ₀_prior .+ (n > 0 ? ν.S₂ .- n .* μ .^ 2 : 0.0)) ./ (n₀ + n)
    γ₁ = (n₀ .* ν.γ₁_prior .+ (n > 1 ? ν.S₁₂ .- (n - 1) .* μ .^ 2 : 0.0)) ./ (n₀ + max(n - 1, 0))
    σ_j² = clamp.(-γ₁, 0.0, γ₀ ./ 2)
    return sqrt.(max.(γ₀ .- 2σ_j², 0.0)), sqrt.(σ_j²)
end

mutable struct JointStrideStaticEstimator <: AbstractJointStrideEstimator
    t::Vector{Float64}
    pos::Matrix{Float64}
    quat::Matrix{Float64}
    δx::Vector{Float64}
    Σ::Matrix{Float64}
    i::Int
    β::Vector{Float64}          # per-channel stride bias, physical units
    noise::StrideNoise
    params::HsgpParameters
    correction_mask::Vector{Int}
    p::Int
end

function JointStrideStaticEstimator(N::Int; params::HsgpParameters,
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw], kwargs...)
    mask = _channel_mask(corrected_channels)
    p = length(mask)
    return JointStrideStaticEstimator(zeros(N), zeros(3, N), zeros(4, N),
        zeros(6 + p), zeros(6 + p, 6 + p), 1, zeros(p), StrideNoise(params),
        params, mask, p)
end

mutable struct JointStrideHsgpEstimator <: AbstractJointStrideEstimator
    t::Vector{Float64}
    pos::Matrix{Float64}
    quat::Matrix{Float64}
    δx::Vector{Float64}
    Σ::Matrix{Float64}
    i::Int
    β::Vector{Float64}          # HSGP weights, normalised output, channel-major
    noise::StrideNoise
    params::HsgpParameters
    per_dim_eigvals::Matrix{Float64}
    correction_mask::Vector{Int}
    p::Int
end

function JointStrideHsgpEstimator(N::Int; params::HsgpParameters,
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw], kwargs...)
    mask = _channel_mask(corrected_channels)
    p = length(mask)
    nβ = p * params.m
    return JointStrideHsgpEstimator(zeros(N), zeros(3, N), zeros(4, N),
        zeros(6 + nβ), zeros(6 + nβ, 6 + nβ), 1, zeros(nβ), StrideNoise(params),
        params, calc_eigenvalues(params.LL, params.m, params.d), mask, p)
end

"""
    stride_model(c, feature_type, feature) -> (y₀, Φ)

The corrected channels' stride error as `y = y₀ + Φ β`, in physical units.
`feature` is the raw (unnormalised) feature.
"""
function stride_model(c::JointStrideStaticEstimator, ::FeatureType, ::AbstractVector{Float64})
    return zeros(c.p), Matrix{Float64}(I, c.p, c.p)
end

function stride_model(c::JointStrideHsgpEstimator, feature_type::FeatureType, feature::AbstractVector{Float64})
    mask = c.correction_mask
    z = normalize_feature!(feature_type; feature=copy(feature),
        input_stats=c.params.input_stats, mid_norm=c.params.mid_norm)[1]
    ϕ = calc_eigenvectors(reshape(z, 1, c.params.d), c.params.LL, c.per_dim_eigvals)
    return c.params.output_stats[1][mask], kron(Diagonal(c.params.output_stats[2][mask]), ϕ)
end

function _init_prior!(c::JointStrideStaticEstimator, init_model)
    mask = c.correction_mask
    if isnothing(init_model)
        # A constant is the SE kernel's ℓ → ∞ limit, whose prior variance is σ_f².
        c.β .= 0.0
        c.Σ[7:end, 7:end] = Diagonal([(getfield(c.params.hp, Symbol(_OUTPUT_NAMES[j]))[end] *
                                       c.params.output_stats[2][j])^2 for j in mask])
    else
        c.β .= init_model[1][mask]
        c.Σ[7:end, 7:end] = init_model[2][mask, mask]
    end
end

function _init_prior!(c::JointStrideHsgpEstimator, init_model)
    m, mask = c.params.m, c.correction_mask
    rng(j) = ((j-1)*m+1):(j*m)
    if isnothing(init_model)
        c.β .= 0.0
        for (j, orig) in enumerate(mask)
            hp = getfield(c.params.hp, Symbol(_OUTPUT_NAMES[orig]))
            c.Σ[6 .+ rng(j), 6 .+ rng(j)] = Diagonal(
                power_spectral_density(sqrt.(c.per_dim_eigvals), hp[2], hp[3]))
        end
    else
        β_full, Σβ_full = init_model
        for (j1, o1) in enumerate(mask)
            c.β[rng(j1)] = β_full[_full_range(o1, m)]
            for (j2, o2) in enumerate(mask)
                c.Σ[6 .+ rng(j1), 6 .+ rng(j2)] = Σβ_full[_full_range(o1, m), _full_range(o2, m)]
            end
        end
    end
end

function initialize_corrector!(c::AbstractJointStrideEstimator;
    t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64},
    Σpq_init::AbstractMatrix{Float64},
    init_model::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}=nothing, kwargs...)
    c.i = 1
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx .= 0.0
    c.Σ .= 0.0
    c.Σ[1:6, 1:6] = Σpq_init
    _init_prior!(c, init_model)
end

"""
    propagate_stride!(c; t, Δp, Δq, Σpq, R_bh, ins_stride, ref_frame, feature_type, feature) -> (y, Σ_y)

One stride of the corrector's dynamics. `Δp`, `Δq`, `Σpq` are the raw INS
increment as `dynamic_update!` takes it (`Δp` in the INS body frame at the
previous footfall, `Σpq` over `[ε_p^{b_i}; ε_q^{b_{i+1}}]`), `ins_stride` the
same stride in its local frame, and `R_bh` maps that local frame into the INS
body frame. Returns the
applied correction in the 4-channel convention and its predictive covariance
(`ΦΣββΦ' + σ_w²`), or `nothing` for a corrector without a stride model.

For a plain `AbstractEstimator` this is the uncorrected `dynamic_update!`.
"""
function propagate_stride!(c::AbstractEstimator; t::Float64, Δp::AbstractVector{Float64},
    Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    dynamic_update!(c; t=t, Δp=Δp, Δq=Δq, Σpq=Σpq)
    return nothing
end

function propagate_stride!(c::AbstractJointStrideEstimator; t::Float64,
    Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, R_bh::AbstractMatrix{Float64},
    ins_stride::AbstractVector{Float64}, ref_frame::ReferenceFrame,
    feature_type::FeatureType, feature::AbstractVector{Float64}, kwargs...)

    mask, p = c.correction_mask, c.p
    y₀, Φ = stride_model(c, feature_type, feature)
    y = y₀ + Φ * c.β

    # Channel selectors: position channels into R³ (S_p), yaw channel (s_ψ).
    S_p = zeros(3, p)
    s_ψ = zeros(1, p)
    for (j, orig) in enumerate(mask)
        orig == 4 ? (s_ψ[1, j] = 1.0) : (S_p[orig, j] = 1.0)
    end
    e₃ = [0.0, 0.0, 1.0]

    # Nominal propagation. The INS's local stride is placed by the corrector's
    # own local frame (its heading, for HEADING), so the corrector's roll/pitch
    # -- which only mocap position ever constrains -- cannot bend the stride or
    # its yaw increment away from the INS stride the target was built from.
    R = quat_to_matrix(c.quat[:, c.i])
    A_wl = stride_local(ref_frame; R_wb=R, ΔpΔθ3=zeros(4))[3][1:3, 1:3]
    B_p = A_wl * S_p                        # ∂p⁺/∂y in world
    Δp_w = A_wl * ins_stride[1:3] + B_p * y

    q_raw = quat_multiply(c.quat[:, c.i], Δq)
    Δψ_raw = matrix_to_euler(quat_to_matrix(q_raw))[3] - matrix_to_euler(R)[3]
    δψ = wrap_pi(ins_stride[4] + (s_ψ*y)[1] - Δψ_raw)

    c.i += 1
    c.t[c.i] = t
    c.pos[:, c.i] = c.pos[:, c.i-1] + Δp_w
    c.quat[:, c.i] = normalize_quat(quat_multiply(quat_exp([0.0, 0.0, δψ]), q_raw))
    R⁺ = quat_to_matrix(c.quat[:, c.i])

    # Error-state Jacobians: F = [A Bβ; 0 I]. Only the placement frame's
    # attitude moves the stride: heading alone for HEADING.
    Pθ = ref_frame == HEADING ? e₃ * e₃' : Matrix{Float64}(I, 3, 3)
    A = [I -skew(Δp_w)*Pθ; zeros(3, 3) quat_to_matrix(quat_exp([0.0, 0.0, δψ]))]
    B_y = [B_p; e₃ * s_ψ]                   # 6×p, ∂[p⁺; θ⁺]/∂y
    Bβ = B_y * Φ
    B_all = [A_wl zeros(3); zeros(3, 3) e₃]  # 6×4, the same over all channels

    # Σ ← F Σ F' using the block structure: β's own block is unchanged.
    T = hcat(A, Bβ) * c.Σ                   # [A Bβ] Σ, 6 × n
    Σxβ = T[:, 7:end]
    Σxx = T[:, 1:6] * A' + Σxβ * Bβ'

    # Odometry noise (ε_p from the INS body frame through the local frame),
    # then the stride residual the model does not explain.
    G = [A_wl*R_bh' zeros(3, 3); zeros(3, 3) R⁺]
    σ_w, _ = noise_split(c.noise)
    Σxx += G * Σpq * G' + B_all * Diagonal(σ_w .^ 2) * B_all'

    c.Σ[1:6, 1:6] = (Σxx + Σxx') / 2
    c.Σ[1:6, 7:end] = Σxβ
    c.Σ[7:end, 1:6] = Σxβ'
    c.δx .= 0.0

    y_full, Σy_full = zeros(4), zeros(4, 4)
    y_full[mask] = y
    Σy_full[mask, mask] = Φ * c.Σ[7:end, 7:end] * Φ' + Diagonal(σ_w[mask] .^ 2)
    return y_full, Σy_full
end

"""
    observe_stride_error!(c, stride_err)

Feed one ground-truth stride error (both ends under mocap) to the corrector's
noise estimate. A no-op for correctors without one.
"""
observe_stride_error!(c::AbstractEstimator, stride_err::AbstractVector{Float64}) = nothing
observe_stride_error!(c::AbstractJointStrideEstimator, stride_err::AbstractVector{Float64}) =
    update!(c.noise, stride_err)

function posyaw_measurement_update!(c::AbstractJointStrideEstimator;
    curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    θ3 = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    r = [curr_pos - c.pos[:, c.i]; wrap_pi(curr_θ3 - θ3)]

    # H touches only [δp; δθ_z], so H Σ is four rows of Σ.
    rows = [1, 2, 3, 6]
    HΣ = c.Σ[rows, :]
    # Footfall jitter on the fix; horizontal taken isotropic, since the
    # position channels are in the stride's local frame.
    _, σ_j = noise_split(c.noise)
    σ_h² = (σ_j[1]^2 + σ_j[2]^2) / 2
    S = Symmetric(HΣ[:, rows] + Σy + Diagonal([σ_h², σ_h², σ_j[3]^2, σ_j[4]^2]))
    K = HΣ' / S
    c.δx .+= K * (r - c.δx[rows])
    c.Σ .-= K * HΣ
    c.Σ .= (c.Σ .+ c.Σ') ./ 2
end

function relinearize!(c::AbstractJointStrideEstimator)
    c.pos[:, c.i] += c.δx[1:3]
    c.quat[:, c.i] = normalize_quat(quat_multiply(quat_exp(c.δx[4:6]), c.quat[:, c.i]))
    c.β .+= c.δx[7:end]
    c.δx .= 0.0
end

function get_model(c::JointStrideStaticEstimator)::Tuple{Vector{Float64},Matrix{Float64}}
    β, Σβ = zeros(4), zeros(4, 4)
    β[c.correction_mask] = c.β
    Σβ[c.correction_mask, c.correction_mask] = c.Σ[7:end, 7:end]
    return β, Σβ
end

function get_model(c::JointStrideHsgpEstimator)::Tuple{Vector{Float64},Matrix{Float64}}
    m, mask = c.params.m, c.correction_mask
    rng(j) = ((j-1)*m+1):(j*m)
    β, Σβ = zeros(4m), zeros(4m, 4m)
    for (j1, o1) in enumerate(mask)
        β[_full_range(o1, m)] = c.β[rng(j1)]
        for (j2, o2) in enumerate(mask)
            Σβ[_full_range(o1, m), _full_range(o2, m)] = c.Σ[6 .+ rng(j1), 6 .+ rng(j2)]
        end
    end
    return β, Σβ
end
