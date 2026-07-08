
function ∂θ3_∂δθ_right(R::AbstractMatrix{T})::AbstractVector{T} where T<:Real
    size(R) == (3, 3) || throw(DimensionMismatch("Expected matrix of size (3,3), got $(size(R))"))
    denom = (R[1, 1]^2 + R[2, 1]^2)
    denom < 1e-9 && @warn "`∂θ3_∂δθ_right` Close to singular division, denom = $(denom)"
    return [
        0,
        (R[2, 1] * R[1, 3] - R[1, 1] * R[2, 3]) / (denom),
        (R[1, 1] * R[2, 2] - R[2, 1] * R[1, 2]) / (denom)
    ]
end

function ∂θ3_∂δθ_right(q::AbstractVector{T})::AbstractVector{T} where T<:Real
    length(q) == 4 || throw(DimensionMismatch("Expected vector of length 4, got $(length(q))"))
    return ∂θ3_∂δθ_right(quat_to_matrix(q))
end

function ∂θ3_∂δθ_left(R::AbstractMatrix{T})::AbstractVector{T} where T<:Real
    size(R) == (3, 3) || throw(DimensionMismatch("Expected matrix of size (3,3), got $(size(R))"))
    denom = (R[1, 1]^2 + R[2, 1]^2)
    denom < 1e-9 && @warn "`∂θ3_∂δθ_left` Close to singular division, denom = $(denom)"
    return [
        (-R[1, 1] * R[3, 1]) / denom,
        (-R[2, 1] * R[3, 1]) / denom,
        1.0
    ]
end

function ∂θ3_∂δθ_left(q::AbstractVector{T})::AbstractVector{T} where T<:Real
    length(q) == 4 || throw(DimensionMismatch("Expected vector of length 4, got $(length(q))"))
    return ∂θ3_∂δθ_left(quat_to_matrix(q))
end

function ∂feature_norm𝑥𝑦𝑧_∂δpδθ(R_wl::AbstractMatrix{T}, σ_input_𝑥𝑦𝑧::AbstractVector{T})::AbstractMatrix{T} where T<:Real
    return [Diagonal(1 ./ σ_input_𝑥𝑦𝑧) * R_wl' zeros(T, 3, 3)]
end

function ∂feature_normΔT_∂δpδθ(::Type{T})::AbstractVector{T} where T<:Real
    return zeros(T, 6)
end

function ∂feature_normΔθ3_∂δpδθ(R::AbstractMatrix{T}, σ_input_θ::T)::AbstractVector{T} where T<:Real
    # return [zeros(T, 1, 3) ∂θ3_∂δθ_left(R) ./ σ_output_θ]
    return [zeros(T, 1, 3) [0.0, 0.0, 1.0] ./ σ_input_θ]
end

function ∂feature_normΔθ3_∂δpδθ(q::AbstractVector{T}, σ_input_θ::T)::AbstractVector{T} where T<:Real
    return ∂feature_normΔθ3_∂δpδθ(quat_to_matrix(q), σ_input_θ)
end