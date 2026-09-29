
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
