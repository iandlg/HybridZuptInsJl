
"""
    power_spectral_density(omega::AbstractMatrix{T}, ls::Union{Real,AbstractVector}, sigma_f::Real) where T<:Real -> Vector{T}

Power spectral density (PSD) for the Squared Exponential (SE) kernel.

# Mathematical definition

```math
S(\\boldsymbol{\\omega}) = \\sigma_f^2 (\\sqrt{2\\pi})^D
    \\left(\\prod_{i=1}^{D} \\ell_i\\right)
    \\exp\\left(-\\frac{1}{2} \\sum_{i=1}^{D} \\ell_i^2 \\omega_i^2\\right)

Arguments

    omega: Matrix of frequencies, size (m_star, d), where m_star is the number of
    frequency points and d is the input dimension. Each column corresponds to one dimension.

    ls: Length scale(s). Can be a scalar (same for all dimensions) or a vector of length d.

    sigma_f: Standard deviation (signal amplitude) of the SE kernel.

Returns

    A vector of length m_star containing the PSD value for each frequency point.

Example
julia

# Isotropic length scale (same for both dimensions)
omega = randn(100, 2)
psd = power_spectral_density(omega, 0.5, 1.0)

# Anisotropic length scales
psd = power_spectral_density(omega, [0.3, 1.2], 2.0)

"""
function power_spectral_density(
    omega::AbstractMatrix{T},
    ls::Union{Real,AbstractVector},
    sigma_f::Real
) where T<:Real
    d = size(omega, 2)
    ls_vec = ls isa Real ? fill(ls, d) : vec(ls)
    c = (sqrt(2π))^d
    prod_ls = prod(ls_vec)
    ls_sq = ls_vec .^ 2
    exponent = -0.5 * (omega .^ 2 * ls_sq)
    exp_term = exp.(exponent)
    return sigma_f^2 * c * prod_ls * exp_term
end


"""
    calc_eigenvalues(L::AbstractVector{<:Real}, m::Int, d::Int) -> Matrix{Float64}

Calculate eigenvalues of the Laplacian on `[-L₁,L₁] × ... × [-L_d,L_d]`
with Dirichlet boundary conditions, returning the `m` smallest.

For each dimension `i`, the 1‑D eigenvalues are `λ_{n_i} = (π n_i / (2 L_i))²` with
`n_i = 1,2,…`. The full eigenvalues are the sum over dimensions. The function
selects the `m` smallest sums and returns the per‑dimension eigenvalue components.

# Arguments
- `L`: Domain half‑widths per dimension, length `d`.
- `m`: Number of eigenvalues (and eigenfunctions) to return.
- `d`: Number of input dimensions.

# Returns
- `selected_per_dim_eigenvalues`: Matrix of size `(m, d)` containing the per‑dimension
  eigenvalue components for the `m` smallest eigenvalues, sorted in ascending order
  of the summed eigenvalue.
"""
function calc_eigenvalues(L::AbstractVector{<:Real}, m::Int, d::Int)::Matrix{Float64}
    L_float = Float64.(L)
    L_min = minimum(L_float)

    # Number of indices per dimension (at least 1)
    N_per_dim = ceil.(Int, m^(1 / d) * L_float ./ L_min)
    @show N_per_dim

    # Generate all index combinations
    indices_per_dim = [1:N_per_dim[i] for i in 1:d]
    grid = collect(Iterators.product(indices_per_dim...))   # Array of tuples

    # Flatten each coordinate and combine into a (total, d) matrix
    NN = hcat([vec(getindex.(grid, i)) for i in 1:d]...)   # (total, d)

    # Compute per‑dimension eigenvalues
    per_dim_eigvals = (π * NN ./ (2 .* L_float')) .^ 2   # (total, d)
    total_eigvals = sum(per_dim_eigvals, dims=2)[:]      # (total,)

    # Select m smallest
    sort_idx = sortperm(total_eigvals)[1:m]
    selected = per_dim_eigvals[sort_idx, :]

    return selected
end

"""
    calc_eigenvectors(Xs::AbstractMatrix{<:Real}, L::AbstractVector{<:Real},
                      per_dim_eigvals::AbstractMatrix{<:Real}) -> Matrix{Float64}

Calculate eigenvectors of the Laplacian on a rectangular domain with Dirichlet boundary
conditions. These eigenvectors serve as basis functions for the Hilbert Space Gaussian
Process (HSGP) approximation.

# Arguments
- `Xs`: Input points of size `(n_samples, d)`.
- `L`: Domain half‑widths of length `d`, i.e. domain is `[-L₁, L₁] × ... × [-L_d, L_d]`.
- `per_dim_eigvals`: Per‑dimension eigenvalues of size `(m, d)`, where each row
  corresponds to a multi‑index `(n₁, …, n_d)` and each column `j` gives
  `(π n_j / (2 L_j))²`.

# Returns
- `phi`: Basis matrix of size `(n_samples, m)` containing the eigenvector values
  (product of sine functions) evaluated at the input points.
"""
function calc_eigenvectors(Xs::AbstractMatrix{<:Real}, L::AbstractVector{<:Real},
    per_dim_eigvals::AbstractMatrix{<:Real})::Matrix{Float64}
    n, d = size(Xs)
    m = size(per_dim_eigvals, 1)
    @assert length(L) == d "Length of L must equal number of dimensions d"
    @assert size(per_dim_eigvals, 2) == d "per_dim_eigvals must have d columns"

    # Convert to Float64 for numerical stability
    Xs_f = Float64.(Xs)
    L_f = Float64.(L)
    eigvals_f = Float64.(per_dim_eigvals)

    # term1: sqrt(eigenvalues) with shape (1, m, d)
    sqrt_eig = sqrt.(eigvals_f)
    term1 = reshape(sqrt_eig, 1, m, d)

    # term2: (Xs + L) with shape (n, 1, d)
    term2 = reshape(Xs_f .+ L_f', n, 1, d)

    # c = 1 / sqrt(L) as a (1, 1, d) array
    c = reshape(1.0 ./ sqrt.(L_f), 1, 1, d)

    # phi_raw: sin(term1 * term2) * c, shape (n, m, d)
    phi_raw = c .* sin.(term1 .* term2)

    # Product over the last dimension (d) -> (n, m)
    phi = dropdims(prod(phi_raw, dims=3), dims=3)

    return phi
end