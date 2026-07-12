
"""
    power_spectral_density(omega, ls, sigma_f)

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
    ls::Union{Real,AbstractVector{T}},
    sigma_f::Real
)::AbstractVector{T} where T<:Real
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
    calc_eigenvalues(L::AbstractVector{T}, m::Int, d::Int)::AbstractMatrix{T} where T<:Real

Calculate eigenvalues of the Laplacian on `[-L₁,L₁] x ... x [-L_d,L_d]`
with Dirichlet boundary conditions, returning the `m` smallest.

For each dimension `i`, the 1-D eigenvalues are `λ_{n_i} = (π n_i / (2 L_i))²` with
`n_i = 1,2,…`. The full eigenvalues are the sum over dimensions. The function
selects the `m` smallest sums and returns the per-dimension eigenvalue components.

# Arguments
- `L`: Domain half-widths per dimension, length `d`.
- `m`: Number of eigenvalues (and eigenfunctions) to return.
- `d`: Number of input dimensions.

# Returns
- `selected_per_dim_eigenvalues`: Matrix of size `(m, d)` containing the per-dimension
  eigenvalue components for the `m` smallest eigenvalues, sorted in ascending order
  of the summed eigenvalue.
"""
function calc_eigenvalues(L::AbstractVector{<:Real}, m::Int, d::Int)::AbstractMatrix{Float64}
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

    # Compute per-dimension eigenvalues
    per_dim_eigvals = (π * NN ./ (2 .* L_float')) .^ 2   # (total, d)
    total_eigvals = sum(per_dim_eigvals, dims=2)[:]      # (total,)

    # Select m smallest
    sort_idx = sortperm(total_eigvals)[1:m]
    selected = per_dim_eigvals[sort_idx, :]

    return selected
end

function calc_eigenvalues(L::Real, m)::AbstractMatrix{Float64}
    calc_eigenvalues([L], m, 1)
end
"""
    calc_eigenvectors(Xs::AbstractMatrix{<:Real}, L::AbstractVector{<:Real},
                      per_dim_eigvals::AbstractMatrix{<:Real}) -> Matrix{Float64}

Calculate eigenvectors of the Laplacian on a rectangular domain with Dirichlet boundary
conditions. These eigenvectors serve as basis functions for the Hilbert Space Gaussian
Process (HSGP) approximation.

# Arguments
- `Xs`: Input points of size `(n_samples, d)`.
- `L`: Domain half-widths of length `d`, i.e. domain is `[-L₁, L₁] x ... x [-L_d, L_d]`.
- `per_dim_eigvals`: Per-dimension eigenvalues of size `(m, d)`, where each row
  corresponds to a multi-index `(n₁, …, n_d)` and each column `j` gives
  `(π n_j / (2 L_j))²`.

# Returns
- `phi`: Basis matrix of size `(n_samples, m)` containing the eigenvector values
  (product of sine functions) evaluated at the input points.
"""
function calc_eigenvectors(Xs::AbstractMatrix{T}, L::AbstractVector{<:Real},
    per_dim_eigvals::AbstractMatrix{T})::AbstractMatrix{T} where T<:Real
    n, d = size(Xs)
    m = size(per_dim_eigvals, 1)
    @assert length(L) == d "Length of L must equal number of dimensions d"
    @assert size(per_dim_eigvals, 2) == d "per_dim_eigvals must have d columns"

    # term1: sqrt(eigenvalues) with shape (1, m, d)
    sqrt_eig = sqrt.(per_dim_eigvals)
    term1 = reshape(sqrt_eig, 1, m, d)

    # term2: (Xs + L) with shape (n, 1, d)
    term2 = reshape(Xs .+ L', n, 1, d)

    # c = 1 / sqrt(L) as a (1, 1, d) array
    c = reshape(1.0 ./ sqrt.(L), 1, 1, d)

    # phi_raw: sin(term1 * term2) * c, shape (n, m, d)
    phi_raw = c .* sin.(term1 .* term2)

    # Product over the last dimension (d) -> (n, m)
    phi = dropdims(prod(phi_raw, dims=3), dims=3)

    return phi
end

"""
    calc_eigenvectors_dx(Xs::AbstractMatrix{<:Real}, L::AbstractVector{<:Real},
                         per_dim_eigvals::AbstractMatrix{<:Real}, di::Int) -> Matrix{Float64}

Calculate the derivative of the Laplacian eigenvectors on a rectangular domain with
Dirichlet boundary conditions with respect to input dimension `di`.

Each eigenfunction is a product of 1-D sine functions:

    φ(x) = ∏ⱼ (1/√Lⱼ) sin(√λⱼ (xⱼ + Lⱼ))

where √λⱼ = π nⱼ / (2 Lⱼ) is recovered from `per_dim_eigvals`.

Differentiating with respect to xᵢ replaces the i-th factor by its derivative:

    ∂φ/∂xᵢ = (√λᵢ / √Lᵢ) cos(√λᵢ (xᵢ + Lᵢ)) · ∏ⱼ≠ᵢ (1/√Lⱼ) sin(√λⱼ (xⱼ + Lⱼ))

which is equivalent to the original `calc_eigenvectors`, but with the `di`-th factor
replaced by `(√λᵢ / √Lᵢ) cos(√λᵢ (xᵢ + Lᵢ))` instead of `(1/√Lᵢ) sin(...)`.

# Arguments
- `Xs`: Input points of size `(n_samples, d)`.
- `L`: Domain half-widths of length `d`, i.e. domain is `[-L₁,L₁] x … x [-Lₐ,Lₐ]`.
- `per_dim_eigvals`: Per-dimension eigenvalue components of size `(m, d)`, where each
  entry `[k, j]` equals `(π nⱼ / (2 Lⱼ))²`. Matches the output of `calc_eigenvalues`.
- `di`: Dimension index (1-based) with respect to which to differentiate.

# Returns
- `dphi`: Derivative basis matrix of size `(n_samples, m)`.
"""
function calc_eigenvectors_dx(
    Xs::AbstractMatrix{<:Real},
    L::AbstractVector{<:Real},
    per_dim_eigvals::AbstractMatrix{<:Real},
    di::Int
)::Matrix{Float64}
    n, d = size(Xs)
    m = size(per_dim_eigvals, 1)
    @assert length(L) == d "Length of L must equal number of dimensions d"
    @assert size(per_dim_eigvals, 2) == d "per_dim_eigvals must have d columns"
    @assert 1 <= di <= d "di must be a valid dimension index (1 to d)"

    Xs_f = Float64.(Xs)
    L_f = Float64.(L)
    eigvals_f = Float64.(per_dim_eigvals)

    sqrt_eig = sqrt.(eigvals_f)           # (m, d)  — each entry is π nⱼ / (2 Lⱼ)
    term1 = reshape(sqrt_eig, 1, m, d)   # (1, m, d)
    term2 = reshape(Xs_f .+ L_f', n, 1, d)  # (n, 1, d): xⱼ + Lⱼ for all j

    # Base scale factor 1/√Lⱼ, shape (1, 1, d)
    c = reshape(1.0 ./ sqrt.(L_f), 1, 1, d)

    # Build (n, m, d) factor array:
    #   - sine  factor for j ≠ di
    #   - cosine factor (with extra √λᵢ) for j == di
    angles = term1 .* term2               # (n, m, d)

    phi_raw = c .* sin.(angles)           # default: sine factors for all dims

    # Overwrite dimension di with the cosine derivative factor:
    #   (√λᵢ / √Lᵢ) cos(√λᵢ (xᵢ + Lᵢ))
    #   = sqrt_eig[:, di] * c[:, :, di] * cos(angles[:, :, di])
    phi_raw[:, :, di] .= (
        reshape(sqrt_eig[:, di], 1, m) .*   # √λᵢ, broadcast over n
        c[1, 1, di] .*                       # 1/√Lᵢ
        cos.(angles[:, :, di])               # cos(√λᵢ (xᵢ + Lᵢ))
    )

    # Product over dimensions -> (n, m)
    dphi = dropdims(prod(phi_raw, dims=3), dims=3)

    return dphi
end
"""
    nlml(w, y, lambda, Phiy, PhiPhi, d, m, opt, theta, use_linear)

Compute the negative log marginal likelihood and its gradient w.r.t. the
log-transformed hyperparameters for the SE kernel reduced-rank GP.

Only the hyperparameters indicated by `opt` (indices 1:σ_n, 2:ℓ, 3:σ_f, 4:σ_lin) are
being optimised; `w` contains their logs in that order. Non-optimised
hyperparameters are taken from `theta`. When `use_linear=false`, σ_lin is
excluded from the basis and its gradient is never computed.
"""
function nlml(
    w::AbstractVector{T},
    y::AbstractVector{T},
    lambda::AbstractVector{T},
    Phiy::AbstractVector{T},
    PhiPhi::AbstractMatrix{T},
    d::Int,
    m::Int,
    opt::AbstractVector{Bool},
    theta::AbstractVector{T},
    use_linear::Bool
)::Tuple{T,Vector{T}} where {T<:Real}

    θ = copy(theta)
    θ[opt] .= exp.(w)

    σ_n, ℓ, σ_f, σ_lin = θ
    σ_n² = σ_n^2
    σ_f² = σ_f^2
    σ_lin² = σ_lin^2

    n = length(y)
    basis_count = (use_linear ? d : 0) + m   # total number of basis functions

    # Spectral densities (prior variances) for each basis function
    C = (√(2π))^d
    k_se = σ_f² * C * ℓ^d .* exp.(-0.5 * lambda * ℓ^2)
    k = use_linear ? vcat(fill(σ_lin², d), k_se) : k_se

    # A = ΦᵀΦ + σ_n² diag(1/k)
    A = PhiPhi + Diagonal(σ_n² ./ k)

    Lchol = cholesky(A; check=false)
    if !issuccess(Lchol)
        return T(Inf), zeros(T, length(w))
    end
    L = Lchol.L

    v = L \ Phiy
    yiQy = (dot(y, y) - dot(v, v)) / σ_n²
    logdetQ = (n - basis_count) * log(σ_n²) + sum(log.(k)) + 2 * sum(log.(diag(L)))
    nll = 0.5 * yiQy + 0.5 * logdetQ + 0.5 * n * log(2π)

    # Pre-computations shared by all gradients
    vv = L' \ v
    Linv_diag = L \ Diagonal(1 ./ k)
    LLk_diag = diag(L' \ Linv_diag)          # diag((LLᵀ)⁻¹ diag(1/k))

    # ── ∂/∂σ_n² ──────────────────────────────────────────────────────────────
    dlogdetQ_σ² = (n - basis_count) / σ_n² + sum(LLk_diag)
    dyiQy_σ² = dot(vv, (Diagonal(1 ./ k) * vv)) / σ_n² - yiQy / σ_n²
    grad_σ² = 0.5 * (dlogdetQ_σ² + dyiQy_σ²)

    # ── ∂/∂ℓ ─────────────────────────────────────────────────────────────────
    dk_se_dℓ = k_se .* (d / ℓ .- ℓ .* lambda)
    dk_dℓ = use_linear ? vcat(zeros(T, d), dk_se_dℓ) : dk_se_dℓ
    dlogdetQ_ℓ = sum(dk_dℓ ./ k) - σ_n² * sum(LLk_diag .* (dk_dℓ ./ k))
    dyiQy_ℓ = -dot(vv, (Diagonal(dk_dℓ ./ k .^ 2) * vv))
    grad_ℓ = 0.5 * (dlogdetQ_ℓ + dyiQy_ℓ)

    # ── ∂/∂σ_f² ──────────────────────────────────────────────────────────────
    dk_se_dσ_f² = k_se / σ_f²
    dk_dσ_f² = use_linear ? vcat(zeros(T, d), dk_se_dσ_f²) : dk_se_dσ_f²
    dlogdetQ_σ_f² = sum(dk_dσ_f² ./ k) - σ_n² * sum(LLk_diag .* (dk_dσ_f² ./ k))
    dyiQy_σ_f² = -dot(vv, (Diagonal(dk_dσ_f² ./ k .^ 2) * vv))
    grad_σ_f² = 0.5 * (dlogdetQ_σ_f² + dyiQy_σ_f²)

    # ── ∂/∂σ_lin² (only when use_linear=true) ────────────────────────────────
    if use_linear
        dk_dσ_lin² = vcat(ones(T, d), zeros(T, m))
        dlogdetQ_σ_lin² = sum(dk_dσ_lin² ./ k) - σ_n² * sum(LLk_diag .* (dk_dσ_lin² ./ k))
        dyiQy_σ_lin² = -dot(vv, (Diagonal(dk_dσ_lin² ./ k .^ 2) * vv))
        grad_σ_lin² = 0.5 * (dlogdetQ_σ_lin² + dyiQy_σ_lin²)
    end

    # ── Map to log-parameter gradients: ∂/∂(log θ) = θ · ∂/∂θ ──────────────
    grad = similar(w)
    idx = 1
    if opt[1]   # log σ_n  →  2σ_n² · ∂/∂σ_n²
        grad[idx] = 2 * σ_n² * grad_σ²
        idx += 1
    end
    if opt[2]   # log ℓ
        grad[idx] = ℓ * grad_ℓ
        idx += 1
    end
    if opt[3]   # log σ_f  →  2σ_f² · ∂/∂σ_f²
        grad[idx] = 2 * σ_f² * grad_σ_f²
        idx += 1
    end
    if opt[4] && use_linear   # log σ_lin  →  2σ_lin² · ∂/∂σ_lin²
        grad[idx] = 2 * σ_lin² * grad_σ_lin²
        idx += 1
    end

    return nll, grad
end


"""
    hsgp_regression(x, y, xt, m; kwargs...)

Reduced-rank Gaussian process regression (Hilbert-space approximation) with an
optional linear kernel component plus a squared-exponential kernel.

# Arguments
- `x`            : training inputs (n x d)
- `y`            : training targets (n-vector)
- `xt`           : test inputs (nₜ x d)
- `m`            : number of SE basis functions

# Keyword Arguments
- `LL`           : domain bounds (2 x d), or `nothing` → auto-computed with 10 % padding
- `theta`        : `[σₙ, ℓ, σ_f, σ_lin]` (standard deviations); σ_lin is ignored when `use_linear=false`
- `opt`          : which of `theta` to optimise — length-4 `Bool` vector (default all `true`)
- `use_linear`   : include the linear kernel component (default `true`)
- `predcf`       : kernel components for prediction — `[1]` = linear, `[2]` = SE (default `[1, 2]`)
- `lower`        : lower bounds for `[σₙ, ℓ, σ_f, σ_lin]` (default `1e-6`)
- `upper`        : upper bounds for `[σₙ, ℓ, σ_f, σ_lin]` (default `1e6`)
- `rng`          : optional RNG; when provided and `theta` is empty, sample `theta` uniformly in log-space from the bounds
- `optimizer`    : `Optim.jl` optimizer (default `LBFGS()`)
- `optim_options`: `Optim.Options` (default `Optim.Options()`)

# Returns
`(Eft, Varft, theta, lik, Lvec)` where:
- `Eft`   : posterior mean at test points
- `Varft` : posterior marginal variance at test points
- `theta` : final (possibly optimised) hyperparameters
- `lik`   : negative log marginal likelihood at the final hyperparameters
- `Lvec`  : half-widths of the scaled domain
"""
function hsgp_regression(
    x::AbstractMatrix{T},
    y::AbstractVector{T},
    xt::AbstractMatrix{T},
    m::Int;
    LL::Union{Nothing,AbstractMatrix{T}}=nothing,
    theta::AbstractVector{T}=T[],
    opt::AbstractVector{Bool}=trues(4),
    use_linear::Bool=true,
    predcf::AbstractVector{Int}=[1, 2],
    lower::AbstractVector{T}=fill(T(1e-6), 4),
    upper::AbstractVector{T}=fill(T(1e6), 4),
    rng::Optional{Random.AbstractRNG}=nothing,
    optimizer=Optim.LBFGS(),
    optim_options=Optim.Options()
) where {T<:Real}

    d = size(x, 2)

    # Silently exclude σ_lin from optimisation when the linear kernel is off
    effective_opt = copy(opt)
    if !use_linear
        effective_opt[4] = false
    end

    # ---------- Domain boundaries ----------
    xmin = minimum(x, dims=1)[:]
    xmax = maximum(x, dims=1)[:]
    if LL === nothing || isempty(LL)
        pm = 0.1 * minimum(xmax - xmin)
        LL = [xmin' .- pm; xmax' .+ pm]
    end

    # ---------- Default hyperparameters ----------
    if isempty(theta)
        if rng === nothing
            theta = clamp.(T[0.8, 0.2, 1.0, 1e-9], lower, upper)   # σ_n, ℓ, σ_f, σ_lin
        else
            if any(!isfinite, lower) || any(!isfinite, upper)
                error("random start requires finite lower and upper bounds")
            end
            if any(lower .<= zero(T)) || any(upper .<= zero(T))
                error("random start requires positive lower and upper bounds")
            end
            theta = exp.(log.(lower) .+ rand(rng, length(lower)) .* (log.(upper) .- log.(lower)))
            @info "random start : " theta
        end
    else
        theta = clamp.(theta, lower, upper)
    end

    # ---------- Scale inputs to [-Lᵢ, Lᵢ] ----------
    mid = (LL[1, :] .+ LL[2, :]) ./ 2
    Lvec = (LL[2, :] .- LL[1, :]) ./ 2
    @show Lvec
    x = x .- mid'
    xt = xt .- mid'

    # ---------- Eigenbasis for the SE kernel ----------
    per_dim_eigvals = calc_eigenvalues(Lvec, m, d)
    lambda = vec(sum(per_dim_eigvals, dims=2))

    # ---------- Training basis matrix ----------
    Phi_se_train = calc_eigenvectors(x, Lvec, per_dim_eigvals)   # (n, m)
    Phi_train = use_linear ? hcat(x, Phi_se_train) : Phi_se_train  # (n, d+m) or (n, m)

    Phiy = Phi_train' * y
    PhiPhi = Phi_train' * Phi_train

    # ---------- Optimisation ----------
    if any(effective_opt)
        w0 = log.(theta[effective_opt])

        obj(w) = nlml(w, y, lambda, Phiy, PhiPhi, d, m, effective_opt, theta, use_linear)[1]
        function grad!(G, w)
            G[:] = nlml(w, y, lambda, Phiy, PhiPhi, d, m, effective_opt, theta, use_linear)[2]
        end

        lower_w = log.(lower[effective_opt])
        upper_w = log.(upper[effective_opt])

        result = Optim.optimize(obj, grad!, lower_w, upper_w, w0, Optim.Fminbox(optimizer), optim_options)
        theta[effective_opt] = exp.(result.minimizer)
    end

    # ---------- Final NLL ----------
    lik, _ = nlml(log.(theta[effective_opt]), y, lambda, Phiy, PhiPhi, d, m, effective_opt, theta, use_linear)
    isnan(lik) && @warn "Negative log marginal likelihood is invalid: $lik"

    # ---------- Posterior weights ----------
    σ_n, ℓ, σ_f, σ_lin = theta
    C = (√(2π))^d
    k_se = σ_f^2 * C * ℓ^d .* exp.(-0.5 * lambda * ℓ^2)
    k = use_linear ? vcat(fill(σ_lin^2, d), k_se) : k_se

    A = PhiPhi + Diagonal(σ_n^2 ./ k)
    L = cholesky(A).L
    foo = L' \ (L \ Phiy)

    # ---------- Test basis matrix ----------
    lin_requested = use_linear && (1 ∈ predcf)
    Phi_lin_test = lin_requested ? xt : zeros(T, size(xt, 1), d)
    Phi_se_test = (2 ∈ predcf) ? calc_eigenvectors(xt, Lvec, per_dim_eigvals) :
                  zeros(T, size(xt, 1), m)
    Phi_test = use_linear ? hcat(Phi_lin_test, Phi_se_test) : Phi_se_test

    Eft = Phi_test * foo
    V = Phi_test / L'
    Varft = σ_n^2 .* vec(sum(abs2.(V), dims=2))

    return Eft, Varft, theta, lik, Lvec, mid, per_dim_eigvals, foo
end