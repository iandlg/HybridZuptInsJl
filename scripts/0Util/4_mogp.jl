using CairoMakie
using LinearAlgebra
using Random

# ── Kernels & model ────────────────────────────────────────────────────────────

exp_kernel(x, y, ls) = exp.(-0.5 * ((x .- y') .^ 2) / ls^2)

function coregion_matrix(W, κ)
    return W * W' + Diagonal(κ)
end

function icm_kernel(X_blocks, W, κ, ls)
    d = length(X_blocks)
    B = coregion_matrix(W, κ)
    N_each = length.(X_blocks)
    offsets = [0; cumsum(N_each)]
    N = sum(N_each)
    K = zeros(N, N)
    for i in 1:d, j in 1:d
        rows = offsets[i]+1:offsets[i+1]
        cols = offsets[j]+1:offsets[j+1]
        K[rows, cols] = B[i, j] .* exp_kernel(X_blocks[i], X_blocks[j], ls)
    end
    return K
end

function predict(X_blocks, y, X_star, out_idx, W, κ, ls, σ²)
    d = length(X_blocks)
    B = coregion_matrix(W, κ)
    N_each = length.(X_blocks)
    offsets = [0; cumsum(N_each)]
    K = icm_kernel(X_blocks, W, κ, ls)
    Kn = Symmetric(K + σ² * I)
    L = cholesky(Kn).L
    N_star = length(X_star)
    K_star = zeros(N_star, sum(N_each))
    for j in 1:d
        cols = offsets[j]+1:offsets[j+1]
        K_star[:, cols] = B[out_idx, j] .* exp_kernel(X_star, X_blocks[j], ls)
    end
    α = L' \ (L \ y)
    μ_star = K_star * α
    v = L \ K_star'
    k_ss = B[out_idx, out_idx] .* ones(N_star)
    σ²_star = max.(k_ss .- vec(sum(v .^ 2, dims=1)), 0.0)
    return μ_star, σ²_star
end
# ── Likelihood ─────────────────────────────────────────────────────────────────

"""
Log marginal likelihood: -0.5*(y'K⁻¹y + log|K| + N*log(2π))
"""
function log_marginal_likelihood(y, K, σ²)
    N = length(y)
    Kn = K + σ² * I                       # add noise
    L = cholesky(Symmetric(Kn)).L
    α = L' \ (L \ y)
    return -0.5 * (y'α + 2 * sum(log.(diag(L))) + N * log(2π))
end

using Optim

function objective(θ)
    W_ = reshape(θ[1:d*rank], d, rank)
    κ_ = exp.(θ[d*rank+1:d*rank+d])    # exp to keep positive
    ls_ = exp(θ[end-1])
    σ²_ = exp(θ[end])
    return -log_marginal_likelihood(y, icm_kernel(X_blocks, W_, κ_, ls_), σ²_)
end

# ── Data ───────────────────────────────────────────────────────────────────────

rng = MersenneTwister(42)
N1 = 5
N2 = 2
x1 = collect(range(0, 5, length=N1))
x2 = collect(range(0, 5, length=N2))
y1 = sin.(x1) .+ 0.1 .* randn(rng, N1)
y2 = sin.(x2) .+ 0.3 .+ 0.1 .* randn(rng, N2)   # offset but correlated
y = vcat(y1, y2)
X_blocks = [x1, x2]

# ── Hyperparameters ────────────────────────────────────────────────────────────

d = 2
rank = 1
W = fill(0.8, d, rank)
κ = fill(0.1, d)
ls = 3.0
σ² = 0.01

θ = vcat(vec(W), log.(κ), log(ls), log(σ²))
result = optimize(objective, θ, NelderMead())

display(θ)
W = reshape(θ[1:d*rank], (d, rank))
κ = exp.(θ[(d*rank+1):(d*rank+d)])
ls = exp.(θ[(d*rank+1+d)])
σ² = exp.(θ[(d*rank+2+d)])
# σ² = 20
# ── Predictions ───────────────────────────────────────────────────────────────

x_star = collect(range(-0.5, 5.5, length=200))

μ1, σ²1 = predict(X_blocks, y, x_star, 1, W, κ, ls, σ²)
μ2, σ²2 = predict(X_blocks, y, x_star, 2, W, κ, ls, σ²)

σ1 = sqrt.(σ²1)
σ2 = sqrt.(σ²2)

# ── Plot ───────────────────────────────────────────────────────────────────────

fig = Figure(size=(900, 500), fontsize=13)

colors = [:royalblue, :tomato]

for (i, (μ, σ, xi, yi, color, title)) in enumerate(zip(
    [μ1, μ2], [σ1, σ2], [x1, x2], [y1, y2], colors,
    ["Output 1  —  sin(x) + noise", "Output 2  —  sin(x) + 0.3 + noise"]))

    ax = Axis(fig[1, i],
        title=title,
        xlabel="x",
        ylabel=i == 1 ? "f(x)" : "",
        titlesize=14,
        xlabelsize=12,
        ylabelsize=12,
    )

    # 2σ confidence band
    band!(ax, x_star, μ .- 2σ, μ .+ 2σ,
        color=(color, 0.15))

    # 1σ confidence band
    band!(ax, x_star, μ .- σ, μ .+ σ,
        color=(color, 0.25))

    # Posterior mean
    lines!(ax, x_star, μ,
        color=color,
        linewidth=2.5,
        label="posterior mean")

    # Training observations
    scatter!(ax, xi, yi,
        color=:white,
        strokecolor=color,
        strokewidth=2,
        markersize=8,
        label="observations")

    axislegend(ax, position=:lt, labelsize=11, framevisible=false)
    xlims!(ax, -0.5, 5.5)
end

Label(fig[0, :],
    "Multioutput GP — ICM (intrinsic coregionalization model)",
    fontsize=16, font=:bold)

fig