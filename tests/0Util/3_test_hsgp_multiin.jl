include("../../src/HybridZuptInsJl.jl")
using .HybridZuptInsJl
using LinearAlgebra, Distributions, Random, Plots, GaussianProcesses, OrderedCollections

# ----------------------------- Parameters -----------------------------
ls = 0.1
var_f = 2.0^2
var_n = 0.25^2
d = 3                     # dimensionality
N_train = 400
N_test = 200
n_dim = d                 # for consistency

L = [1.0 for _ in 1:d]      # half‑length of the domain in each dimension
m = 2000                   # number of basis functions per dimension
margin = 1.5
L_extended = [Li * margin for Li in L]

# ----------------------------- Kernel (isotropic SE) -----------------------------
function sq_exp_kernel(X1, X2, ls, var_f)
    # X1: N1 x d, X2: N2 x d → returns N1 x N2 matrix
    dist_sq = pairwise_sq_dist(X1, X2)   # squared Euclidean distances
    return var_f * exp.(-0.5 * dist_sq / ls^2)
end

function pairwise_sq_dist(X1, X2)
    # efficient pairwise squared distances
    n1, d = size(X1)
    n2, d2 = size(X2)
    @assert d == d2
    sum1 = vec(sum(X1 .^ 2, dims=2))
    sum2 = vec(sum(X2 .^ 2, dims=2))
    return sum1 .+ sum2' .- 2 * X1 * X2'
end

# ----------------------------- True function -----------------------------
true_fun(x) = sin(2π * sum(x)) + cos(7π * sum(x) / 2)   # x is a 3‑tuple or vector

# ----------------------------- Generate data -----------------------------
rng = MersenneTwister(5)
# Training points (3D)
x_train = rand(rng, Uniform(-L[1], L[1]), N_train, n_dim)
# Test points (inside extended domain, for evaluation)
x_test = rand(rng, Uniform(-L_extended[1], L_extended[1]), N_test, n_dim)

# Evaluate true function
f_train = [true_fun(x_train[i, :]) for i in 1:N_train]
f_test = [true_fun(x_test[i, :]) for i in 1:N_test]

# Add noise
y_train = f_train + sqrt(var_n) * randn(rng, N_train)

println("Data generated: $(N_train) training points, $(N_test) test points in $(d)D.")

# ----------------------------- Exact GP (batch) -----------------------------
K_yy = sq_exp_kernel(x_train, x_train, ls, var_f) + var_n * I(N_train)
K_sx = sq_exp_kernel(x_test, x_train, ls, var_f)   # test × train
K_xs = sq_exp_kernel(x_train, x_test, ls, var_f)   # train × test
K_ss = sq_exp_kernel(x_test, x_test, ls, var_f)   # test × test

# Cholesky for stability
C = cholesky(K_yy)
gp_mean = K_sx * (C \ y_train)
v = C \ K_xs
gp_cov = K_ss - K_xs' * v
gp_std = sqrt.(diag(gp_cov))

# ----------------------------- HSGP (batch) -----------------------------
# Eigenvalues (m per dimension, shape m × d)
eigvals = HybridZuptInsJl.calc_eigenvalues(L_extended, m, d)          # (m, d)
# Basis matrices
phi = HybridZuptInsJl.calc_eigenvectors(x_train, L_extended, eigvals)   # (N_train, m)
phi_star = HybridZuptInsJl.calc_eigenvectors(x_test, L_extended, eigvals)   # (N_test,  m)

# Power spectral density (frequency = sqrt(eigenvalues))
ω = sqrt.(eigvals)                     # (m, d)
psd = HybridZuptInsJl.power_spectral_density(ω, ls, sqrt(var_f))   # (m,)

# Build and solve A α = Φᵀ y
A = var_n * Diagonal(1.0 ./ psd) + phi' * phi       # (m, m)
α = A \ (phi' * y_train)

# Posterior mean and variance
hsgp_mean = phi_star * α
# Covariance: var_n * Φ_* * A^{-1} * Φ_*ᵀ
W = A \ phi_star'
hsgp_cov = var_n * phi_star * W
hsgp_std = sqrt.(diag(hsgp_cov))

# ----------------------------- HSGP (sequential) -----------------------------
per_dim_eigvals = HybridZuptInsJl.calc_eigenvalues(L_extended, m, d)
ω_seq = sqrt.(per_dim_eigvals)
psd_seq = HybridZuptInsJl.power_spectral_density(ω_seq, ls, sqrt(var_f))
β = zeros(Float64, m)
Pβ = Diagonal(psd_seq)      # initial covariance (prior precision = diag(psd))

for i in 1:N_train
    φ_i = HybridZuptInsJl.calc_eigenvectors(reshape(x_train[i, :], 1, d), L_extended, per_dim_eigvals)
    β, Pβ = HybridZuptInsJl.measurement_update(β, Pβ, [y_train[i]], φ_i, fill(var_n, 1, 1))
end

# Predict on test points
φ_test = HybridZuptInsJl.calc_eigenvectors(x_test, L_extended, per_dim_eigvals)
hsgp_seq_mean = φ_test * β
hsgp_seq_cov = φ_test * Pβ * φ_test'
hsgp_seq_std = sqrt.(diag(hsgp_seq_cov))

# ----------------------------- GP with GaussianProcesses.jl (reference) -----------------------------
using GaussianProcesses
# Isotropic SE kernel with fixed parameters
kernel = SE(log(ls), log(sqrt(var_f)))
gp_ref = GP(x_train', y_train, MeanZero(), kernel, log(sqrt(var_n)))
gp_ref_mean, gp_ref_var = predict_f(gp_ref, x_test')
gp_ref_std = sqrt.(gp_ref_var)

# ----------------------------- Comparison (RMSE) -----------------------------
# RMSE of posterior mean
rmse_gp_ref = sqrt(mean((gp_mean - gp_ref_mean) .^ 2))
rmse_hsgp = sqrt(mean((gp_mean - hsgp_mean) .^ 2))
rmse_seq = sqrt(mean((gp_mean - hsgp_seq_mean) .^ 2))

# RMSE of posterior standard deviation (restricted to interior points to avoid boundary effects)
interior = all(abs.(x_test) .< L[1], dims=2)[:]   # points strictly inside original domain
rmse_std_hsgp = sqrt(mean((gp_std[interior] .- hsgp_std[interior]) .^ 2))
rmse_std_seq = sqrt(mean((gp_std[interior] .- hsgp_seq_std[interior]) .^ 2))

println("\n--- Comparison on 3D data (isotropic SE kernel) ---")
println("RMSE (mean)   Exact GP vs GP.jl    : $(round(rmse_gp_ref, digits=6))")
println("RMSE (mean)   Exact GP vs batch HSGP: $(round(rmse_hsgp, digits=6))")
println("RMSE (mean)   Exact GP vs seq HSGP  : $(round(rmse_seq, digits=6))")
println("RMSE (std)    Exact GP vs batch HSGP: $(round(rmse_std_hsgp, digits=6)) (interior only)")
println("RMSE (std)    Exact GP vs seq HSGP  : $(round(rmse_std_seq, digits=6))  (interior only)")

N = length(gp_mean[interior])
preds = OrderedDict(
    "GP" => HybridZuptInsJl.CorrectionIO(float.(collect(1:N)), gp_mean[interior], gp_std[interior]),
    "GP.jl" => HybridZuptInsJl.CorrectionIO(float.(collect(1:N)), gp_ref_mean[interior], gp_ref_std[interior]),
    "HSGP" => HybridZuptInsJl.CorrectionIO(float.(collect(1:N)), hsgp_mean[interior], hsgp_std[interior]),
    "HSGP seq" => HybridZuptInsJl.CorrectionIO(float.(collect(1:N)), hsgp_seq_mean[interior], hsgp_seq_std[interior]),
)
HybridZuptInsJl.plot_regression_results(preds)