include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

m = 5
d = 3
ls = 0.5
var_f = 2.1
L = [1, 2, 3]

@show eig_per_dim = HybridZuptInsJl.calc_eigenvalues(L, m, d)
display(eig_per_dim)


x = [1 2 3; 4 5 6]
eig_vects = HybridZuptInsJl.calc_eigenvectors(x, L, eig_per_dim)
display(eig_vects)

eig_vects_dx1 = HybridZuptInsJl.calc_eigenvectors_dx(x, L, eig_per_dim, 1)
display(eig_vects_dx1)

##
ls = 0.1
var_f = 2^2
var_n = 0.25^2
d = 1
N_train = 50
N_plot = 500
n_dim = 1

L = [1]
m = 30
margin = 2
L_extended = [Li * margin for Li in L]

## 
using LinearAlgebra, Distributions, Random, Plots

# ---------- Squared exponential kernel ----------
exp_kernel(x, y, ls) = exp.(-0.5 * ((x .- y') .^ 2) / ls^2)

# ---------- Ground truth function ----------
true_fun(x) = sin.(2π * x) + cos.(7π * x / 2)

# ---------- Generate training data ----------
rng = MersenneTwister(5)
x_train = rand(rng, Uniform(-L[1], L[1]), N_train)
x_plot = range(-5, 5, length=N_plot)

# Evaluate true function at training and plotting points
f_train = true_fun(x_train)
f_plot = true_fun(x_plot)

# Add noise to observations
y_train = f_train + sqrt(var_n) * randn(rng, N_train)

# ---------- Compute posterior predictive (optional, if you want GP) ----------
# Not strictly needed for the plot, but to compute confidence bands as in the Python code:
K_ff = var_f * exp_kernel(x_plot, x_plot, ls)
std_f = sqrt.(diag(K_ff))

# ---------- Plot ----------
plot(x_plot, f_plot, linewidth=2, label="True function", color=:black)
scatter!(x_train, y_train, markersize=4, label="Noisy observations", color=:red)
plot!(x_plot, f_plot .+ 2 * std_f, linewidth=0, fillrange=f_plot .- 2 * std_f,
    fillalpha=0.3, color=:gray, label="±2σ of f")
plot!(xlabel="x", ylabel="f(x)", title="Synthetic Data from True Function")


## 
x_test = range(-L_extended[1], L_extended[1]; length=N_plot)
# Kernel matrix between training points
K_yy = var_f * exp_kernel(x_train, x_train, ls) + var_n * I(N_train)
# Inverse of K_yy (for covariance formula)
K_yy_inv = inv(K_yy)

# Cross kernel matrices
K_sx = var_f * exp_kernel(x_test, x_train, ls)   # test x train
K_xs = var_f * exp_kernel(x_train, x_test, ls)   # train x test
K_ss = var_f * exp_kernel(x_test, x_test, ls)    # test x test

# Posterior mean and covariance
gp_mean = K_sx * (K_yy \ y_train)                # more stable: solve once
gp_cov = K_ss - K_sx * K_yy_inv * K_xs

# To get marginal variances (for confidence bands)
gp_std = sqrt.(diag(gp_cov))

## 

# ---------- HSGP posterior ----------
# 1. Eigenvalues and basis matrices
eigvals = HybridZuptInsJl.calc_eigenvalues(L_extended, m, d)       # (m, d)
x_train_mat = reshape(x_train, :, 1)
x_test_mat = reshape(x_test, :, 1)

phi = HybridZuptInsJl.calc_eigenvectors(x_train_mat, L_extended, eigvals)
phi_star = HybridZuptInsJl.calc_eigenvectors(x_test_mat, L_extended, eigvals)
# 2. Power spectral density (includes σ_f² factor)
omega = sqrt.(eigvals)                                           # (m, d)
psd = HybridZuptInsJl.power_spectral_density(omega, ls, sqrt(var_f))   # (m,)

# 3. Build matrix A = var_n * diag(1/psd) + ΦᵀΦ
A = var_n * Diagonal(1.0 ./ psd) + phi' * phi                    # (m, m)

# 4. Solve A α = Φᵀ y
α = A \ (phi' * y_train)                                         # (m,)

# 5. Posterior mean
hsgp_mean = phi_star * α                                         # (N_test,)

# 6. Posterior covariance: var_n * Φ_* * A^{-1} * Φ_*ᵀ
# Solve A * W = Φ_*ᵀ  => W = A \ phi_star'
W = A \ phi_star'                                                # (m, N_test)

hsgp_cov = var_n * phi_star * W                                  # (N_test, N_test)

# (Optional) marginal variances for confidence bands
hsgp_std = sqrt.(diag(hsgp_cov))

## 
using Plots

# Assume the following are already defined:
# x_test, gp_mean, gp_std, hsgp_mean, hsgp_std, x_train, y_train, true_fun

# Left subplot: Exact GP
p1 = plot(x_test, true_fun.(x_test);
    linewidth=0.5, linestyle=:dash, color=:black, label="True function")
plot!(p1, x_test, gp_mean; linewidth=1.5, color=:blue, label="GP mean")
plot!(p1, x_test, gp_mean .- 2 .* gp_std, fillrange=gp_mean .+ 2 .* gp_std,
    fillalpha=0.3, color=:blue, label="±2σ", linewidth=0)
scatter!(p1, x_train, y_train; color=:red, markersize=2, label="Observations")
title!(p1, "Exact GP")
xlabel!(p1, "x")
ylabel!(p1, "f(x)")

# Right subplot: HSGP
p2 = plot(x_test, true_fun.(x_test);
    linewidth=0.5, linestyle=:dash, color=:black, label="True function")
plot!(p2, x_test, hsgp_mean; linewidth=1.5, color=:blue, label="HSGP mean")
plot!(p2, x_test, hsgp_mean .- 2 .* hsgp_std, fillrange=hsgp_mean .+ 2 .* hsgp_std,
    fillalpha=0.3, color=:blue, label="±2σ", linewidth=0)
scatter!(p2, x_train, y_train; color=:red, markersize=2, label="Observations")
title!(p2, "HSGP")
xlabel!(p2, "x")
ylabel!(p2, "f(x)")

# Combine both subplots side by side
plot(p1, p2; layout=(1, 2), size=(1000, 400))