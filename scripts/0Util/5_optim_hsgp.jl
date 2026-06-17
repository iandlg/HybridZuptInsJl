include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
import Optim, Random, Distributions, Plots
ls = 0.1
var_f = 2.0^2
var_n = 0.25^2
d = 1                     # dimensionality
N_train = 20
N_test = 200
n_dim = d                 # for consistency

L = [1.0 for _ in 1:d]      # half‑length of the domain in each dimension
m = 1000                   # number of basis functions per dimension
margin = 1.5
L_extended = [Li * margin for Li in L]

true_fun(x) = sin(2π * sum(x)) + cos(7π * sum(x) / 2)   # x is a 3‑tuple or vector

# ----------------------------- Generate data -----------------------------
rng = Random.MersenneTwister(5)
# Training points (3D)
x_train = Random.rand(rng, Distributions.Uniform(-L[1], L[1]), N_train, n_dim)
# Test points (inside extended domain, for evaluation)
x_test = Random.rand(rng, Distributions.Uniform(-L_extended[1], L_extended[1]), N_test, n_dim)

if d == 1
    sort!(x_test; dims=1)
end
# Evaluate true function
f_train = [true_fun(x_train[i, :]) for i in 1:N_train]
f_test = [true_fun(x_test[i, :]) for i in 1:N_test]

# Add noise
y_train = f_train + sqrt(var_n) * Random.randn(rng, N_train)
@show size(x_train)
@show size(y_train)
@show size(x_test)
Eft, Varft, theta_final, lik = HybridZuptInsJl.hsgp_regression(x_train, y_train, x_test, m;
    optimizer=Optim.LBFGS(),
    optim_options=Optim.Options(iterations=200, g_tol=1e-6),
    opt=[true, true, true, true],
    predcf=[1, 2]
)

@info theta_final

p1 = Plots.plot(x_test, true_fun.(x_test);
    linewidth=0.5, linestyle=:dash, color=:black, label="True function")
Plots.plot!(p1, x_test, Eft; linewidth=1.5, color=:blue, label="GP mean")
Plots.plot!(p1, x_test, Eft .- 2 .* sqrt.(Varft), fillrange=Eft .+ 2 .* sqrt.(Varft),
    fillalpha=0.3, color=:blue, label="±2σ", linewidth=0)
Plots.scatter!(p1, x_train, y_train; color=:red, markersize=2, label="Observations")
Plots.title!(p1, "Exact GP")
Plots.xlabel!(p1, "x")
Plots.ylabel!(p1, "f(x)")

Plots.plot(p1; layout=(2, 2), size=(1000, 900))
