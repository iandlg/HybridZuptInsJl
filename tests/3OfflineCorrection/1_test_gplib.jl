using GaussianProcesses
using Random
using Optim
using LinearAlgebra

import PDMats

# Resolve ambiguity between GaussianProcesses.jl and PDMats.jl
function LinearAlgebra.ldiv!(
    A::PDMats.PDMat{Float64,Matrix{Float64}},
    B::AbstractVecOrMat{Float64}
)
    LinearAlgebra.ldiv!(A.chol, B)
    return B
end
#Training data
d, n = 2, 50;         #Dimension and number of observations
x = 2π * rand(d, n);                               #Predictors
y = vec(sin.(x[1, :]) .* sin.(x[2, :])) + 0.05 * rand(n);  #Responses

mZero = MeanZero()                             # Zero mean function
kern = SE(0.0, 0.0)    # Sum kernel with Matern 5/2 ARD kernel 
# with parameters [log(ℓ₁), log(ℓ₂)] = [0,0] and log(σ) = 0
# and Squared Exponential Iso kernel with
# parameters log(ℓ) = 0 and log(σ) = 0
@show size(x)
@show size(y)
gp = GP(x, y, mZero, kern, -2.0)          # Fit the GP

μ, σ² = predict_y(gp, 2π * rand(d, n));

optimize!(gp; ker)