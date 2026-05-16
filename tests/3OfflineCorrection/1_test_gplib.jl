using GaussianProcesses
using Random
using Plots

Random.seed!(20140430)
# Training data
n = 10;                          #number of training points
x = 2π * rand(n);              #predictors
y = sin.(x) + 0.05 * randn(n);   #regressors


#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.1, 0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(x, y, mZero, kern, logObsNoise)       #Fit the GP

GaussianProcesses.get_params(gp)

μ, σ² = predict_y(gp, range(0, stop=2π, length=100));

using Plots  #Load Plots.jl package

GaussianProcesses.plot(gp; xlabel="x", ylabel="y", title="Gaussian process", legend=false, fmt=:png)      # Plot the GP

using Optim
optimize!(gp; method=ConjugateGradient())   # Optimise the hyperparameters

GaussianProcesses.plot(gp; legend=false, fmt=:png)   #Plot the GP after the hyperparameters have been optimised 