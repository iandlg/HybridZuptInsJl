module HybridZuptInsJl

using LinearAlgebra, Statistics
using Quaternions
using Rotations, DSP          # QuatRotation, RotMatrix, Slerp
using Interpolations     # LinearInterpolation
using CSV, DataFrames, JSON
using GLMakie
import Optim
import LeastSquaresOptim
import GaussianProcesses
import PDMats
using Distributions
using Dates

# Resolve ambiguity between GaussianProcesses.jl and PDMats.jl
# function LinearAlgebra.ldiv!(
#     A::PDMats.PDMat{Float64,Matrix{Float64}},
#     B::AbstractVecOrMat{Float64}
# )
#     LinearAlgebra.ldiv!(A.chol, B)
#     return B
# end

include("0Util/Orientation.jl")
include("0Util/InsConfig.jl")
include("0Util/IO.jl")
include("0Util/Util.jl")
include("0Util/HSGP.jl")
include("0Util/KalmanFilter.jl")
include("1Data/TimeSeries.jl")
include("1Data/Trajectory.jl")
include("1Data/InertialData.jl")
include("1Data/Structs.jl")

include("2ZuptIns/Equations.jl")
include("2ZuptIns/Zupt.jl")
include("2ZuptIns/SmoothZuptIns.jl")
include("2ZuptIns/RigidTransform.jl")

include("3OfflineCorrection/BatchCorrection.jl")
include("3OfflineCorrection/HypVariability.jl")

include("4OnlineCorrection/HybridZuptIns.jl")

include("Plotting/Trajectory.jl")
include("Plotting/InertialData.jl")
include("Plotting/Steps.jl")
include("Plotting/OfflineCorrection.jl")
include("Plotting/PlotHypVariability.jl")

export TimeSeries, Trajectory, InertialData
export truncate_to_overlap, is_compatible
export temporal_alignment, rmse
export step_vectors_body, step_vectors_heading
export accel, gyro


end

