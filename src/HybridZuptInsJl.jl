module HybridZuptInsJl

using LinearAlgebra, Statistics
using Rotations, DSP          # QuatRotation, RotMatrix, Slerp
using Interpolations     # LinearInterpolation
using CSV, DataFrames
using GLMakie

include("0Util/Orientation.jl")
include("0Util/InsConfig.jl")
include("1Data/TimeSeries.jl")
include("1Data/Trajectory.jl")
include("1Data/InertialData.jl")
include("2ZuptIns/Equations.jl")
include("2ZuptIns/Zupt.jl")
include("2ZuptIns/SmoothZuptIns.jl")

include("Plotting/Trajectory.jl")
include("Plotting/InertialData.jl")

export TimeSeries, Trajectory, InertialData
export truncate_to_overlap, is_compatible
export euler_nb, temporal_alignment, rmse
export step_vectors_body, step_vectors_heading
export accel, gyro


end

