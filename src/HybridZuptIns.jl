module HybridZuptInsJl

using LinearAlgebra, Statistics
using Rotations          # QuatRotation, RotMatrix, Slerp
using Interpolations     # LinearInterpolation
using CSV, DataFrames


include("0Data/TimeSeries.jl")
include("0Data/Trajectory.jl")
include("0Data/InertialData.jl")

export TimeSeries, Trajectory, InertialData
export truncate_to_overlap, is_compatible
export euler_nb, temporal_alignment, rmse
export step_vectors_body, step_vectors_heading
export accel, gyro


end

