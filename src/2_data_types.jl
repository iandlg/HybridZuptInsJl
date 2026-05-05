# types.jl
# Data types for trajectory and inertial measurement data.
# Depends on: CSV, DataFrames, LinearAlgebra

const MM_TO_M = 0.001


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

abstract type TimeSeries end

Base.length(d::TimeSeries) = length(d.t)


# ---------------------------------------------------------------------------
# InertialData
# ---------------------------------------------------------------------------
"""
	InertialData

Data structure containing inertial data
"""
struct InertialData <: TimeSeries
    t::Vector{Float64}   # (N,)    timestamps in seconds
    u::Matrix{Float64}   # (6, N)  stacked [accel; gyro]

    function InertialData(t, u)
        N = length(t)
        size(u) == (6, N) || throw(ArgumentError("u must be (6, $N), got $(size(u))"))
        new(t, u)
    end
end

Base.getindex(d::InertialData, idx) = InertialData(d.t[idx], d.u[:, idx])

"""
	load_inertial(path)

Load raw IMU data from a CSV file.

Expected columns (1-indexed):
	3   : timestamp
	4-6 : accelerometer (x, y, z)
	7-9 : gyroscope (x, y, z)
"""
function load_inertial(path::AbstractString)::InertialData
    df = CSV.read(path, DataFrame)
    t = Float64.(df[:, 3])
    accel = Matrix{Float64}(df[:, 4:6])'   # (3, N)
    gyro = Matrix{Float64}(df[:, 7:9])'   # (3, N)
    return InertialData(t, vcat(accel, gyro))
end

load_inertial(dir::AbstractString, num::Int)::InertialData =
    load_inertial(joinpath(dir, "$(num)_IMURaw.csv"))


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

struct Trajectory <: TimeSeries
    t::Vector{Float64}              # (N,)
    pos::Matrix{Float64}            # (3, N)   position in metres
    R::Array{Float64,3}            # (3, 3, N) rotation matrices
    vel::Union{Matrix{Float64},Nothing}  # (3, N) or nothing

    function Trajectory(t, pos, R, vel=nothing)
        N = length(t)
        size(pos) == (3, N) || throw(ArgumentError("pos must be (3, $N), got $(size(pos))"))
        size(R) == (3, 3, N) || throw(ArgumentError("R must be (3, 3, $N), got $(size(R))"))
        if !isnothing(vel)
            size(vel) == (3, N) || throw(ArgumentError("vel must be (3, $N), got $(size(vel))"))
        end
        new(t, pos, R, vel)
    end
end

Base.getindex(d::Trajectory, idx) = Trajectory(d.t[idx], d.pos[:, idx], d.R[:, :, idx])

"""
	load_trajectory(path)

Load ground truth position and orientation from a CSV file.

Expected columns (1-indexed):
	1     : timestamp (ms)
	3-5   : XYZ position (mm)
	6-14  : flattened row-major 3x3 rotation matrix
"""
function load_trajectory(path::AbstractString)::Trajectory
    print(path)
    df = CSV.read(path, DataFrame, header=false, missingstring=["", "NA", "NaN"])
    data = coalesce.(Matrix(df), 0.0)  # replace all missing with 0.0

    N = size(data, 1)
    t = data[:, 1] ./ 1000.0                        # ms → s
    pos = MM_TO_M .* data[:, 3:5]'                    # (3, N)

    # reshape flat 9-element rows into (3, 3, N), converting row-major → column-major
    R = Array{Float64}(undef, 3, 3, N)
    for i in 1:N
        R[:, :, i] = reshape(data[i, 6:14], 3, 3)'
    end

    # keep only rows where both pos and R are non-zero
    pos_valid = vec(sum(pos .^ 2, dims=1)) .> 0
    R_valid = [sum(R[:, :, i] .^ 2) > 0 for i in 1:N]
    valid = findall(pos_valid .& R_valid)

    return clean_trajectory(Trajectory(t[valid], pos[:, valid], R[:, :, valid]))
end

load_trajectory(dir::AbstractString, num::Int)::Trajectory =
    load_trajectory(joinpath(dir, "$(num)_Synchronized_Reference.csv"))

"""
	clean_trajectory(traj)

Remove entries with implausible Euler-angle jumps (> 0.3 rad).
"""
function clean_trajectory(traj::Trajectory)
    euler = matrix_to_euler(traj.R)           # (3, N) — you supply this function
    roll, pitch = euler[1, :], euler[2, :]
    yaw = unwrap(euler[3, :])

    d_roll = abs.(diff(roll))
    d_pitch = abs.(diff(pitch))
    d_yaw = abs.(diff(yaw))

    bad = findall((d_roll .> 0.3) .| (d_pitch .> 0.3) .| (d_yaw .> 0.3))
    bad = unique(vcat(bad, bad .+ 1))

    keep = setdiff(1:length(traj.t), bad)
    return traj[keep]
end

# simple phase unwrap (mirrors np.unwrap)
function unwrap(angles::Vector{Float64})
    out = copy(angles)
    for i in 2:length(out)
        d = out[i] - out[i-1]
        out[i] -= 2π * round(d / (2π))
    end
    return out
end

