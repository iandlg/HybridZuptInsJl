const MM_TO_M = 0.001

struct Trajectory <: AbstractTimeSeries
    t::Vector{Float64}         # (N,)
    pos::Matrix{Float64}         # (3, N)
    R_nb::Array{Float64,3}       # (3, 3, N)
    vel::Union{Matrix{Float64},Nothing}  # (3, N) or nothing

    function Trajectory(t, pos, R_nb, vel=nothing)
        N = length(t)
        all(diff(t) .> 0) || throw(ArgumentError("t must be strictly increasing"))
        size(pos) == (3, N) || throw(DimensionMismatch("pos must be (3, $N)"))
        size(R_nb) == (3, 3, N) || throw(DimensionMismatch("R_nb must be (3,3,$N)"))
        isnothing(vel) || size(vel) == (3, N) || throw(DimensionMismatch("vel must be (3, $N)"))
        new(t, pos, R_nb, vel)
    end
end

# Slice operator
Base.getindex(tr::Trajectory, mask) = Trajectory(
    tr.t[mask],
    tr.pos[:, mask],
    tr.R_nb[:, :, mask],
    tr.vel === nothing ? nothing : tr.vel[:, mask]
)

# Euler angles (3, N) — roll/pitch/yaw from rotation matrices
function euler_nb(tr::Trajectory)
    N = length(tr.t)
    eu = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        R = RotMatrix{3}(tr.R_nb[:, :, i])
        eu[:, i] .= Rotations.params(RotXYZ(R))  # roll, pitch, yaw
    end
    return eu
end

# Load from CSV (same column convention as Python)
function Trajectory(path::AbstractString)
    df = CSV.read(path, DataFrame; header=false)
    # Drop NaN rows
    df = dropmissing(df)
    # Strictly increasing timestamps
    mask = [true; df[2:end, 1] .> df[1:end-1, 1]]
    df = df[mask, :]
    # Filter zero pos/rot
    pos_ok = vec(sum(Matrix(df[:, 3:5]) .^ 2, dims=2) .!= 0)
    rot_ok = vec(sum(Matrix(df[:, 6:14]) .^ 2, dims=2) .!= 0)
    df = df[pos_ok.&rot_ok, :]
    data = Matrix(df)
    N = size(data, 1)
    t = data[:, 1] ./ 1000
    pos = MM_TO_M .* data[:, 3:5]'          # (3, N)
    R = permutedims(reshape(data[:, 6:14], N, 3, 3), (3, 2, 1))  # (3,3,N)

    return clean(Trajectory(t, pos, R))
end

Trajectory(dir::AbstractString, num::Int) =
    Trajectory(joinpath(dir, "$(num)_Synchronized_Reference.csv"))

# Remove large Euler-angle jumps (same threshold as Python)
function clean(tr::Trajectory)
    eu = matrix_to_euler(tr.R_nb)
    yaw = unwrap(eu[3, :])
    d_roll = abs.(diff(eu[1, :]))
    d_pitch = abs.(diff(eu[2, :]))
    d_yaw = diff(yaw)
    bad = findall((d_roll .> 0.3) .| (d_pitch .> 0.3) .| (d_yaw .> 0.3))
    bad = vcat(bad, bad .+ 1)
    keep = setdiff(1:length(tr.t), bad)
    return tr[keep]
end

# Temporal alignment via linear interp (pos) + Slerp (rotation)
function temporal_alignment(tr::Trajectory, inertial_t::Vector{Float64})
    tg = tr.t
    pg = tr.pos
    Rg = tr.R_nb

    # Zero-order-hold extend left / right
    if inertial_t[begin] < tg[begin]
        tg = [inertial_t[begin]; tg]
        pg = [pg[:, 1] pg]
        Rg = cat(Rg[:, :, 1::nothing], Rg; dims=3)
    end
    if inertial_t[end] > tg[end]
        tg = [tg; inertial_t[end]]
        pg = [pg pg[:, end]]
        Rg = cat(Rg, Rg[:, :, end::nothing]; dims=3)
    end

    # Linear interp for position
    pos = vcat([LinearInterpolation(tg, pg[i, :])(inertial_t)' for i in 1:3]...)

    # SLERP for rotations using Rotations.jl QuatRotation
    quats = [QuatRotation(RotMatrix{3}(Rg[:, :, i])) for i in axes(Rg, 3)]
    R_interp = Array{Float64}(undef, 3, 3, length(inertial_t))
    for (j, τ) in enumerate(inertial_t)
        k = searchsortedlast(tg, τ)
        k = clamp(k, 1, length(tg) - 1)
        a = (τ - tg[k]) / (tg[k+1] - tg[k])
        R_interp[:, :, j] .= Matrix(slerp(quats[k], quats[k+1], a))
    end

    return Trajectory(inertial_t, pos, R_interp)
end

# 2-D RMSE (x/y only)
function rmse(tr::Trajectory, gt::Trajectory)
    is_compatible(tr, gt) || throw(ArgumentError("incompatible time series"))
    return sqrt(mean(sum((tr.pos[1:2, :] .- gt.pos[1:2, :]) .^ 2; dims=1)))
end

# Step vectors in body frame
function step_vectors_body(tr::Trajectory, step_seg::Vector{Int})
    seg = step_seg
    Δpos = tr.pos[:, seg[2:end]] .- tr.pos[:, seg[1:end-1]]       # (3, N_steps)
    steps = similar(Δpos)
    for k in axes(Δpos, 2)
        steps[:, k] = tr.R_nb[:, :, seg[k]]' * Δpos[:, k]          # Rᵀ·Δp
    end
    return steps
end

# Step vectors in heading frame (yaw-only rotation)
function step_vectors_heading(tr::Trajectory, step_seg::Vector{Int})
    seg = step_seg
    N = length(seg) - 1
    eu = euler_nb(tr)[:, seg[1:end-1]]      # (3, N_steps)
    Δpos = tr.pos[:, seg[2:end]] .- tr.pos[:, seg[1:end-1]]
    steps = similar(Δpos)
    for k in 1:N
        ψ = eu[3, k]                            # yaw only
        R_hn = RotMatrix(RotZ(-ψ))               # heading→navigation inverse
        steps[:, k] = R_hn * Δpos[:, k]
    end
    return steps
end