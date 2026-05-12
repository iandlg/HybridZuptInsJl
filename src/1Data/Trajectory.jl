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
        Rg = cat(Rg[:, :, 1:1], Rg; dims=3)
    end
    if inertial_t[end] > tg[end]
        tg = [tg; inertial_t[end]]
        pg = [pg pg[:, end]]
        Rg = cat(Rg, Rg[:, :, end:end]; dims=3)
    end

    # Linear interp for position
    pos = vcat([Interpolations.LinearInterpolation(tg, pg[i, :])(inertial_t)' for i in 1:3]...)

    quats = [Quaternions.Quaternion(QuatRotation(RotMatrix{3}(Rg[:, :, i])))
             for i in axes(Rg, 3)]
    R_interp = Array{Float64}(undef, 3, 3, length(inertial_t))

    for (j, τ) in enumerate(inertial_t)
        k = searchsortedlast(tg, τ)
        k = clamp(k, 1, length(tg) - 1)
        a = clamp((τ - tg[k]) / (tg[k+1] - tg[k]), 0.0, 1.0)
        q = Quaternions.slerp(quats[k], quats[k+1], a)
        R_interp[:, :, j] .= Matrix(QuatRotation(q.s, q.v1, q.v2, q.v3))
    end

    return Trajectory(inertial_t, pos, R_interp)
end

# 2-D RMSE (x/y only)
function rmse(tr::Trajectory, gt::Trajectory)
    is_compatible(tr, gt) || throw(ArgumentError("incompatible time series"))
    n = size(tr.pos, 2)
    err_sq = sum((tr.pos[1:2, :] .- gt.pos[1:2, :]) .^ 2, dims=1)[:]   # flatten to vector
    return sqrt.(cumsum(err_sq) ./ (1:n))
end

# Step vectors in body frame
function step_vectors_body(tr::Trajectory, seg::Vector{Int})
    steps = zeros(3, length(seg) - 1)
    for k in 2:length(seg)
        Δp = tr.pos[:, seg[k]] - tr.pos[:, seg[k-1]]
        steps[:, k-1] = tr.R_nb[:, :, seg[k-1]]' * Δp   # R at k-1, not k
    end
    return steps
end

# Step vectors in heading frame (yaw-only rotation)
function step_vectors_heading(tr::Trajectory, step_seg::Vector{Int})
    seg = step_seg
    N = length(seg) - 1
    eu = matrix_to_euler(tr.R_nb)[:, seg[1:end-1]]      # (3, N_steps)
    Δpos = tr.pos[:, seg[2:end]] .- tr.pos[:, seg[1:end-1]]
    steps = similar(Δpos)
    for k in 1:N
        eu_nh = [0, 0, eu[3, k]]
        R_hn = euler_to_matrix(eu_nh)'
        steps[:, k] = R_hn * Δpos[:, k]
    end
    return steps
end

# Helper to compute step lengths from a trajectory and segment indices
function step_lengths(traj::Trajectory, segs::Vector{Int})
    n_steps = length(segs) - 1
    lengths = Vector{Float64}(undef, n_steps)
    for i in 1:n_steps
        p1 = traj.pos[:, segs[i]]
        p2 = traj.pos[:, segs[i+1]]
        lengths[i] = sqrt(sum((p2 .- p1) .^ 2))
    end
    return lengths
end