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
Base.lastindex(tr::Trajectory) = length(tr)

function Trajectory(dir::AbstractString, id::Int, kwargs...)
    src = resolve_source(dir)
    t, pos, R_nb, vel = read_raw_trajectory(src, dir, id, kwargs...)
    return clean(Trajectory(t, pos, R_nb, vel))
end

function read_raw_trajectory(::ANG, dir::AbstractString, id::Int, kwargs...)
    path = joinpath(dir, "$(id)_Synchronized_Reference.csv")
    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"

    df = CSV.read(path, DataFrame; header=false)
    # Drop NaN rows
    df = dropmissing(df)
    # Strictly increasing timestamps
    mask = [true; df[2:end, 1] .> df[1:(end-1), 1]]
    df = df[mask, :]
    # Filter zero pos/rot
    pos_ok = vec(sum(Matrix(df[:, 3:5]) .^ 2, dims=2) .!= 0)
    rot_ok = vec(sum(Matrix(df[:, 6:14]) .^ 2, dims=2) .!= 0)
    df = df[pos_ok .& rot_ok, :]
    data = Matrix(df)
    N = size(data, 1)
    t = data[:, 1] ./ 1000
    pos = MM_TO_M .* data[:, 3:5]'          # (3, N)
    R = permutedims(reshape(data[:, 6:14], N, 3, 3), (3, 2, 1))  # (3,3,N)
    return t, pos, R, nothing
end

function read_raw_trajectory(::MTI, dir::AbstractString, id::Int, kwargs...)
    error("Not implemented")
end

function read_raw_trajectory(::ANG2, dir::AbstractString, id::Int, kwargs...)
    subdirs = filter(readdir(dir)) do entry
        isdir(joinpath(dir, entry)) || return false
        s = string(id)
        startswith(entry, s) && (length(entry) == length(s) || !isdigit(entry[length(s)+1]))
    end
    isempty(subdirs) && error("No subdirectory starting with '$id' found in $dir")
    length(subdirs) > 1 && @warn "Multiple matches for id=$id in $dir, using first: $(subdirs[1])"

    holodeck_dir = joinpath(dir, subdirs[1], "HolodeckOutput")

    sync_files = filter(readdir(holodeck_dir)) do entry
        startswith(entry, "Synchronized$(id)")
    end
    isempty(sync_files) && error("No $id* file found in $holodeck_dir")

    path = joinpath(holodeck_dir, sync_files[1])
    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"

    df = CSV.read(
        path, DataFrame;
        header=false,
        # skipto=2,
        delim=(' '),
        ignorerepeated=true,   # treat multiple spaces as one delimiter
        missingstring=""
    )

    # Drop NaN rows
    df = dropmissing(df)

    # Strictly increasing timestamps
    mask = [true; df[2:end, 1] .> df[1:(end-1), 1]]
    df = df[mask, :]
    # Filter zero pos/rot
    pos_ok = vec(sum(Matrix(df[:, 3:5]) .^ 2, dims=2) .!= 0)
    rot_ok = vec(sum(Matrix(df[:, 6:14]) .^ 2, dims=2) .!= 0)
    df = df[pos_ok .& rot_ok, :]
    data = Matrix(df)
    N = size(data, 1)
    t = Vector{Float64}(data[:, 1] .* 1e-3)
    @show t[1]
    pos = MM_TO_M .* data[:, 3:5]'          # (3, N)
    R = permutedims(reshape(data[:, 6:14], N, 3, 3), (3, 2, 1))  # (3,3,N)

    # Find start time
    doc_dir = joinpath(dir, subdirs[1], "doc")
    doc_files = filter(readdir(doc_dir)) do entry
        startswith(entry, "doc_$id")
    end
    isempty(doc_files) && error("No doc_$id* file found in $doc_dir")
    length(doc_files) > 1 && @warn "Multiple matches for doc file id=$id in $doc_dir, using first: $(doc_files[1])"

    lines = readlines(joinpath(doc_dir, doc_files[1]))

    t_start = nothing
    for line in reverse(lines)
        m = match(r"TS\s*=\s*([+-]?[0-9]*\.?[0-9]+)", line)
        if !isnothing(m)
            t_start = parse(Float64, m.captures[1])
            break
        end
    end
    isnothing(t_start) && error("No 'TS = <float>' pattern found in $(doc_files[1])")

    mask = t .>= t_start

    return t[mask], pos[:, mask], R[:, :, mask], nothing
end

function read_raw_trajectory(::DCSC, dir::AbstractString, id::Int, kwargs...)
    subdirs = filter(readdir(dir)) do entry
        isdir(joinpath(dir, entry)) || return false
        s = string(id)
        startswith(entry, s) && (length(entry) == length(s) || !isdigit(entry[length(s)+1]))
    end
    isempty(subdirs) && error("No subdirectory starting with '$id' found in $dir")
    length(subdirs) > 1 && @warn "Multiple matches for id=$id in $dir, using first: $(subdirs[1])"

    dcsc_dir = joinpath(dir, subdirs[1], "OptiTrackOutput")

    sync_files = filter(readdir(dcsc_dir)) do entry
        endswith(entry, ".csv")
    end
    isempty(sync_files) && error("No $id* file found in $dcsc_dir")

    path = joinpath(dcsc_dir, sync_files[1])
    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"

    df = CSV.read(
        path, DataFrame;
        header=false,
        skipto=8,
    )

    # Drop NaN rows
    df = dropmissing(df)

    # Strictly increasing timestamps
    mask = [true; df[2:end, 2] .> df[1:(end-1), 2]]
    df = df[mask, :]

    # Filter zero pos/rot
    pos_ok = vec(sum(Matrix(df[:, 3:5]) .^ 2, dims=2) .!= 0)
    rot_ok = vec(sum(Matrix(df[:, 6:9]) .^ 2, dims=2) .!= 0)
    df = df[pos_ok .& rot_ok, :]
    data = Matrix(df[:, 2:end])
    N = size(data, 1)
    t = Vector{Float64}(data[:, 1])
    @show t[1]
    # quat = data[:, [5, 2, 3, 4]]'          # (4, N)
    C = [
        1.0 0.0 0.0
        0.0 0.0 -1.0
        0.0 1.0 0.0
    ]
    pos = data[:, [6, 7, 8]]'        # 3 , N [1, 3, 2]
    pos = C * pos
    quat = data[:, [5, 2, 3, 4]]'  # w, qx, qz, qy
    mats = quat_to_matrix(quat)
    for i in axes(mats, 3)
        mats[:, :, i] = C * mats[:, :, i]
    end

    return t, pos, mats, nothing
end

# Remove large Euler-angle jumps 
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

function temporal_alignment(tr::Trajectory, inertial_t::Vector{Float64})
    tg = tr.t
    pg = tr.pos
    Rg = tr.R_nb

    # Zero-order-hold boundary extension
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

    # Linear interpolation for position (3 × M)
    M = length(inertial_t)
    pos = Matrix{Float64}(undef, 3, M)
    for i in 1:3
        itp = Interpolations.linear_interpolation(tg, pg[i, :])
        pos[i, :] = itp.(inertial_t)
    end

    # Slerp for orientations — use Rotations.slerp directly on QuatRotation
    quats = [QuatRotation(RotMatrix{3}(Rg[:, :, i])) for i in axes(Rg, 3)]
    R_interp = Array{Float64}(undef, 3, 3, M)

    for (j, τ) in enumerate(inertial_t)
        k = clamp(searchsortedlast(tg, τ), 1, length(tg) - 1)
        a = clamp((τ - tg[k]) / (tg[k+1] - tg[k]), 0.0, 1.0)
        R_interp[:, :, j] .= Matrix(slerp(quats[k], quats[k+1], a))
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

"""
    rmse_yaw(tr::Trajectory, gt::Trajectory) -> Vector{Float64}

Cumulative RMS yaw error [rad], same running-mean convention as [`rmse`](@ref):
element `k` is the RMSE over samples `1:k`, so callers take `[end]` for the
window RMSE.

Exists because `rmse` scores horizontal position only (`pos[1:2,:]`). Any
experiment about the yaw channel — the yaw-only correction comparison in
`scripts/5Results/3_yaw_channel-*`, for instance — is otherwise measured
entirely in x/y and never on the quantity it claims to study.

Errors are wrapped to (-π, π] before squaring, so a trajectory that differs
from ground truth by a full turn scores 0 rather than 2π.
"""
function rmse_yaw(tr::Trajectory, gt::Trajectory)
    is_compatible(tr, gt) || throw(ArgumentError("incompatible time series"))
    n = size(tr.pos, 2)
    err_sq = Vector{Float64}(undef, n)
    for k in 1:n
        ψ_tr = matrix_to_euler(tr.R_nb[:, :, k])[3]
        ψ_gt = matrix_to_euler(gt.R_nb[:, :, k])[3]
        err_sq[k] = wrap_pi(ψ_tr - ψ_gt)^2
    end
    return sqrt.(cumsum(err_sq) ./ (1:n))
end

function total_distance(tr::Trajectory; dims=2)::Float64
    @assert (dims == 2 || dims == 3) "Can only compute distance in 2D or 3D got: $(dims)D"

    N = length(tr.t)
    if N < 2
        return 0.0
    end

    total = 0.0
    if dims == 2
        # Use only x (row 1) and y (row 2)
        for i in 1:(N-1)
            dx = tr.pos[1, i+1] - tr.pos[1, i]
            dy = tr.pos[2, i+1] - tr.pos[2, i]
            total += sqrt(dx * dx + dy * dy)
        end
    else # dims == 3
        for i in 1:(N-1)
            dx = tr.pos[1, i+1] - tr.pos[1, i]
            dy = tr.pos[2, i+1] - tr.pos[2, i]
            dz = tr.pos[3, i+1] - tr.pos[3, i]
            total += sqrt(dx * dx + dy * dy + dz * dz)
        end
    end
    return total
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
    eu = matrix_to_euler(tr.R_nb)[:, seg[1:(end-1)]]      # (3, N_steps)
    Δpos = tr.pos[:, seg[2:end]] .- tr.pos[:, seg[1:(end-1)]]
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

"""
    angular_velocity_from_rotations(t, R)

Computes angular rate using central difference.
"""
function angular_velocity_from_rotations(t::AbstractVector{T}, R::AbstractArray{T,3}) where T<:Real
    N = length(t)
    @assert size(R) == (3, 3, N)
    ω = zeros(3, N-2)
    for i in 2:(N-1)
        dt = t[i+1] - t[i-1]
        ΔR = R[:, :, i-1]' * R[:, :, i+1]   # body‑frame increment
        # Use the matrix logarithm for rotation vectors
        ω[:, i-1] = skew2vec(log(ΔR)) / dt   # rotation vector / dt
    end
    return ω
end

angular_velocity_from_rotations(trj::Trajectory) = angular_velocity_from_rotations(trj.t, trj.R_nb)

"""
    add_gaussian_noise(traj::Trajectory; pos_std=nothing, pos_bias=zeros(3),
                        att_std=nothing, att_bias=zeros(3))

Add Gaussian noise to a trajectory's position and/or orientation (Euler angles).
Both are optional; skipped if their `std` is `nothing`. `std`/`bias` args accept
a scalar (same value for all 3 dims) or a 3-vector.
"""
function add_gaussian_noise(
    traj::Trajectory;
    pos_std::Union{Nothing,Float64,AbstractVector{Float64}}=nothing,
    pos_bias::Union{AbstractVector{Float64},NTuple{3,Float64}}=zeros(3),
    att_std::Union{Nothing,Float64,AbstractVector{Float64}}=nothing,
    att_bias::Union{AbstractVector{Float64},NTuple{3,Float64}}=zeros(3),
    rng::Random.AbstractRNG=Random.Xoshiro(123)
)::Trajectory
    N = size(traj.pos, 2)

    # --- Position noise (optional) ---
    pos_noisy = traj.pos
    if !isnothing(pos_std)
        std_vec = pos_std isa Float64 ? fill(pos_std, 3) : collect(pos_std)
        bias_vec = collect(pos_bias)

        @assert length(std_vec) == 3 "pos_std must be a scalar or a 3-element vector, got length $(length(std_vec))"
        @assert length(bias_vec) == 3 "pos_bias must be a 3-element vector, got length $(length(bias_vec))"

        noise = randn(rng, 3, N) .* std_vec .+ bias_vec
        pos_noisy = traj.pos .+ noise
    end

    # --- Orientation noise (optional) ---
    R_nb_noisy = traj.R_nb
    if !isnothing(att_std)
        att_std_vec = att_std isa Float64 ? fill(att_std, 3) : collect(att_std)
        att_bias_vec = collect(att_bias)

        @assert length(att_std_vec) == 3 "att_std must be a scalar or a 3-element vector, got length $(length(att_std_vec))"
        @assert length(att_bias_vec) == 3 "att_bias must be a 3-element vector, got length $(length(att_bias_vec))"

        att_noise = randn(rng, 3, N) .* att_std_vec .+ att_bias_vec

        euler = matrix_to_euler(traj.R_nb)          # (3, N)
        euler_noisy = euler .+ att_noise
        R_nb_noisy = euler_to_matrix(euler_noisy)    # (3, 3, N)
    end

    return Trajectory(traj.t, pos_noisy, R_nb_noisy, traj.vel)
end