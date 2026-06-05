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

function Trajectory(dir::AbstractString, id::Int)
    src = resolve_source(dir)
    t, pos, R_nb, vel = read_raw_trajectory(src, dir, id)
    return clean(Trajectory(t, pos, R_nb, vel))
end

function read_raw_trajectory(::ANG, dir::AbstractString, id::Int)
    path = joinpath(dir, "$(id)_Synchronized_Reference.csv")
    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"

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
    return t, pos, R, nothing
end

function read_raw_trajectory(::MTI, dir::AbstractString, id::Int)
    error("Not implemented")
end

function read_raw_trajectory(::ANG2, dir::AbstractString, id::Int)
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
        delim=' ',
        ignorerepeated=true,   # treat multiple spaces as one delimiter
        missingstring=""
    )

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