struct InertialData <: AbstractTimeSeries
    t::Vector{Float64}    # (N,)
    u::Matrix{Float64}    # (6, N)  [accel; gyro]

    function InertialData(t, u)
        N = length(t)
        all(diff(t) .> 0) || throw(ArgumentError("t must be strictly increasing"))
        size(u) == (6, N) || throw(DimensionMismatch("u must be (6, $N)"))
        new(t, u)
    end
end

Base.getindex(id::InertialData, mask) =
    InertialData(id.t[mask], id.u[:, mask])

function InertialData(dir::AbstractString, id::Int)
    src = resolve_source(dir)
    t, u = read_raw_imu(src, dir, id)
    return InertialData(t, u)
end

function InertialData(path::AbstractString)
    dir = dirname(path)
    src = resolve_source(dir)
    id_str = match(r"(\d+)_IMURaw", basename(path))
    id = isnothing(id_str) ? 0 : parse(Int, id_str.captures[1])
    t, u = read_raw_imu(src, dir, id)
    return InertialData(t, u)
end

function read_raw_imu(::ANG, dir::AbstractString, id::Int)
    path = joinpath(dir, "$(id)_IMURaw.csv")
    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"
    df = CSV.read(path, DataFrame; comment="//")
    drop = intersect(names(df), ["PacketCounter", "", "SystemTimestamp", "Column2"])
    select!(df, Not(drop))

    t = Vector{Float64}(df[:, 1])
    u = vcat(Matrix(df[:, 2:4])', Matrix(df[:, 5:7])')   # (6, N)
    return t, u
end

function read_raw_imu(::MTI, dir::AbstractString, id::Int)
    path = joinpath(dir, "$(id)_IMURaw.csv")
    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"
    df = CSV.read(path, DataFrame; comment="//")
    drop = intersect(names(df), ["PacketCounter", "", "SystemTimestamp", "Column2"])
    select!(df, Not(drop))

    t = Vector{Float64}(df[:, 1])
    u = vcat(Matrix(df[:, 2:4])', Matrix(df[:, 5:7])')   # (6, N)
    return t .* 1e-5, u
end

function read_raw_imu(::ANG2, dir::AbstractString, id::Int)
    # Find the subdirectory whose name starts with the given id
    subdirs = filter(readdir(dir)) do entry
        isdir(joinpath(dir, entry)) || return false
        s = string(id)
        startswith(entry, s) && (length(entry) == length(s) || !isdigit(entry[length(s)+1]))
    end

    isempty(subdirs) && error("No subdirectory starting with '$id' found in $dir")
    length(subdirs) > 1 && @warn "Multiple matches for id=$id in $dir, using first: $(subdirs[1])"
    path = joinpath(dir, subdirs[1], "IMURaw.txt")

    @info "From DIR($dir) ID($id) reading file :\n  → $(path)"
    # Read raw lines to inspect structure
    lines = readlines(path)

    # Parse header
    header_line = strip(lines[1])
    headers = split(header_line, r"\s+")  # split on whitespace

    # Count data columns from first data row
    data_line = strip(lines[2])
    n_data_cols = length(split(data_line, r"\s+"))

    # The data has one more column than named headers (unnamed second column)
    # Insert a placeholder name at position 2
    col_names = Vector{String}(vcat(
        [headers[1]],
        ["Column2"],           # unnamed second column
        headers[2:end]
    ))
    # Ensure we have the right number of names
    @assert length(col_names) == n_data_cols "Column count mismatch: $(length(col_names)) names vs $n_data_cols data columns"

    # Re-read with CSV.jl using our corrected headers
    df = CSV.read(
        path,
        DataFrame;
        header=col_names,
        skipto=2,              # skip the original header row
        delim=' ',
        ignorerepeated=true,   # treat multiple spaces as one delimiter
        missingstring=""
    )
    drop = intersect(names(df), ["PacketCounter", "", "SystemTimestamp", "Column2"])
    select!(df, Not(drop))

    t = Vector{Float64}(df[:, 1])
    u = vcat(Matrix(df[:, 2:4])', Matrix(df[:, 5:7])')   # (6, N)
    return t, u
end

