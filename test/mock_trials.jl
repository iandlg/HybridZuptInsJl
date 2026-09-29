# Mock trials: real ANG2 / DCSC recordings cut to their first `seconds`, written in
# the datasets' own on-disk layout so the real readers and preprocessing run on them.
# The copied files are only the ones `read_raw_imu`/`read_raw_trajectory` open.

"The trial directory of `id` in `dir`, by the readers' `<id>` + non-digit rule."
function trial_subdir(dir::AbstractString, id::Int)::String
    s = string(id)
    return only(filter(readdir(dir)) do e
        isdir(joinpath(dir, e)) && startswith(e, s) && (length(e) == length(s) || !isdigit(e[length(s)+1]))
    end)
end

"Copy `src` to `dst`, keeping the first `n_header` lines and the data lines `keep` accepts."
function copy_rows(keep::Function, src::AbstractString, dst::AbstractString; n_header::Int)
    mkpath(dirname(dst))
    lines = readlines(src)
    open(dst, "w") do io
        foreach(l -> println(io, l), lines[1:n_header])
        foreach(l -> !isempty(strip(l)) && keep(l) && println(io, l), lines[(n_header+1):end])
    end
end

field(line::AbstractString, i::Int) = parse(Float64, split(strip(line), r"\s+|,")[i])

"""
    write_mock_ang2(dst, id; seconds, margin=5.0)

ANG2 trial `id` from `TS` (its doc file) to `TS + seconds`: IMU time is column 3 [s],
Holodeck time column 1 [ms], both on the same clock.
"""
function write_mock_ang2(dst::AbstractString, id::Int; seconds::Real, margin::Real=5.0)
    src = DATA_DIRS["ANG2"]
    sub = trial_subdir(src, id)
    doc = only(filter(startswith("doc_$id"), readdir(joinpath(src, sub, "doc"))))
    ts = parse(Float64, match(r"TS\s*=\s*([+-]?[0-9]*\.?[0-9]+)",
        last(filter(contains(r"TS\s*="), readlines(joinpath(src, sub, "doc", doc))))).captures[1])

    mkpath(joinpath(dst, sub, "doc"))
    cp(joinpath(src, sub, "doc", doc), joinpath(dst, sub, "doc", doc))
    copy_rows(l -> field(l, 3) <= ts + seconds,
        joinpath(src, sub, "IMURaw.txt"), joinpath(dst, sub, "IMURaw.txt"); n_header=1)
    holo = only(filter(startswith("Synchronized$id"), readdir(joinpath(src, sub, "HolodeckOutput"))))
    copy_rows(l -> 1e-3 * field(l, 1) <= ts + seconds + margin,
        joinpath(src, sub, "HolodeckOutput", holo), joinpath(dst, sub, "HolodeckOutput", holo); n_header=0)
    return dst
end

"""
    write_mock_dcsc(dst, id; seconds, margin=5.0)

DCSC trial `id`, first `seconds` of IMU (100 Hz, time from the packet counter, which
does not start at 1 on every trial) and the OptiTrack rows that cover them. The
OptiTrack clock is `synchronize`'s `lag` off the IMU clock (-33 s on trial 1, -1 to
-3 s on trials 3-14), and `preprocess` needs mocap over the whole IMU window, so the
cut is taken on the full trial's lag.
"""
function write_mock_dcsc(dst::AbstractString, id::Int; seconds::Real, margin::Real=5.0)
    src = DATA_DIRS["DCSC"]
    sub = trial_subdir(src, id)
    imu = readlines(joinpath(src, sub, "IMURaw.txt"))
    # `//` comments, then a `PacketCounter,...` row on some trials only.
    n_header = findfirst(l -> isdigit(first(l)), imu) - 1
    t0 = (field(imu[n_header+1], 1) - 1) / 100
    copy_rows(l -> (field(l, 1) - 1) / 100 <= t0 + seconds,
        joinpath(src, sub, "IMURaw.txt"), joinpath(dst, sub, "IMURaw.txt"); n_header=n_header)
    _, _, lag = HybridZuptInsJl.synchronize(HybridZuptInsJl.InertialData(src, id), HybridZuptInsJl.Trajectory(src, id))
    opti = only(filter(endswith(".csv"), readdir(joinpath(src, sub, "OptiTrackOutput"))))
    copy_rows(l -> field(l, 2) <= t0 + seconds - lag + margin,
        joinpath(src, sub, "OptiTrackOutput", opti), joinpath(dst, sub, "OptiTrackOutput", opti); n_header=7)

    # Rewritten on every call: the rows of every trial mocked into `dst` so far.
    mocked(l) = isdir(joinpath(dst, trial_subdir(src, parse(Int, split(l, ",")[2]))))
    copy_rows(mocked, joinpath(src, "track-metadatas.csv"), joinpath(dst, "track-metadatas.csv"); n_header=1)
    return dst
end

"""
    mock_dataset(key, ids; seconds) -> String

A fresh directory holding mock trials `ids` of dataset `key`, registered with
`resolve_source` under the same source tag as the real dataset.
"""
function mock_dataset(key::AbstractString, ids::Vector{Int}; seconds::Real)::String
    dst = mktempdir()
    writer = Dict("ANG2" => write_mock_ang2, "DCSC" => write_mock_dcsc)[key]
    foreach(id -> writer(dst, id; seconds=seconds), ids)
    HybridZuptInsJl._DIR_TO_SOURCE[dst] = HybridZuptInsJl.resolve_source(DATA_DIRS[key])
    return dst
end
