abstract type AbstractTimeSeries end

struct TimeSeries <: AbstractTimeSeries
    t::Vector{Float64}   # strictly increasing timestamps (s)

    function TimeSeries(t)
        length(size(t)) == 1 || throw(ArgumentError("t must be 1-D"))
        all(diff(t) .> 0) || throw(ArgumentError("t must be strictly increasing"))
        new(t)
    end
end

Base.length(ts::AbstractTimeSeries) = length(ts.t)

# ── Truncate several series to their overlapping interval ───────────
function truncate_to_overlap(series::AbstractTimeSeries...)
    length(series) >= 2 || throw(ArgumentError("need ≥ 2 series"))

    t_start = maximum(s.t[begin] for s in series)
    t_end = minimum(s.t[end] for s in series)
    # @info "Start time $t_start s"
    # @info "End time $t_end s"
    t_start < t_end || throw(ArgumentError("no overlapping interval"))
    return Tuple(getindex(s, (s.t .>= t_start) .& (s.t .<= t_end)) for s in series)
end

function is_compatible(series::AbstractTimeSeries...)
    length(series) >= 2 || throw(ArgumentError("need ≥ 2 series"))
    base = series[begin]
    return all(size(s.t) == size(base.t) && maximum(abs.(s.t .- base.t)) <= 1e-9
               for s in series[begin+1:end])
end
