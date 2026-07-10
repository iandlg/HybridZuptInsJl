"""
    CorrectionIO <: AbstractTimeSeries

Mutable struct for storing correction inputs and outputs from GP, HSGP, or online correction methods.

# Fields
- `t::Vector{Float64}`: Time stamps (strictly increasing)
- `data::Matrix{Float64}`: Correction IO, size (n_channels, n_steps) where:
- `data_std::Union{Nothing, Matrix{Float64}}`: Optional standard deviations with same shape as data
"""
mutable struct CorrectionIO <: AbstractTimeSeries
    t::Vector{Float64}
    data::Matrix{Float64}
    data_std::Union{Nothing,Matrix{Float64}}

    function CorrectionIO(
        t::Vector{Float64},
        data::Matrix{Float64},
        data_std::Union{Nothing,Matrix{Float64}}=nothing
    )
        length(size(t)) == 1 || throw(ArgumentError("t must be 1-D"))
        all(diff(t) .> 0) || throw(ArgumentError("t must be strictly increasing"))
        length(t) == size(data, 2) || throw(ArgumentError("length(t) must match size(output,2)"))

        if !isnothing(data_std)
            size(data_std) == size(data) ||
                throw(ArgumentError("output_std must have same shape as output"))
        end

        new(t, data, data_std)
    end
end

function CorrectionIO(n_channel::Int, std::Bool)
    CorrectionIO(
        Float64[],
        Matrix{Float64}(undef, n_channel, 0),
        std ? Matrix{Float64}(undef, n_channel, 0) : nothing
    )
end

function CorrectionIO(
    t::Vector{Float64},
    data::Vector{Float64},
    data_std::Union{Nothing,Vector{Float64}}=nothing)
    CorrectionIO(
        t,
        reshape(data, (1, length(data))),
        isnothing(data_std) ? nothing : reshape(data_std, (1, length(data_std)))
    )
end

function CorrectionIO(d::Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}})
    t = d["t"]
    output = d["output"]
    output_std = get(d, "output_std", nothing)

    # Handle flat vector case: treat as single time step or single output row
    output_mat = output isa Vector ? hcat(output...) : output
    output_std_mat = isnothing(output_std) ? nothing :
                     output_std isa Vector ? hcat(output_std...) : output_std

    if isempty(output_mat)
        return nothing
    end

    return CorrectionIO(t, output_mat, output_std_mat)
end

function Base.getindex(s::CorrectionIO, mask::AbstractVector{Bool})
    CorrectionIO(
        s.t[mask],
        s.data[:, mask],
        isnothing(s.data_std) ? nothing : s.data_std[:, mask]
    )
end

function Base.getindex(s::CorrectionIO, idx::AbstractVector{<:Integer})
    CorrectionIO(
        s.t[idx],
        s.data[:, idx],
        isnothing(s.data_std) ? nothing : s.data_std[:, idx]
    )
end

Base.size(s::CorrectionIO) = size(s.data)
Base.size(s::CorrectionIO, d::Int) = size(s.data, d)
Base.length(s::CorrectionIO) = size(s.data, 2)

function append_io!(s::CorrectionIO, t::Float64, data::Vector{Float64}, data_std::Union{Nothing,Vector{Float64}}=nothing)
    n_channel = size(s.data, 1)
    length(data) == n_channel || throw("Wrong number of data channels.")
    isempty(s.t) || s.t[end] < t || throw("Time series must be strictly increasing.")
    if !isnothing(data_std)
        length(data_std) == n_channel || throw("Wrong number of std channels")
        s.data_std = hcat(s.data_std, data_std)
    end
    push!(s.t, t)
    s.data = hcat(s.data, data)
end

"""
    concatenate_io(ios::Vector{CorrectionIO})::CorrectionIO

Combine multiple `CorrectionIO` objects into one.
Time stamps are replaced by a simple 1:total_steps index to guarantee
strictly increasing order after concatenation.
All inputs must have the same number of data channels and consistent
`data_std` presence (all must have it or all must lack it).
"""
function concatenate_io(ios::Vector{CorrectionIO})
    isempty(ios) && error("Cannot concatenate an empty vector of CorrectionIO")
    n_chan = size(first(ios).data, 1)
    total_steps = sum(io -> length(io.t), ios)
    t = collect(1.0:total_steps)               # Float64 to match existing

    # Check consistency
    has_std = !isnothing(first(ios).data_std)
    for io in ios
        size(io.data, 1) == n_chan || error("Channel count mismatch in concatenated IO")
        if has_std != !isnothing(io.data_std)
            error("Inconsistent data_std: all CorrectionIO must either have or lack data_std")
        end
    end

    all_data = reduce(hcat, [io.data for io in ios])
    all_std = has_std ? reduce(hcat, [io.data_std for io in ios]) : nothing

    return CorrectionIO(t, all_data, all_std)
end

struct SeHyperparams
    pos_1::Vector{Float64}  # for x position
    pos_2::Vector{Float64}  # for y position
    pos_3::Vector{Float64}  # for z position
    yaw::Vector{Float64}    # [σ_n, length_scale (can be multiple elements), σ_f] for yaw

    function SeHyperparams(pos_0, pos_1, pos_2, yaw)
        p0 = Vector{Float64}(pos_0)
        p1 = Vector{Float64}(pos_1)
        p2 = Vector{Float64}(pos_2)
        y = Vector{Float64}(yaw)

        all(>(0), p0) || throw(ArgumentError("All elements of pos_0 must be positive, got $p0"))
        all(>(0), p1) || throw(ArgumentError("All elements of pos_1 must be positive, got $p1"))
        all(>(0), p2) || throw(ArgumentError("All elements of pos_2 must be positive, got $p2"))
        all(>(0), y) || throw(ArgumentError("All elements of yaw must be positive, got $y"))

        new(p0, p1, p2, y)
    end
end

function SeHyperparams(d::Dict{String,Union{Any,Vector{Any}}})
    SeHyperparams(
        Float64.(d["pos_1"]),
        Float64.(d["pos_2"]),
        Float64.(d["pos_3"]),
        Float64.(d["yaw"]),
    )
end

function σ_n(hp::SeHyperparams)
    σ_n = Vector{Float64}(undef, 4)
    for (idx, field) in enumerate(fieldnames(SeHyperparams))
        σ_n[idx] = getfield(hp, field)[1]
    end
    return σ_n
end
# Helper: create a new SeHyperparams with one element changed
function modify_sehp(hp::SeHyperparams, group::Symbol, idx::Int, new_val::Float64)
    new_vec = copy(getfield(hp, group))
    new_vec[idx] = new_val
    return SeHyperparams(
        group == :pos_1 ? new_vec : hp.pos_1,
        group == :pos_2 ? new_vec : hp.pos_2,
        group == :pos_3 ? new_vec : hp.pos_3,
        group == :yaw ? new_vec : hp.yaw
    )
end

function to_json(filename::AbstractString, hp::SeHyperparams;
    metadata::Dict{String,Any}=Dict{String,Any}()
)
    envelope = OrderedDict{String,Any}(
        "saved_at" => string(now()),          # Dates.now()
        "metadata" => metadata,
        "params" => hp
    )
    open(filename, "w") do f
        JSON.print(f, envelope, 4)
    end
end

function to_json(
    filename::AbstractString, hp::Vector{SeHyperparams};
    metadata::Dict{String,Any}=Dict{String,Any}()
)
    envelope = OrderedDict{String,Any}(
        "saved_at" => string(now()),          # Dates.now()
        "metadata" => metadata,
        "params" => hp
    )
    open(filename, "w") do f
        JSON.print(f, envelope, 4)
    end
end

struct HsgpParameters
    hp::SeHyperparams
    d::Int
    m::Int
    LL::Vector{Float64}
    input_stats::Vector{Vector{Float64}}   # [mean(d,), std(d,)]
    output_stats::Vector{Vector{Float64}}  # [[mean_pos(3,), mean_yaw], [std_pos(3,), std_yaw]]
    mid_norm::Vector{Float64}

    function HsgpParameters(
        hp::SeHyperparams,
        d::Int,
        m::Int,
        LL::Union{Float64,Vector{Float64}};
        input_stats::Union{Nothing,Vector{Vector{Float64}}}=nothing,
        output_stats::Union{Nothing,Vector{Vector{Float64}}}=nothing,
        mid_norm::Optional{Vector{Float64}}=nothing
    )
        LL_vec = LL isa Float64 ? fill(LL, d) : copy(LL)
        length(LL_vec) == d ||
            throw(ArgumentError("length(LL) = $(length(LL_vec)) != d = $d"))
        any(LL_vec .<= 0) &&
            throw(ArgumentError("All LL must be positive, got $LL_vec"))

        _safe_std(v) = map(x -> abs(x) < eps(Float64) ? 1.0 : x, v)

        input_stats_ = isnothing(input_stats) ? [zeros(d), ones(d)] :
                       [input_stats[1], _safe_std(input_stats[2])]
        length(input_stats_[1]) == d && length(input_stats_[2]) == d ||
            throw(ArgumentError("input_stats vectors must have length d=$d"))

        if isnothing(output_stats)
            output_stats_ = [zeros(4), ones(4)]
        else
            mean_vec = convert(Vector{Float64}, output_stats[1])
            std_vec = convert(Vector{Float64}, output_stats[2])
            length(mean_vec) == 4 && length(std_vec) == 4 ||
                throw(ArgumentError("output_stats vectors must have length 4 (3 pos + 1 yaw)"))
            output_stats_ = [mean_vec, _safe_std(std_vec)]
        end

        mid_norm_ = isnothing(mid_norm) ? fill(0.0, d) : mid_norm

        new(hp, d, m, LL_vec, input_stats_, output_stats_, mid_norm_)
    end
end

function HsgpParameters(d::Dict{String,Union{Any,Vector{Any}}})
    hp = SeHyperparams(Dict(d["hp"]))

    input_stats = haskey(d, "input_stats") && !isnothing(d["input_stats"]) ?
                  [Float64.(d["input_stats"][1]), Float64.(d["input_stats"][2])] : nothing

    output_stats = haskey(d, "output_stats") && !isnothing(d["output_stats"]) ?
                   [Float64.(d["output_stats"][1]), Float64.(d["output_stats"][2])] : nothing

    mid_norm = haskey(d, "mid_norm") && !isnothing(d["mid_norm"]) ?
               Float64.(d["mid_norm"]) : nothing

    HsgpParameters(
        hp,
        d["d"],
        d["m"],
        Float64.(d["LL"]);
        input_stats=input_stats,
        output_stats=output_stats,
        mid_norm
    )
end

function basecopy(
    base::HsgpParameters;
    new_hp=nothing,
    new_input_stats=nothing,
    new_output_stats=nothing,
    new_LL=nothing,
    new_d=nothing,
    new_m=nothing,
    new_mid=nothing
)
    hp = isnothing(new_hp) ? base.hp : new_hp
    d = isnothing(new_d) ? base.d : new_d
    m = isnothing(new_m) ? base.m : new_m
    LL = isnothing(new_LL) ? base.LL : new_LL
    ins = isnothing(new_input_stats) ? base.input_stats : new_input_stats
    outs = isnothing(new_output_stats) ? base.output_stats : new_output_stats
    mid = isnothing(new_mid) ? base.mid_norm : new_mid
    return HsgpParameters(hp, d, m, LL; input_stats=ins, output_stats=outs, mid_norm=mid)
end

function to_json(
    filename::AbstractString, p::HsgpParameters;
    metadata::Dict{String,Any}=Dict{String,Any}()
)
    envelope = OrderedDict{String,Any}(
        "saved_at" => string(now()),          # Dates.now()
        "metadata" => metadata,
        "params" => p
    )
    open(filename, "w") do f
        JSON.print(f, envelope, 4)
    end
end

function from_json(::Type{T}, filename::AbstractString) where {T}
    raw = JSON.parsefile(filename)
    metadata = Dict(raw["metadata"])
    saved_at = raw["saved_at"]

    # Check if T is a Vector type
    if T <: AbstractVector
        # Extract element type and convert each element
        ElementType = eltype(T)
        obj = [ElementType(Dict(item)) for item in raw["params"]]
    else
        obj = T(Dict(raw["params"]))
    end
    return obj, metadata, saved_at
end

σ_n(p::HsgpParameters) = σ_n(p.hp)