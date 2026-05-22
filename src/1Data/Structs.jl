"""
    CorrectionOutput <: AbstractTimeSeries

Struct for storing correction outputs from GP, HSGP, or online correction methods.
Contains position and yaw corrections with optional standard deviations.

# Fields
- `t::Vector{Float64}`: Time stamps (strictly increasing)
- `output::Matrix{Float64}`: Correction outputs, size (4, n_steps) where:
  - rows 1-3: position corrections (x, y, z) in meters
  - row 4: yaw correction in radians
- `output_std::Union{Nothing, Matrix{Float64}}`: Optional standard deviations with same shape as output
"""
struct CorrectionOutput <: AbstractTimeSeries
    t::Vector{Float64}
    output::Matrix{Float64}           # 4 x n_steps: [pos1, pos2, pos3, yaw]
    output_std::Union{Nothing,Matrix{Float64}}

    function CorrectionOutput(
        t::Vector{Float64},
        output::Matrix{Float64},
        output_std::Union{Nothing,Matrix{Float64}}=nothing
    )
        length(size(t)) == 1 || throw(ArgumentError("t must be 1-D"))
        all(diff(t) .> 0) || throw(ArgumentError("t must be strictly increasing"))
        size(output, 1) == 4 || throw(ArgumentError("output must have 4 rows (pos1,pos2,pos3,yaw)"))
        length(t) == size(output, 2) || throw(ArgumentError("length(t) must match size(output,2)"))

        if !isnothing(output_std)
            size(output_std) == size(output) ||
                throw(ArgumentError("output_std must have same shape as output"))
        end

        new(t, output, output_std)
    end
end

function CorrectionOutput(d::Dict{String,Union{Vector{Vector{Float64}},Vector{Float64}}})
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

    return CorrectionOutput(t, output_mat, output_std_mat)
end

function Base.getindex(s::CorrectionOutput, mask::AbstractVector{Bool})
    CorrectionOutput(
        s.t[mask],
        s.output[:, mask],
        isnothing(s.output_std) ? nothing : s.output_std[:, mask]
    )
end

function Base.getindex(s::CorrectionOutput, idx::AbstractVector{<:Integer})
    CorrectionOutput(
        s.t[idx],
        s.output[:, idx],
        isnothing(s.output_std) ? nothing : s.output_std[:, idx]
    )
end

struct SeHyperparams
    yaw::Vector{Float64}    # [σ_f, length_scale, σ_n] for yaw
    pos_1::Vector{Float64}  # for x position
    pos_2::Vector{Float64}  # for y position
    pos_3::Vector{Float64}  # for z position

    # Inner constructor to enforce length 3
    function SeHyperparams(yaw, pos_0, pos_1, pos_2)
        for v in (yaw, pos_0, pos_1, pos_2)
            if length(v) != 3
                throw(ArgumentError("Each hyperparameter vector must have exactly 3 elements"))
            end
        end
        new(Vector{Float64}(yaw), Vector{Float64}(pos_0),
            Vector{Float64}(pos_1), Vector{Float64}(pos_2))
    end
end


function SeHyperparams(d::Dict{String,Union{Any,Vector{Any}}})
    SeHyperparams(
        Float64.(d["yaw"]),
        Float64.(d["pos_1"]),
        Float64.(d["pos_2"]),
        Float64.(d["pos_3"])
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

    function HsgpParameters(
        hp::SeHyperparams,
        d::Int,
        m::Int,
        LL::Union{Float64,Vector{Float64}};
        input_stats::Union{Nothing,Vector{Vector{Float64}}}=nothing,
        output_stats::Union{Nothing,Vector{Vector{Float64}}}=nothing
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

        new(hp, d, m, LL_vec, input_stats_, output_stats_)
    end
end

function HsgpParameters(d::Dict{String,Union{Any,Vector{Any}}})
    hp = SeHyperparams(Dict(d["hp"]))

    input_stats = haskey(d, "input_stats") && !isnothing(d["input_stats"]) ?
                  [Float64.(d["input_stats"][1]), Float64.(d["input_stats"][2])] : nothing

    output_stats = haskey(d, "output_stats") && !isnothing(d["output_stats"]) ?
                   [Float64.(d["output_stats"][1]), Float64.(d["output_stats"][2])] : nothing
    @show output_stats
    HsgpParameters(
        hp,
        d["d"],
        d["m"],
        Float64.(d["LL"]);
        input_stats=input_stats,
        output_stats=output_stats
    )
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

# function from_json(filename::AbstractString)
#     raw = JSON.parsefile(filename)
#     metadata = Dict(raw["metadata"])
#     saved_at = raw["saved_at"]
#     params = HsgpParameters(Dict(raw["params"]))
#     return params, metadata, saved_at
# end

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