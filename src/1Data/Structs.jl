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
    input_stats::Vector{Vector{Float64}}   # [mean(d,), std(d,)]
    yaw_stats::Vector{Float64}             # [mean, std]
    pos_stats::Vector{Vector{Float64}}     # [mean(3,), std(3,)]
    LL::Vector{Float64}

    function HsgpParameters(
        hp::SeHyperparams,
        d::Int,
        m::Int,
        LL::Union{Float64,Vector{Float64}};
        input_stats::Union{Nothing,Vector{Vector{Float64}}}=nothing,
        yaw_stats::Union{Nothing,Vector{Float64}}=nothing,
        pos_stats::Union{Nothing,Vector{Vector{Float64}}}=nothing
    )
        LL_vec = LL isa Float64 ? fill(LL, d) : copy(LL)
        length(LL_vec) == d ||
            throw(ArgumentError("length(LL) = $(length(LL_vec)) != d = $d"))
        any(LL_vec .<= 0) &&
            throw(ArgumentError("All LL must be positive, got $LL_vec"))

        _safe_std(v) = map(x -> abs(x) < eps(Float64) ? 1.0 : x, v)
        _safe_std_scalar(x) = abs(x) < eps(Float64) ? 1.0 : x

        input_stats_ = isnothing(input_stats) ? [zeros(d), ones(d)] :
                       [input_stats[1], _safe_std(input_stats[2])]
        length(input_stats_[1]) == d && length(input_stats_[2]) == d ||
            throw(ArgumentError("input_stats vectors must have length d=$d"))

        yaw_stats_ = isnothing(yaw_stats) ? [0.0, 1.0] :
                     [yaw_stats[1], _safe_std_scalar(yaw_stats[2])]
        length(yaw_stats_) == 2 ||
            throw(ArgumentError("yaw_stats must have 2 elements [mean, std]"))

        pos_stats_ = isnothing(pos_stats) ? [zeros(3), ones(3)] :
                     [pos_stats[1], _safe_std(pos_stats[2])]
        length(pos_stats_[1]) == 3 && length(pos_stats_[2]) == 3 ||
            throw(ArgumentError("pos_stats vectors must have length 3"))

        new(hp, d, m, input_stats_, yaw_stats_, pos_stats_, LL_vec)
    end
end

function HsgpParameters(d::Dict{String,Any})
    hp = SeHyperparams(Dict(d["hp"]))

    input_stats = haskey(d, "input_stats") && !isnothing(d["input_stats"]) ?
                  [Float64.(d["input_stats"][1]), Float64.(d["input_stats"][2])] : nothing

    yaw_stats = haskey(d, "yaw_stats") && !isnothing(d["yaw_stats"]) ?
                Float64.(d["yaw_stats"]) : nothing

    pos_stats = haskey(d, "pos_stats") && !isnothing(d["pos_stats"]) ?
                [Float64.(d["pos_stats"][1]), Float64.(d["pos_stats"][2])] : nothing

    HsgpParameters(
        hp,
        d["d"],
        d["m"],
        Float64.(d["LL"]);
        input_stats=input_stats,
        yaw_stats=yaw_stats,
        pos_stats=pos_stats
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