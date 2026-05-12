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
    hp::SeHyperparams          # kernel hyperparameters (σ_f, ℓ, σ_n) for each output
    d::Int                     # input dimension
    m::Int                     # number of basis functions
    feature_std::Vector{Float64}   # per‑dimension standard deviation for scaling
    feature_mean::Vector{Float64}  # per‑dimension mean for scaling
    LL::Vector{Float64}            # domain half‑width per dimension (post‑scaling)

    function HsgpParameters(
        hp::SeHyperparams,
        d::Int,
        m::Int,
        feature_std::Vector{Float64},
        feature_mean::Vector{Float64},
        LL::Union{Float64,Vector{Float64}}
    )
        # dimension checks
        if length(feature_std) != d
            throw(ArgumentError("length(feature_std) = $(length(feature_std)) != d = $d"))
        end
        if length(feature_mean) != d
            throw(ArgumentError("length(feature_mean) = $(length(feature_mean)) != d = $d"))
        end

        # convert LL to vector of length d
        LL_vec = LL isa Float64 ? fill(LL, d) : copy(LL)
        if length(LL_vec) != d
            throw(ArgumentError("length(LL) = $(length(LL_vec)) != d = $d"))
        end

        # optional: check positivity of LL
        if any(LL_vec .<= 0)
            throw(ArgumentError("All domain half‑widths (LL) must be positive, got $LL_vec"))
        end

        new(hp, d, m, feature_std, feature_mean, LL_vec)
    end
end

function HsgpParameters(d::Dict{String,Any})
    hp = SeHyperparams(Dict(d["hp"]))
    HsgpParameters(
        hp,
        d["d"],
        d["m"],
        Float64.(d["feature_std"]),
        Float64.(d["feature_mean"]),
        Float64.(d["LL"])
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
    obj = T(Dict(raw["params"]))
    return obj, metadata, saved_at
end