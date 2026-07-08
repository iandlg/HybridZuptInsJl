# Source tag (one per file format / data origin)
abstract type AbstractDataSource end
struct ANG <: AbstractDataSource end
struct MTI <: AbstractDataSource end
struct ANG2 <: AbstractDataSource end
struct DCSC <: AbstractDataSource end

# Resolve a directory path to its source tag
const _DIR_TO_SOURCE = Dict{String,AbstractDataSource}(
    "data/angermann_high_precision" => ANG(),
    "data/mti-100-recordings" => MTI(),
    "data/angermann_v2" => ANG2(),
    "data/dcsc_optitrack" => DCSC()
)

function resolve_source(dir::AbstractString)::AbstractDataSource
    src = get(_DIR_TO_SOURCE, dir, nothing)
    isnothing(src) && error("Unknown data directory: $dir\nKnown dirs: $(keys(_DIR_TO_SOURCE))")
    return src
end


"""
    list_trial_ids(data_dir::AbstractString) -> Vector{Int}

Scan a data directory and return all integer trial IDs found.
Dispatches to a source‑specific method.
"""
function list_trial_ids(data_dir::AbstractString)::Vector{Int}
    src = resolve_source(data_dir)
    return list_trial_ids(src, data_dir)
end

"""
    list_trial_ids(::ANG2, data_dir::AbstractString)::Vector{Int}

Scans the "data/angermann_v2" directory and return all integer trial IDs found.
"""
function list_trial_ids(::ANG2, data_dir::AbstractString)::Vector{Int}
    ids = Int[]
    for entry in readdir(data_dir)
        entry_path = joinpath(data_dir, entry)
        if isdir(entry_path)
            m = match(r"^(\d+)\D", entry)   # capture leading digits before a non-digit
            !isnothing(m) && push!(ids, parse(Int, m.captures[1]))
        end
    end
    return sort(unique(ids))
end

"""
    list_trial_ids(::ANG, data_dir::AbstractString)::Vector{Int}

Scans the "data/angermann_high_precision" directory and return all integer trial IDs found.
"""
function list_trial_ids(::ANG, data_dir::AbstractString)::Vector{Int}
    ids = Int[]
    for entry in readdir(data_dir)
        m = match(r"^(\d+)_IMURaw", entry)
        !isnothing(m) && push!(ids, parse(Int, m.captures[1]))
    end
    return sort(unique(ids))
end

"""
    _convert_hyperparams_to_struct_list(hyperparams_dict::Dict{String,Union{Matrix{Float64},Nothing}}) -> Union{Vector{SeHyperparams},Nothing}

Convert hyperparameters from dimension-organized dictionary to a list of SeHyperparams structs per fold.
Each fold gets a SeHyperparams struct with hyperparameters [σ_f, ℓ, σ_n] for each dimension.

If any dimension has hyperparams, returns a vector of SeHyperparams structs.
Otherwise returns nothing.
"""
function _convert_hyperparams_to_struct_list(
    hyperparams_dict::Dict{String,Union{Matrix{Float64},Nothing}}
)::Union{Nothing,Vector{SeHyperparams}}
    # Check if we have any hyperparams
    has_any = any(!isnothing(v) for v in values(hyperparams_dict))
    if !has_any
        return nothing
    end

    # Determine number of folds from available matrices
    n_folds = nothing
    for v in values(hyperparams_dict)
        if !isnothing(v)
            n_folds = size(v, 1)
            break
        end
    end

    if isnothing(n_folds)
        return nothing
    end

    # Build SeHyperparams struct for each fold
    structs = Vector{SeHyperparams}(undef, n_folds)
    for fold in 1:n_folds
        yaw_hyp = isnothing(hyperparams_dict["yaw"]) ? ones(3) : hyperparams_dict["yaw"][fold, 2:end]
        pos_1_hyp = isnothing(hyperparams_dict["pos_1"]) ? ones(3) : hyperparams_dict["pos_1"][fold, 2:end]
        pos_2_hyp = isnothing(hyperparams_dict["pos_2"]) ? ones(3) : hyperparams_dict["pos_2"][fold, 2:end]
        pos_3_hyp = isnothing(hyperparams_dict["pos_3"]) ? ones(3) : hyperparams_dict["pos_3"][fold, 2:end]

        structs[fold] = SeHyperparams(pos_1_hyp, pos_2_hyp, pos_3_hyp, yaw_hyp)
    end

    return structs
end

function load_hp_variation_results(csv_path::String, json_path::String)::Tuple{DataFrame,Dict}
    df = CSV.read(csv_path, DataFrame)
    metadata = JSON.parsefile(json_path)
    return df, metadata
end