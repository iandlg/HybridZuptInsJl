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
        yaw_hyp = isnothing(hyperparams_dict["yaw"]) ? ones(3) : hyperparams_dict["yaw"][fold, 2:4]
        pos_1_hyp = isnothing(hyperparams_dict["pos_1"]) ? ones(3) : hyperparams_dict["pos_1"][fold, 2:4]
        pos_2_hyp = isnothing(hyperparams_dict["pos_2"]) ? ones(3) : hyperparams_dict["pos_2"][fold, 2:4]
        pos_3_hyp = isnothing(hyperparams_dict["pos_3"]) ? ones(3) : hyperparams_dict["pos_3"][fold, 2:4]

        structs[fold] = SeHyperparams(yaw_hyp, pos_1_hyp, pos_2_hyp, pos_3_hyp)
    end

    return structs
end

"""
    read_hyperparameters_json(filepath::String) -> Vector{SeHyperparams}

Read hyperparameters from a JSON file and return a vector of SeHyperparams structs.

The JSON file should contain an array of objects, where each object represents
hyperparameters for a fold and has keys: "yaw", "pos_1", "pos_2", "pos_3".
Each key maps to an array of 3 numbers [σ_f, ℓ, σ_n].

# Arguments
- `filepath`: Path to the JSON file.

# Returns
- `Vector{SeHyperparams}`: A vector where each element is a SeHyperparams struct for one fold.

# Example
```julia
hypers = read_hyperparameters_json("out/hyperparameters/py_hypers.json")
```
"""
function read_hyperparameters_json(filepath::String)
    data = JSON.parsefile(filepath, Vector{SeHyperparams})
    return data
end
