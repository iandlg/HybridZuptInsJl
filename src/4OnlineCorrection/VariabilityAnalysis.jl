struct ParamSpec
    name::String
    get_current::Function   # (HsgpParameters) -> Float64
    set_new::Function       # (HsgpParameters, Float64) -> HsgpParameters
    value_generator::Function  # (current_value) -> Vector{Float64}
end

# Convenience constructor using default log_around
function ParamSpec(name, get_current, set_new; log_range=(-2.0, 2.0), n_steps=9)
    generator(val) = log_around(val, log_range, n_steps)
    ParamSpec(name, get_current, set_new, generator)
end

# 
mutable struct ParamGrid
    group_names::Vector{String}           # e.g., ["pos_1", "pos_2", "input_mean"]
    max_idx::Int                          # maximum number of parameters in any group
    specs::Matrix{Union{Nothing,ParamSpec}} # rows = groups, cols = 1:max_idx
end

# Helper to create an empty grid from a list of groups and their max index per group
function ParamGrid(groups::Vector{String}, max_indices::Vector{Int})
    max_idx = maximum(max_indices)
    n_groups = length(groups)
    specs = Matrix{Union{Nothing,ParamSpec}}(undef, n_groups, max_idx)
    ParamGrid(groups, max_idx, specs)
end

# Place a spec at a specific (group_idx, param_idx) in the grid
function place_spec!(grid::ParamGrid, group_idx::Int, param_idx::Int, spec::ParamSpec)
    grid.specs[group_idx, param_idx] = spec
    return grid
end

function grid_to_dict(grid::ParamGrid)::AbstractDict
    # Convert grid to serializable format
    grid_dict = Dict(
        "group_names" => grid.group_names,
        "max_idx" => grid.max_idx,
        "specs" => map(x -> isnothing(x) ? nothing : x.name, grid.specs)  # store only names
    )
    return grid_dict
end

function grid_from_dict(data::AbstractDict)::ParamGrid
    group_names = data["group_names"]
    max_idx = data["max_idx"]
    spec_names = data["specs"]  # matrix of strings or nothing
    # Rebuild grid without the actual ParamSpec objects (only names)
    # For plotting we only need the names and layout, not the getters/setters.
    specs = Matrix{Union{Nothing,ParamSpec}}(undef, length(group_names), max_idx)
    for i in 1:size(specs, 1), j in 1:size(specs, 2)
        if !isnothing(spec_names[i][j])
            # We can create a dummy spec with just name; plotting only uses name to match DataFrame.
            specs[i, j] = ParamSpec(spec_names[i][j], p -> 0.0, (p, v) -> p, (v) -> [v])
        end
    end
    return ParamGrid(group_names, max_idx, specs)
end

function make_rmse_evaluator(
    data_dir::String,
    trial_id::Int,
    train_ratio::Float64,
    feature_type::FeatureType,
    ref_frame::ReferenceFrame,
    m::Union{Nothing,Int}=nothing
)::Function

    # 1. Align INS and GT (same for all hyperparameter variations)
    ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated =
        compute_aligned_ins_trajectory(data_dir, trial_id)

    # 2. Initial state from aligned trajectory
    x_init = vcat(
        ins_traj_aligned.pos[:, 1],
        ins_traj_aligned.vel[:, 1],
        matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
    )

    # 3. Return the evaluator closure
    function rmse_evaluator(hsgp_params::HsgpParameters)::Float64
        # Optionally force m to be consistent (if needed)
        hsgp_params = HsgpParameters(
            hsgp_params.hp, hsgp_params.d, isnothing(m) ? hsgp_params.m : m, hsgp_params.LL;
            input_stats=hsgp_params.input_stats,
            output_stats=hsgp_params.output_stats
        )

        # Run online correction
        _, hsgp_ins_traj, segs, _, _, _, _ =
            hybrid_zupt_aided_ins(
                inertial_updated, sim_config_updated, gt_traj_aligned, hsgp_params;
                x_init=x_init,
                train_ratio=train_ratio,
                feature_type=feature_type,
                ref_frame=ref_frame
            )

        return rmse(hsgp_ins_traj[segs], gt_traj_aligned[segs])[end] # final RMSE
    end

    return rmse_evaluator
end

function make_hp_param_grid(base_hp::SeHyperparams, groups::Vector{Symbol};
    log_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    n_steps::Int=9)::Tuple{Vector{ParamSpec},ParamGrid}

    group_names = [string(g) for g in groups]
    max_indices = [length(getfield(base_hp, g)) for g in groups]
    grid = ParamGrid(group_names, max_indices)
    specs = ParamSpec[]

    for (grp_idx, group) in enumerate(groups)
        vec = getfield(base_hp, group)
        for (idx, base_val) in enumerate(vec)
            name = "$group[$idx]"
            getter(p) = getfield(p.hp, group)[idx]
            setter(p, new_val) = basecopy(p; new_hp=modify_sehp(p.hp, group, idx, new_val))
            spec = ParamSpec(name, getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, grp_idx, idx, spec)
        end
    end
    return specs, grid
end

# Helper: modify input_stats (mean or std)
function set_input_stat(p::HsgpParameters, which_stat::Int, dim::Int, new_val::Float64)
    new_vec = copy(p.input_stats[which_stat])
    new_vec[dim] = new_val
    if which_stat == 1
        return basecopy(p; new_input_stats=[new_vec, p.input_stats[2]])
    else
        return basecopy(p; new_input_stats=[p.input_stats[1], new_vec])
    end
end

# Helper: modify output_stats (mean or std)
function set_output_stat(p::HsgpParameters, which_stat::Int, dim::Int, new_val::Float64)
    new_vec = copy(p.output_stats[which_stat])
    new_vec[dim] = new_val
    if which_stat == 1
        return basecopy(p; new_output_stats=[new_vec, p.output_stats[2]])
    else
        return basecopy(p; new_output_stats=[p.output_stats[1], new_vec])
    end
end

function make_stats_param_grid(base_params::HsgpParameters;
    log_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    n_steps::Int=9)::Tuple{Vector{ParamSpec},ParamGrid}

    d = base_params.d
    p = length(base_params.output_stats[1])
    group_names = ["input_mean", "input_std", "output_mean", "output_std"]
    max_indices = [d, d, p, p]
    grid = ParamGrid(group_names, max_indices)
    specs = ParamSpec[]

    # Input means (group_idx=1)
    for dim in 1:d
        name = "input_mean[$dim]"
        getter(p) = p.input_stats[1][dim]
        setter(p, val) = set_input_stat(p, 1, dim, val)
        spec = ParamSpec(name, getter, setter; log_range=log_range, n_steps=n_steps)
        push!(specs, spec)
        place_spec!(grid, 1, dim, spec)
    end

    # Input stds (group_idx=2)
    for dim in 1:d
        name = "input_std[$dim]"
        getter(p) = p.input_stats[2][dim]
        setter(p, val) = set_input_stat(p, 2, dim, val)
        spec = ParamSpec(name, getter, setter; log_range=log_range, n_steps=n_steps)
        push!(specs, spec)
        place_spec!(grid, 2, dim, spec)
    end

    # Output means (group_idx=3)
    for dim in 1:p
        name = "output_mean[$dim]"
        getter(p) = p.output_stats[1][dim]
        setter(p, val) = set_output_stat(p, 1, dim, val)
        spec = ParamSpec(name, getter, setter; log_range=log_range, n_steps=n_steps)
        push!(specs, spec)
        place_spec!(grid, 3, dim, spec)
    end

    # Output stds (group_idx=4)
    for dim in 1:p
        name = "output_std[$dim]"
        getter(p) = p.output_stats[2][dim]
        setter(p, val) = set_output_stat(p, 2, dim, val)
        spec = ParamSpec(name, getter, setter; log_range=log_range, n_steps=n_steps)
        push!(specs, spec)
        place_spec!(grid, 4, dim, spec)
    end

    return specs, grid
end

function vary_hsgp_hyperparameters(
    base_params::HsgpParameters,
    rmse_func::Function,
    param_specs::Vector{ParamSpec};
    include_baseline::Bool=true
)::DataFrame
    results = []
    baseline_rmse = rmse_func(base_params)
    @info "Baseline RMSE: $baseline_rmse"

    for spec in param_specs
        current_val = spec.get_current(base_params)
        test_vals = spec.value_generator(current_val)
        for new_val in test_vals
            new_params = spec.set_new(base_params, new_val)
            rmse_val = rmse_func(new_params)
            push!(results, (
                parameter=spec.name,
                base_value=current_val,
                tested_value=new_val,
                rmse=rmse_val,
                rmse_ratio=rmse_val / baseline_rmse
            ))
        end
    end

    if include_baseline
        push!(results, (
            parameter="baseline",
            base_value=NaN,
            tested_value=NaN,
            rmse=baseline_rmse,
            rmse_ratio=1.0
        ))
    end
    return DataFrame(results)
end