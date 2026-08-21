struct ParamSpec
    name::String
    type::String
    get_current::Function   # (HsgpParameters) -> Float64
    set_new::Function       # (HsgpParameters, Float64) -> HsgpParameters
    value_generator::Function  # (current_value) -> Vector{Float64}
end

const _HYPERPARAM_TYPES = Dict{Int,String}(
    1 => "noise",
    2 => "length_scale",
    3 => "signal_variance"
)

# Convenience constructor using default log_around
function ParamSpec(name, type, get_current, set_new; log_range=(-2.0, 2.0), n_steps=9)
    generator(val) = log_around(val, log_range, n_steps)
    ParamSpec(name, type, get_current, set_new, generator)
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
    specs = Matrix{Union{Nothing,ParamSpec}}(nothing, n_groups, max_idx)
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
    specs = Matrix{Union{Nothing,ParamSpec}}(nothing, length(group_names), max_idx)
    for i in 1:size(specs, 1), j in 1:size(specs, 2)
        if !isnothing(spec_names[j][i])
            # We can create a dummy spec with just name; plotting only uses name to match DataFrame.
            specs[i, j] = ParamSpec(spec_names[j][i], "none", p -> 0.0, (p, v) -> p, (v) -> [v])
        end
    end
    return ParamGrid(group_names, max_idx, specs)
end


function make_rmse_evaluator(
    data_dir::String,
    trial_id::Int,
    train_ratio::Float64,
    feature_type::FeatureType,
    ref_frame::ReferenceFrame, ;
    m::Union{Nothing,Int}=nothing,
    output_channel_idxs=[1, 2, 3, 4],
    hsgp_estimator_factory::Type=JointHsgpEstimator,
    noise_spec::NoiseSpec=NoiseSpec(),
    pred_includes_noise::Bool=false,
    eval_test_half_only::Bool=true
)::Function
    p = length(output_channel_idxs)
    @assert p <= 4 && p>=1 "Wrong number of output channels, got $p"
    @assert all(diff(output_channel_idxs) .> 0) "Output channels must be in ascending order, got $output_channel_idxs"

    # 1. Align INS and GT (same for all hyperparameter variations)
    ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated =
        compute_aligned_ins_trajectory(data_dir, trial_id)

    # 2. Initial state from aligned trajectory
    x_init = vcat(
        ins_traj_aligned.pos[:, 1],
        ins_traj_aligned.vel[:, 1],
        matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
    )

    # 3. Training phase 
    N = length(inertial_updated)
    n_train_cutoff = floor(Int, train_ratio * N)
    gt_available = [n <= n_train_cutoff for n in 1:N]

    # 4. Add Gaussian noise to ground truth 
    gt_noisy = add_gaussian_noise(gt_traj_aligned;
        pos_std=noise_spec.pos_std, pos_bias=noise_spec.pos_bias,
        att_std=noise_spec.att_std, att_bias=noise_spec.att_bias
    )

    # 5. Return the evaluator closure
    function rmse_evaluator(hsgp_params::HsgpParameters)::Float64
        # Optionally force m to be consistent (if needed)
        hsgp_params = HsgpParameters(
            hsgp_params.hp, hsgp_params.d, isnothing(m) ? hsgp_params.m : m, hsgp_params.LL;
            input_stats=hsgp_params.input_stats,
            output_stats=hsgp_params.output_stats
        )
        hsgp_estimator = hsgp_estimator_factory(300; params=hsgp_params,
            corrected_channels=[Symbol(_OUTPUT_NAMES[val]) for val in output_channel_idxs],
            pred_includes_noise=pred_includes_noise)
        _, step_seg, slamHsgp_corr_traj, _ = hybrid_zupt_aided_insv2(
            inertial_updated, sim_config_updated, gt_noisy, hsgp_estimator;
            x_init=x_init, gt_available=gt_available, ref_frame=ref_frame, feature_type=feature_type)

        # Score the *test* portion only, matching run_online_correction_sweep
        # (DataProcessing.jl) and training_data_quality_analysis. Previously this
        # scored the whole trajectory, folding the GT-supervised training half
        # into the metric and making the sensitivity numbers incomparable with
        # every other figure in 5Results. Pass eval_test_half_only=false to
        # reproduce results generated before 2026-08-21.
        gt_step = gt_traj_aligned[step_seg]
        if eval_test_half_only
            k0 = max(1, floor(Int, train_ratio * length(slamHsgp_corr_traj)))
            return rmse(slamHsgp_corr_traj[k0:end], gt_step[k0:end])[end]
        end
        return rmse(slamHsgp_corr_traj, gt_step)[end] # final RMSE
    end

    return rmse_evaluator
end

function make_hp_param_grid(
    base_hp::SeHyperparams, output_channel_idxs::Vector{Int};
    log_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    n_steps::Int=9
)::Tuple{Vector{ParamSpec},ParamGrid}

    channel_names = [_OUTPUT_NAMES[idx] for idx in output_channel_idxs]
    max_indices = [length(getfield(base_hp, Symbol(name))) for name in channel_names]
    grid = ParamGrid(channel_names, max_indices)
    specs = ParamSpec[]

    for (grp_idx, chan_name) in enumerate(channel_names)
        vec = getfield(base_hp, Symbol(chan_name))
        for (idx, base_val) in enumerate(vec)
            name = "$(Symbol(chan_name))[$idx]"
            getter(p) = getfield(p.hp, Symbol(chan_name))[idx]
            setter(p, new_val) = basecopy(p; new_hp=modify_sehp(p.hp, Symbol(chan_name), idx, new_val))
            spec = ParamSpec(name, _HYPERPARAM_TYPES[idx], getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, grp_idx, idx, spec)
        end
    end
    return specs, grid
end

function set_input_stat(p::HsgpParameters, which_stat::Int, dim::Int, new_val::Float64)
    new_vec = copy(p.input_stats[which_stat])
    new_vec[dim] = new_val
    if which_stat == 1
        return basecopy(p; new_input_stats=[new_vec, p.input_stats[2]])
    else
        return basecopy(p; new_input_stats=[p.input_stats[1], new_vec])
    end
end

function set_output_stat(p::HsgpParameters, which_stat::Int, dim::Int, new_val::Float64)
    new_vec = copy(p.output_stats[which_stat])
    new_vec[dim] = new_val
    if which_stat == 1
        return basecopy(p; new_output_stats=[new_vec, p.output_stats[2]])
    else
        return basecopy(p; new_output_stats=[p.output_stats[1], new_vec])
    end
end

function set_mid_norm_stat(p::HsgpParameters, dim::Int, new_val::Float64)
    new_vec = copy(p.mid_norm)
    new_vec[dim] = new_val
    return basecopy(p; new_mid=new_vec)
end

function make_stats_param_grid(base_params::HsgpParameters;
    output_channel_idxs::Vector{Int}=[1, 2, 3, 4],
    log_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    n_steps::Int=9, include_input_params::Bool=true, include_output_params::Bool=true
)::Tuple{Vector{ParamSpec},ParamGrid}
    max_indices = Int[]
    group_names = String[]
    d = base_params.d
    p = length(output_channel_idxs)

    if include_input_params
        append!(max_indices, [d, d, d])
        append!(group_names, ["input_mean", "input_std", "input_center"])
    end


    if include_output_params
        @assert p <= 4 && p>=1 "Wrong number of output channels, got $p"
        @assert all(diff(output_channel_idxs) .> 0) "Output channels must be in ascending order, got $output_channel_idxs"
        append!(max_indices, [p, p])
        append!(group_names, ["output_mean", "output_std"])
    end

    grid = ParamGrid(group_names, max_indices)
    specs = ParamSpec[]

    if include_input_params
        # Input means (group_idx=1)
        for dim in 1:d
            name = "input_mean[$dim]"
            getter(p) = p.input_stats[1][dim]
            setter(p, val) = set_input_stat(p, 1, dim, val)
            spec = ParamSpec(name, "input_mean", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, 1, dim, spec)
        end

        # Input stds (group_idx=2)
        for dim in 1:d
            name = "input_std[$dim]"
            getter(p) = p.input_stats[2][dim]
            setter(p, val) = set_input_stat(p, 2, dim, val)
            spec = ParamSpec(name, "input_std", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, 2, dim, spec)
        end

        # Input Center
        for dim in 1:d
            name = "input_center[$dim]"
            getter(p) = p.mid_norm[dim]
            setter(p, val) = set_mid_norm_stat(p, dim, val)
            spec = ParamSpec(name, "mid_norm", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, 3, dim, spec)
        end
    end

    if include_output_params
        # Output means (group_idx=3)
        for (i, dim) in enumerate(output_channel_idxs)
            name = "output_mean[$dim]"
            getter(p) = p.output_stats[1][dim]
            setter(p, val) = set_output_stat(p, 1, dim, val)
            spec = ParamSpec(name, "output_mean", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, 4, i, spec)
        end

        # Output stds (group_idx=4)
        for (i, dim) in enumerate(output_channel_idxs)
            name = "output_std[$dim]"
            getter(p) = p.output_stats[2][dim]
            setter(p, val) = set_output_stat(p, 2, dim, val)
            spec = ParamSpec(name, "output_std", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, 5, i, spec)
        end
    end

    return specs, grid
end

function vary_hsgp_parameters(
    base_params::HsgpParameters,
    rmse_func::Function,
    param_specs::Vector{ParamSpec};
    include_baseline::Bool=true
)::DataFrame
    results = []
    baseline_rmse = rmse_func(base_params)
    @info "Baseline RMSE: $baseline_rmse"

    inert = String[]
    for spec in param_specs
        current_val = spec.get_current(base_params)
        test_vals = spec.value_generator(current_val)
        spec_rmses = Float64[]
        for new_val in test_vals
            new_params = spec.set_new(base_params, new_val)
            rmse_val = rmse_func(new_params)
            push!(spec_rmses, rmse_val)
            push!(results, (
                parameter=spec.name,
                type=spec.type,
                base_value=current_val,
                tested_value=new_val,
                rmse=rmse_val,
                rmse_ratio=rmse_val / baseline_rmse,
                relative_change=(rmse_val - baseline_rmse) / baseline_rmse
            ))
        end

        # A parameter whose sweep does not move the metric *at all* is almost
        # never a physical insensitivity result: it means the value never
        # reached the code under evaluation. Reporting it as a flat line in the
        # sensitivity figure is actively misleading, so say so loudly.
        # (This is exactly what happened to the `noise` hyperparameters: σ_n is
        # loaded into DecoupledHsgpEstimator and then never read unless
        # `pred_includes_noise=true`. See notes/004.)
        if length(spec_rmses) > 1 && length(unique(spec_rmses)) == 1
            push!(inert, spec.name)
        end
    end

    if !isempty(inert)
        @warn """
        Sensitivity sweep produced a bit-identical metric for $(length(inert)) parameter(s): \
        $(join(inert, ", ")). These parameters do not reach the evaluated code path, so their \
        rows are NOT evidence of insensitivity. Verify the parameter is actually consumed before \
        reporting this result."""
    end

    if include_baseline
        push!(results, (
            parameter="baseline",
            type="none",
            base_value=NaN,
            tested_value=NaN,
            rmse=baseline_rmse,
            rmse_ratio=1.0,
            relative_change=0.0
        ))
    end
    return DataFrame(results)
end