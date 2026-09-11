"""
    ParamSpec(name, type, get_current, set_new, value_generator, probe_kind, probe_unit)

One swept parameter. `probe_kind`/`probe_unit` record *how* the sweep moves it,
so the frame can carry a probe coordinate that does not have to be re-derived
from `tested_value / base_value` downstream:

- `:multiplicative` -- the probe is `tested / base`, a multiplier. Correct for a
  scale parameter (positive, unit-equivariant, no fixed point): the GP length
  scales and signal variances, and the input std.
- `:additive` -- the probe is `(tested - base) / probe_unit(params)`, an offset
  in units of `probe_unit`. Correct for a location parameter, where a multiplier
  is sized by the base value rather than by anything physical, cannot cross
  zero, reverses direction on a negative base, and fixes zero.

`probe_unit` is a function of the base `HsgpParameters` so a location can be
quoted in the scale that matches it -- `mu_x[d]` in units of `sigma_x[d]` -- and
is evaluated once, against the unperturbed parameters.
"""
struct ParamSpec
    name::String
    type::String
    get_current::Function   # (HsgpParameters) -> Float64
    set_new::Function       # (HsgpParameters, Float64) -> HsgpParameters
    value_generator::Function  # (current_value) -> Vector{Float64}
    probe_kind::Symbol      # :multiplicative | :additive
    probe_unit::Function    # (HsgpParameters) -> Float64
end

const _PROBE_KINDS = (:multiplicative, :additive)

# Scale parameters: multiplicative sweep, probe is the multiplier.
function ParamSpec(name, type, get_current, set_new, value_generator)
    ParamSpec(name, type, get_current, set_new, value_generator, :multiplicative, _ -> 1.0)
end

"""
    probe_coordinate(spec, base_params, base_value, tested_value) -> Float64

The swept parameter's position on its own natural axis: a multiplier for a scale
parameter, an offset in `probe_unit` for a location one. Unperturbed is `1.0`
and `0.0` respectively.
"""
function probe_coordinate(spec::ParamSpec, base_params::HsgpParameters,
    base_value::Float64, tested_value::Float64)::Float64
    if spec.probe_kind === :multiplicative
        iszero(base_value) && throw(ArgumentError(
            "parameter \"$(spec.name)\" has base value 0, so a multiplicative probe " *
            "never moves it. Give it an :additive spec instead."))
        return tested_value / base_value
    end
    unit = spec.probe_unit(base_params)
    iszero(unit) && throw(ArgumentError(
        "parameter \"$(spec.name)\" has probe_unit 0, so its offset has no scale to be quoted in."))
    return (tested_value - base_value) / unit
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
        # Rebuilt only to force `m`. Every other field must be carried across
        # explicitly, because the constructor *defaults* what it is not given:
        # `mid_norm` omitted here fell back to `fill(0.0, d)`, so this evaluator
        # ran with no input centering at all while the rest of the pipeline used
        # the trained value, and every `input_center[d]` row of a sensitivity
        # sweep came back bit-identical (see notes/007).
        hsgp_params = HsgpParameters(
            hsgp_params.hp, hsgp_params.d, isnothing(m) ? hsgp_params.m : m, hsgp_params.LL;
            input_stats=hsgp_params.input_stats,
            output_stats=hsgp_params.output_stats,
            mid_norm=hsgp_params.mid_norm
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

"""
    make_hp_param_grid(base_hp, output_channel_idxs; log_range, n_steps, include_noise=true)

Build the one-at-a-time sweep specs for the GP hyperparameters of each requested
output channel.

`include_noise=false` drops the `noise` (σ_n) hyperparameter from the sweep.
That is the useful setting whenever the evaluator runs with
`pred_includes_noise=false`, because σ_n is then loaded into the estimator and
never read: every one of its rows returns a bit-identical RMSE, and the result
is a flat line that looks like an insensitivity finding but is a dead knob.
Excluding it is honest about not having tested it; sweeping it and plotting the
flat line is not.

Parameter *names* keep the true index into the channel vector (`yaw[2]` is the
length scale whether or not noise was swept), so names stay comparable across
runs; only the grid layout closes up.
"""
function make_hp_param_grid(
    base_hp::SeHyperparams, output_channel_idxs::Vector{Int};
    log_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    n_steps::Int=9,
    include_noise::Bool=true
)::Tuple{Vector{ParamSpec},ParamGrid}

    channel_names = [_OUTPUT_NAMES[idx] for idx in output_channel_idxs]
    keep(i) = include_noise || _HYPERPARAM_TYPES[i] != "noise"

    kept_indices = [filter(keep, eachindex(getfield(base_hp, Symbol(name)))) for name in channel_names]
    all(!isempty, kept_indices) || throw(ArgumentError(
        "include_noise=false leaves no hyperparameters to sweep"))

    grid = ParamGrid(channel_names, length.(kept_indices))
    specs = ParamSpec[]

    for (grp_idx, chan_name) in enumerate(channel_names)
        # `col` is the position in the grid; `idx` stays the true index into the
        # channel vector, which is what the parameter name and _HYPERPARAM_TYPES
        # are keyed by. Conflating the two would silently relabel the length
        # scale as the noise term whenever a parameter is skipped.
        for (col, idx) in enumerate(kept_indices[grp_idx])
            name = "$(Symbol(chan_name))[$idx]"
            getter(p) = getfield(p.hp, Symbol(chan_name))[idx]
            setter(p, new_val) = basecopy(p; new_hp=modify_sehp(p.hp, Symbol(chan_name), idx, new_val))
            spec = ParamSpec(name, _HYPERPARAM_TYPES[idx], getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, grp_idx, col, spec)
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

"""
    make_stats_param_grid(base_params; output_channel_idxs, log_range, n_steps,
                          include_input_params=true, include_output_params=true)

One-at-a-time sweep specs for the *normalisation statistics* rather than the GP
hyperparameters: the input mean and std that standardise a feature before it
reaches the kernel, the centering offset (`mid_norm`) that places the HSGP
domain around it, and the output mean/std that scale the prediction back.

The specs are interchangeable with [`make_hp_param_grid`](@ref)'s -- same
`ParamSpec` type, same baseline -- so
`vary_hsgp_parameters(params, f, vcat(hp_specs, stat_specs))` sweeps both
families against one baseline evaluation and the signed-range figure can rank a
length scale against an input std. The `ParamGrid` is per family, because it is
only the panel layout for [`plot_hp_sensitivity`](@ref).

The grid is *not* the same for every family. `input_std` is a scale and keeps the
multiplicative `log_range` sweep; `input_mean` and `input_center` are locations
and get an additive `delta_range` sweep, in units of the matching `sigma_x` and
of normalised space respectively. Both then move normalised space by
`dz = -delta`, so the two location families share one axis and neither is sized
by its own base value. `output_mean` is fitted to exactly zero, which no
multiplicative probe can move; leave `include_output_params=false` unless it is
given an additive spec too.

Group names double as the `type` column, so each family gets its own symbol and
colour in the figures ([`_STAT_PARAM_INFO`](@ref)). The centering family was
typed `mid_norm` (the field it writes) while being *named* `input_center[d]`;
it is now typed `input_center` to match, and the plotting layer aliases the old
spelling so previously saved CSVs still resolve.
"""
function make_stats_param_grid(base_params::HsgpParameters;
    output_channel_idxs::Vector{Int}=[1, 2, 3, 4],
    log_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    delta_range::Tuple{Float64,Float64}=(-2.0, 2.0),
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

    isempty(group_names) && throw(ArgumentError(
        "make_stats_param_grid: include_input_params and include_output_params are both false"))

    grid = ParamGrid(group_names, max_indices)
    specs = ParamSpec[]

    # Row indices are looked up by name rather than hard-coded: with
    # include_input_params=false the output families are rows 1-2, and the
    # literal 4/5 this used to place them at was an out-of-bounds error rather
    # than a mislaid panel.
    row_of(name) = findfirst(==(name), group_names)

    if include_input_params
        # Input means: a LOCATION, swept additively in units of the matching
        # input std, so the probe is the same size in every dimension. Under the
        # multiplicative probe it was sized by mu/sigma, which is 1.478 in dim 1
        # and 0.086 in dim 3 for the trained artifact -- a +-13-sigma excursion
        # against a +-0.8-sigma one, ranked side by side. See notes/008 section 1.
        for dim in 1:d
            name = "input_mean[$dim]"
            getter(p) = p.input_stats[1][dim]
            setter(p, val) = set_input_stat(p, 1, dim, val)
            unit(p) = p.input_stats[2][dim]
            gen(val) = offset_around(val, unit(base_params), delta_range, n_steps)
            spec = ParamSpec(name, "input_mean", getter, setter, gen, :additive, unit)
            push!(specs, spec)
            place_spec!(grid, row_of("input_mean"), dim, spec)
        end

        # Input stds: a SCALE, so the multiplicative probe is the right one and
        # is kept. Note what the resulting axis actually measures, though: `LL`
        # is a stored field and is NOT rescaled with sigma_x, so scaling sigma_x
        # by k dilates z by 1/k about -c inside a fixed box. Both ends are
        # degeneracies of that fixed domain rather than length scales -- which is
        # what `box_exit_frame` is for. See notes/008 section 3.
        for dim in 1:d
            name = "input_std[$dim]"
            getter(p) = p.input_stats[2][dim]
            setter(p, val) = set_input_stat(p, 2, dim, val)
            spec = ParamSpec(name, "input_std", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, row_of("input_std"), dim, spec)
        end

        # Input centering (writes HsgpParameters.mid_norm): a LOCATION that
        # already lives in normalised units, so the offset needs no scaling and
        # `probe_unit` is 1. `c_x[3]` is negative in the trained artifact, which
        # under a multiplier meant "rightward on the axis" was *decrease* for
        # that dimension and *increase* for the other two, along one figure row.
        for dim in 1:d
            name = "input_center[$dim]"
            getter(p) = p.mid_norm[dim]
            setter(p, val) = set_mid_norm_stat(p, dim, val)
            gen(val) = offset_around(val, 1.0, delta_range, n_steps)
            spec = ParamSpec(name, "input_center", getter, setter, gen, :additive, _ -> 1.0)
            push!(specs, spec)
            place_spec!(grid, row_of("input_center"), dim, spec)
        end
    end

    if include_output_params
        # Output means
        for (i, dim) in enumerate(output_channel_idxs)
            name = "output_mean[$dim]"
            getter(p) = p.output_stats[1][dim]
            setter(p, val) = set_output_stat(p, 1, dim, val)
            spec = ParamSpec(name, "output_mean", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, row_of("output_mean"), i, spec)
        end

        # Output stds
        for (i, dim) in enumerate(output_channel_idxs)
            name = "output_std[$dim]"
            getter(p) = p.output_stats[2][dim]
            setter(p, val) = set_output_stat(p, 2, dim, val)
            spec = ParamSpec(name, "output_std", getter, setter; log_range=log_range, n_steps=n_steps)
            push!(specs, spec)
            place_spec!(grid, row_of("output_std"), i, spec)
        end
    end

    return specs, grid
end

"""
    vary_hsgp_parameters(base_params, rmse_func, param_specs; include_baseline=true)

Sweep every spec one at a time against a single baseline evaluation.

Columns: `parameter`, `type`, `base_value`, `tested_value`, `rmse`, `rmse_ratio`,
`relative_change` (a *fraction*, not a percent), plus `probe` and `probe_kind`.

`probe` is the parameter's position on its own natural axis --
[`probe_coordinate`](@ref) -- computed here, where the generator that produced
the value is in scope. Downstream plotting reads that column instead of
recovering a multiplier as `tested_value / base_value`, which is undefined at a
zero base and points the wrong way at a negative one.
"""
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
                probe=probe_coordinate(spec, base_params, current_val, new_val),
                probe_kind=String(spec.probe_kind),
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
            probe=NaN,
            probe_kind="none",
            rmse=baseline_rmse,
            rmse_ratio=1.0,
            relative_change=0.0
        ))
    end
    return DataFrame(results)
end
"""
    raw_features(data_dir, trial_id; ref_frame, feature_type) -> Matrix{Float64}

The trial's raw (un-normalised) stride features, `d x N`.

One pass of [`collect_trial_io_online`](@ref), whose second return value is the
input `CorrectionIO`. `train_ratio` is left at 0 because the features are a
property of the segmentation, not of how much ground truth the filter was given.
"""
function raw_features(data_dir::AbstractString, trial_id::Int;
    ref_frame::ReferenceFrame, feature_type::FeatureType)::Matrix{Float64}
    res = collect_trial_io_online(data_dir, trial_id;
        frame=ref_frame, feature_type=feature_type)
    isnothing(res) && error("raw_features: trial $trial_id in $data_dir failed to load")
    return Matrix{Float64}(res[2].data)
end

"""
    normalised_features(X, params, feature_type) -> Matrix{Float64}

Push raw features `X` (`d x N`) through the estimator's own normalisation:
`z = (x - mu)/sigma - c`, with the angle dimension wrapped between the mean
subtraction and the division.

Calls [`normalize_feature!`](@ref) column by column rather than writing the
arithmetic out again, because the wrap is easy to omit and the result would then
disagree with what the filter actually evaluated.
"""
function normalised_features(X::AbstractMatrix{Float64}, params::HsgpParameters,
    feature_type::FeatureType)::Matrix{Float64}
    Z = similar(X)
    for n in axes(X, 2)
        col = collect(X[:, n])
        z, _ = normalize_feature!(feature_type; feature=col,
            input_stats=params.input_stats, mid_norm=params.mid_norm)
        Z[:, n] = z
    end
    return Z
end

"""
    box_occupancy(X, params, feature_type) -> NamedTuple

Where the normalised features sit relative to the fixed HSGP domain `+-LL`.

`LL` is a stored field of `HsgpParameters`, derived once from the *training*
feature spread with a margin (`compute_input_preprocessing`), and it is **not**
rescaled when the normalisation statistics are perturbed. So a sweep of
`sigma_x` dilates `z` inside a box that does not move, and past a certain
multiplier the features simply leave it -- where the Dirichlet sinusoid basis
cannot represent them. This function is what turns that from an assertion into a
measured crossing point.

Fields:
- `frac_outside` -- fraction of strides with any `|z_d| > LL_d`.
- `max_z_ratio`  -- `max(|z_d| / LL_d)` over all dims and strides; `> 1` means at
  least one stride is outside.
- `z_std`, `z_mean` -- per-dimension, which show the other end of the sweep:
  large `sigma_x` collapses `z` onto `-c` and the GP returns a constant.
"""
function box_occupancy(X::AbstractMatrix{Float64}, params::HsgpParameters,
    feature_type::FeatureType)
    Z = normalised_features(X, params, feature_type)
    ratio = abs.(Z) ./ params.LL
    return (
        frac_outside=mean(vec(any(ratio .> 1.0, dims=1))),
        max_z_ratio=maximum(ratio),
        z_std=vec(std(Z, dims=2)),
        z_mean=vec(mean(Z, dims=2)),
    )
end

"""
    box_exit_frame(X, base_params, param_specs, feature_type) -> DataFrame

[`box_occupancy`](@ref) at every value the sweep tests, for the specs that change
the normalisation. Costs no filter runs -- `z` is a closed-form function of
`(mu_x, sigma_x, c_x)` -- so it can be computed at full resolution alongside a
coarse RMSE sweep.

Joins the sweep frame on `(parameter, tested_value)`. Columns: `parameter`,
`type`, `base_value`, `tested_value`, `probe`, `probe_kind`, `frac_outside`,
`max_z_ratio`, plus `z_std_d`/`z_mean_d` for the perturbed dimension `d`.
"""
function box_exit_frame(X::AbstractMatrix{Float64}, base_params::HsgpParameters,
    param_specs::Vector{ParamSpec}, feature_type::FeatureType)::DataFrame
    rows = []
    for spec in param_specs
        dim = _param_dim(spec.name)
        isnothing(dim) && continue
        current_val = spec.get_current(base_params)
        for new_val in spec.value_generator(current_val)
            occ = box_occupancy(X, spec.set_new(base_params, new_val), feature_type)
            push!(rows, (
                parameter=spec.name,
                type=spec.type,
                base_value=current_val,
                tested_value=new_val,
                probe=probe_coordinate(spec, base_params, current_val, new_val),
                probe_kind=String(spec.probe_kind),
                frac_outside=occ.frac_outside,
                max_z_ratio=occ.max_z_ratio,
                z_std_d=occ.z_std[dim],
                z_mean_d=occ.z_mean[dim],
            ))
        end
    end
    isempty(rows) && throw(ArgumentError(
        "box_exit_frame: none of the given specs perturb the input normalisation"))
    return DataFrame(rows)
end

# "input_std[2]" -> 2. `nothing` for a spec that does not address a feature
# dimension, which is how box_exit_frame skips the GP hyperparameters: their
# bracket is a position in a channel vector, not a dimension of the input space.
function _param_dim(name::AbstractString)::Optional{Int}
    startswith(name, "input_") || return nothing
    m = match(r"\[(\d+)\]$", name)
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

"""
    box_exit_points(box_df) -> DataFrame

Reduce [`box_exit_frame`](@ref) to the answer worth quoting: for each
normalisation parameter and each direction of its probe, the probe value at which
the features first leave the fixed domain.

`exit_probe` is where `max_z_ratio` first exceeds 1 -- the first stride outside
the box; `half_out_probe` is where `frac_outside` first exceeds 0.5. `missing`
means the crossing never happens within the swept range, which is itself the
result for a parameter that stays inside the box throughout.
"""
function box_exit_points(box_df::DataFrame)::DataFrame
    rows = []
    for sub in groupby(box_df, :parameter)
        unperturbed = first(sub.probe_kind) == "multiplicative" ? 1.0 : 0.0
        for dir in (:increasing, :decreasing)
            sel = dir === :increasing ? sub.probe .>= unperturbed : sub.probe .<= unperturbed
            side = sort(sub[sel, :], :probe; rev=(dir === :decreasing))
            nrow(side) > 1 || continue
            first_at(pred) = (i = findfirst(pred); isnothing(i) ? missing : side.probe[i])
            push!(rows, (
                parameter=first(sub.parameter),
                type=first(sub.type),
                probe_kind=first(sub.probe_kind),
                direction=String(dir),
                exit_probe=first_at(side.max_z_ratio .> 1.0),
                half_out_probe=first_at(side.frac_outside .> 0.5),
                max_z_ratio_at_end=last(side.max_z_ratio),
                frac_outside_at_end=last(side.frac_outside),
            ))
        end
    end
    return DataFrame(rows)
end

"""
    sweep_over_trials(base_params, param_specs, make_evaluator, trial_ids;
                      include_baseline=true, checkpoint_dir=nothing) -> DataFrame

Repeat [`vary_hsgp_parameters`](@ref) once per trial and stack the frames with a
`trial_id` column.

`make_evaluator(trial_id)` builds that trial's RMSE closure. Each trial is scored
against **its own** baseline, so `relative_change` is a within-trial contrast and
the design is paired -- the same construction
[`plot_paired_relative_change`](@ref) relies on.

What the repetition buys is a reference *band*, not a noise floor. The pipeline
is deterministic under a fixed seed, so the unperturbed point is exactly 0 in
every trial by construction. The usable reading is the across-trial spread at
each probe: a parameter whose band contains 0 everywhere has no demonstrated
effect, and one whose band stays clear of 0 with a consistent sign does.

`checkpoint_dir` writes each trial's frame as it completes, so a long run
survives a crash and can be replotted without recomputing.
"""
function sweep_over_trials(
    base_params::HsgpParameters,
    param_specs::Vector{ParamSpec},
    make_evaluator::Function,
    trial_ids::AbstractVector{Int};
    include_baseline::Bool=true,
    checkpoint_dir::Optional{String}=nothing
)::DataFrame
    isempty(trial_ids) && throw(ArgumentError("sweep_over_trials: no trial ids given"))
    isnothing(checkpoint_dir) || mkpath(checkpoint_dir)

    frames = DataFrame[]
    for (i, tid) in enumerate(trial_ids)
        @info "Sensitivity sweep: trial $tid ($i/$(length(trial_ids)))"
        df = vary_hsgp_parameters(base_params, make_evaluator(tid), param_specs;
            include_baseline=include_baseline)
        df.trial_id = fill(tid, nrow(df))
        isnothing(checkpoint_dir) ||
            CSV.write(joinpath(checkpoint_dir, "trial_$(tid).csv"), df)
        push!(frames, df)
    end
    return vcat(frames...)
end

"""
    sign_test_p(k, n) -> Float64

Two-sided exact binomial sign test: the probability of a split at least as
lopsided as `k` of `n` under p = 0.5.

The sweep is paired -- every trial is scored against its own baseline -- so the
question "did this perturbation move RMSE the same way in every trial" is a sign
test on `n` paired observations, and needs no assumption about the (very
skewed, see notes/009 section 6) distribution of the changes themselves.
"""
function sign_test_p(k::Int, n::Int)::Float64
    n == 0 && return 1.0
    tail = sum(binomial(n, i) for i in 0:min(k, n - k))
    return min(1.0, 2 * tail / 2.0^n)
end

"""
    probe_agreement(df) -> DataFrame

Reduce a [`sweep_over_trials`](@ref) frame to one row per parameter: how far the
across-trial median moved, and whether the trials agreed about it.

Columns: `parameter`, `type`, `probe_kind`, `span` (range of the across-trial
median curve), `worst_probe` (the non-identity probe with the largest |median|),
`median_pct`, `q25`, `q75` (across trials at `worst_probe`), `n_agree`,
`n_trials`, `p_value`.

`n_agree` is the majority count `max(k, n-k)` over the sign of the change, and
`p_value` is [`sign_test_p`](@ref) on it. This is the statistic that separates a
parameter that moved RMSE from one whose wide range is a handful of trials
disagreeing: with the pipeline deterministic under a fixed seed there is no
run-to-run noise floor to test against, so agreement across trials is the
reference the design does provide.

All percentages, matching the plotting layer and `relative_change * 100`.
"""
function probe_agreement(df::DataFrame)::DataFrame
    work = df[df.parameter.!="baseline", :]
    isempty(work) && throw(ArgumentError("probe_agreement: no swept rows in frame"))
    work = copy(work)
    work.pct = 100 .* float.(work.relative_change)

    rows = []
    for sub in groupby(work, :parameter)
        identity_probe = first(sub.probe_kind) == "multiplicative" ? 1.0 : 0.0
        by_probe = combine(groupby(sub, :probe), :pct => median => :med)
        span = maximum(by_probe.med) - minimum(by_probe.med)

        moved = by_probe[abs.(by_probe.probe .- identity_probe).>1e-9, :]
        nrow(moved) == 0 && continue
        worst = moved.probe[argmax(abs.(moved.med))]

        vals = sub[isapprox.(sub.probe, worst; atol=1e-9), :pct]
        n = length(vals)
        k = count(>(0), vals)
        push!(rows, (
            parameter=first(sub.parameter),
            type=first(sub.type),
            probe_kind=first(sub.probe_kind),
            span=span,
            worst_probe=worst,
            median_pct=median(vals),
            q25=n < 4 ? minimum(vals) : quantile(vals, 0.25),
            q75=n < 4 ? maximum(vals) : quantile(vals, 0.75),
            n_agree=max(k, n - k),
            n_trials=n,
            p_value=sign_test_p(k, n),
        ))
    end
    return sort!(DataFrame(rows), :span; rev=true)
end

"""
    box_exit_over_trials(data_dir, trial_ids, base_params, param_specs, feature_type) -> DataFrame

[`box_exit_frame`](@ref) for each trial, stacked with a `trial_id` column.

The domain boundary is not a constant: across the 11 ANG2 trials the multiplier
at which `sigma_x` carries the features outside `±LL` ranges ×0.13 to ×0.47, and
one trial is already outside before any perturbation. Shading a single trial's
boundary as though it were *the* boundary would misstate that, and the
diagnostic is closed-form in the normalisation statistics, so every trial costs
one feature extraction and no filter runs.
"""
function box_exit_over_trials(
    data_dir::AbstractString,
    trial_ids::AbstractVector{Int},
    base_params::HsgpParameters,
    param_specs::Vector{ParamSpec},
    feature_type::FeatureType;
    ref_frame::ReferenceFrame
)::DataFrame
    frames = DataFrame[]
    for tid in trial_ids
        X = raw_features(data_dir, tid; ref_frame=ref_frame, feature_type=feature_type)
        f = box_exit_frame(X, base_params, param_specs, feature_type)
        f.trial_id = fill(tid, nrow(f))
        push!(frames, f)
    end
    return vcat(frames...)
end

"""
    box_outside_spans(box_df, parameter; level=0.5) -> Vector{Tuple{Float64,Float64}}

The contiguous probe intervals over which at least `level` of the trials have
features outside `±LL`.

Returned as intervals rather than a single threshold because a *location*
parameter leaves the domain at both ends of its probe, so there are two spans
and a lone exit point would describe neither. A parameter that never leaves
returns an empty vector, which is a result in its own right and the caller is
expected to say so rather than draw nothing.
"""
function box_outside_spans(box_df::DataFrame, parameter::AbstractString;
    level::Real=0.5)::Vector{Tuple{Float64,Float64}}
    sub = box_df[box_df.parameter.==parameter, :]
    isempty(sub) && return Tuple{Float64,Float64}[]

    frac = combine(groupby(sub, :probe),
        :max_z_ratio => (v -> count(>(1.0), v) / length(v)) => :frac)
    sort!(frac, :probe)
    outside = frac.frac .>= level

    spans = Tuple{Float64,Float64}[]
    i = 1
    while i <= length(outside)
        if outside[i]
            j = i
            while j < length(outside) && outside[j+1]
                j += 1
            end
            push!(spans, (frac.probe[i], frac.probe[j]))
            i = j + 1
        else
            i += 1
        end
    end
    return spans
end
