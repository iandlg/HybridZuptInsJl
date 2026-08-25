
"""
    collect_trial_io_online(data_dir, trial_id; frame=BODY, feature_type=THREED_STEP)::Optional{Tuple{CorrectionIO, Matrix{Float64}}}

Load, align and extract training IO for a single trial.
Returns `nothing` if the trial fails (logs a warning).
"""
function collect_trial_io_online(
    data_dir::AbstractString,
    trial_id::Int;
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
    train_ratio::Float64=0.0,
    corrector::AbstractEstimator=BaseEstimator(300)
)::Optional{Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}
    try
        ins_traj_aligned, gt_traj_aligned, _, _, inertial_updated, sim_config_updated = HybridZuptInsJl.compute_aligned_ins_trajectory(
            data_dir, trial_id
        )
        N = length(inertial_updated)
        x_init = vcat(
            ins_traj_aligned.pos[:, 1],
            ins_traj_aligned.vel[:, 1],
            matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
        )
        ## Split this
        n_train_cutoff = floor(Int, train_ratio * N)
        gt_available = [n <= n_train_cutoff for n in 1:N]
        _, step_seg, corr_traj, output_data = hybrid_zupt_aided_insv2(
            inertial_updated, sim_config_updated, gt_traj_aligned, corrector;
            x_init=x_init, gt_available=gt_available, ref_frame=frame, feature_type=feature_type)

        return output_data["target"], output_data["input"], corr_traj, gt_traj_aligned, step_seg
    catch e
        @warn "Skipping trial $trial_id in $data_dir" exception = e
        return nothing
    end
end
# First method: original behaviour (no train_ratios / correctors)
function collect_dataset(
    data_dir::AbstractString,
    trial_ids::AbstractVector{Int};
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
)::Dict{Int,Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}
    valid_dict = Dict{Int,Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}()
    failures = Int[]

    for id in trial_ids
        res = collect_trial_io_online(data_dir, id; frame=frame, feature_type=feature_type)
        if !isnothing(res)
            valid_dict[id] = res
        else
            push!(failures, id)
        end
    end

    isempty(valid_dict) && error("No trials loaded successfully from $data_dir")
    n_loaded = length(valid_dict)
    n_total = length(trial_ids)
    if !isempty(failures)
        @warn "$(length(failures)) / $n_total trials failed and were skipped. Failed IDs: $failures"
    end
    @info "Loaded $n_loaded / $n_total trials"
    return valid_dict
end

# Second method: iterate over training ratios × correctors
function collect_dataset(
    data_dir::AbstractString,
    trial_ids::AbstractVector{Int},
    train_ratios::AbstractVector{<:Real},
    correctors::AbstractDict{<:AbstractString,<:AbstractEstimator};
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
)
    # Outer dict: trial_id -> inner dict
    # Inner key: (corrector_name, train_ratio) -> result tuple
    valid_dict = Dict{Int,Dict{Tuple{String,Float64},Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}}()
    failures = Int[]

    for id in trial_ids
        trial_results = Dict{Tuple{String,Float64},
            Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}()
        any_success = false

        for (corr_name, corr_template) in correctors
            for ratio in train_ratios
                # deepcopy to prevent mutation across runs
                corr = deepcopy(corr_template)
                res = collect_trial_io_online(
                    data_dir, id;
                    frame=frame,
                    feature_type=feature_type,
                    train_ratio=ratio,
                    corrector=corr,
                )
                if !isnothing(res)
                    trial_results[(corr_name, ratio)] = res
                    any_success = true
                end
            end
        end

        if any_success
            valid_dict[id] = trial_results
        else
            push!(failures, id)
        end
    end

    isempty(valid_dict) && error("No trials loaded successfully from $data_dir")
    n_loaded = length(valid_dict)
    n_total = length(trial_ids)
    if !isempty(failures)
        @warn "$(length(failures)) / $n_total trials failed and were skipped. Failed IDs: $failures"
    end
    @info "Loaded $n_loaded / $n_total trials"
    return valid_dict
end

"""
    io_dataframe(valid::Vector) -> DataFrame

Collect every raw time-series sample for each input/output channel across all trials.
Returns a long-form DataFrame with columns:
  `trial_id`  - trial index (1-based)
  `io_type`   - `:input` or `:output`
  `channel`   - channel index
  `t`         - timestamp of the sample (from `CorrectionIO.t`)
  `value`     - individual sample value
"""
function io_dataframe(
    valid::AbstractDict{Int,Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}
)
    @assert length(valid) > 0 "No trials in dataset."

    rows = DataFrame(
        trial_id=Int[],
        io_type=Symbol[],
        channel=Int[],
        t=Float64[],
        value=Float64[],
        std=Float64[]
    )
    trial_ids = keys(valid)
    n_input = size(valid[first(trial_ids)][2].data, 1)
    n_output = size(valid[first(trial_ids)][1].data, 1)

    for (i, (target, input, _, _, _)) in valid
        for ch in 1:n_input
            for (t, v, v_std) in zip(input.t, input.data[ch, :], input.data_std[ch, :])
                push!(rows, (trial_id=i, io_type=:input, channel=ch, t=t, value=v, std=v_std))
            end
        end
        for ch in 1:n_output
            for (t, v, v_std) in zip(target.t, target.data[ch, :], target.data_std[ch, :])
                push!(rows, (trial_id=i, io_type=:output, channel=ch, t=t, value=v, std=v_std))
            end
        end
    end

    return rows
end

function result_performance(res::Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}})
    _, _, corr_traj, gt_traj, step_seg = res
    _rmse = rmse(corr_traj, gt_traj[step_seg])[end]
    _rmse_rate = _rmse / total_distance(gt_traj[step_seg])
    return _rmse, _rmse_rate
end

function result_performance(res::Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}, train_ratio::Float64)
    _, _, corr_traj, gt_traj, step_seg = res
    N_step = length(corr_traj)
    n_train_cutoff = floor(Int, train_ratio * N_step)

    _rmse = rmse(corr_traj[n_train_cutoff:end], gt_traj[step_seg][n_train_cutoff:end])[end]
    return _rmse, _rmse / total_distance(gt_traj[step_seg][n_train_cutoff:end])
end

function performance_dataframe(
    dataset::Dict{Int,Dict{Tuple{String,Float64},Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}}
)::DataFrame
    df = DataFrame(
        trial_id=Int[],
        estimator=String[],
        train_ratio=Float64[],
        rmse=Float64[],
        rmse_rate=Float64[],
    )

    for (id, inner_dict) in dataset
        for ((corr_name, ratio), res) in inner_dict
            _rmse, rmse_rate = result_performance(res, ratio)

            push!(df, (id, corr_name, ratio, _rmse, rmse_rate))
        end
    end

    return df
end


function mahal_sqdistances(M::Matrix{Float64})
    d, n = size(M)
    μ = vec(mean(M, dims=2))
    C = M .- μ
    Σ = (C * C') ./ (n - 1) + 1e-8 * I
    L = cholesky(Symmetric(Σ)).U
    Y = L' \ C                      # L'⁻¹ * C
    sqdists = vec(sum(Y .^ 2, dims=1))   # D² per sample, χ²_d distributed
    return sqdists, d
end


"""
    remove_outliers(input_io::CorrectionIO, output_io::CorrectionIO;
                    method="zscore", threshold=3.0, dims=:both)

Remove outlier samples from paired input/output `CorrectionIO` data.

# Arguments
- `input_io`, `output_io`: objects containing data matrices (features x samples) and aligned time vectors.
- `method`: outlier detection method. Currently only `"zscore"` is implemented.
- `threshold`: absolute z-score above which a sample is considered an outlier.
- `dims`: where to look for outliers (`:input`, `:output`, or `:both`). Default `:both` removes a sample if it is an outlier in either the input or output space.

# Returns
- `(input_clean, output_clean)`: new `CorrectionIO` objects with outlier samples removed.

# Example
```julia
input_clean, output_clean = remove_outliers(input_io, output_io;
                                            method="zscore", threshold=3.0, dims=:both)

"""
function remove_outliers(
    input_io::CorrectionIO, output_io::CorrectionIO;
    method::String="zscore",
    dims::Symbol=:both,
    threshold::Float64=3.0,
    alpha::Float64=0.975,
)
    # Check same t
    if input_io.t != output_io.t
        error("Input and output CorrectionIO must have identical time vectors.")
    end
    n_samples = length(input_io.t)

    # Determine which samples to keep
    if method == "zscore"
        keep = trues(n_samples)
        if dims == :input || dims == :both
            X = input_io.data
            for i in 1:size(X, 1)
                z = (X[i, :] .- mean(X[i, :])) ./ std(X[i, :])
                keep .&= abs.(z) .< threshold
            end
        end
        if dims == :output || dims == :both
            Y = output_io.data
            for i in 1:size(Y, 1)
                z = (Y[i, :] .- mean(Y[i, :])) ./ std(Y[i, :])
                keep .&= abs.(z) .< threshold
            end
        end
    elseif method == "mahalanobis"
        keep = trues(n_samples)
        if dims == :input || dims == :both
            sqdists, d = mahal_sqdistances(input_io.data)
            thresh = sqrt(quantile(Chisq(d), alpha))
            keep .&= sqdists .< thresh
        end
        if dims == :output || dims == :both
            sqdists, d = mahal_sqdistances(output_io.data)
            thresh = sqrt(quantile(Chisq(d), alpha))
            keep .&= sqdists .< thresh
        end

    elseif method == "iqr"
        error("IQR method not implemented yet")
    else
        error("Unknown method: $method")
    end

    # Filter inputs
    t_clean = input_io.t[keep]
    data_input_clean = input_io.data[:, keep]
    data_std_input_clean = isnothing(input_io.data_std) ? nothing : input_io.data_std[:, keep]
    input_clean = CorrectionIO(t_clean, data_input_clean, data_std_input_clean)

    # Filter outputs
    data_output_clean = output_io.data[:, keep]
    data_std_output_clean = isnothing(output_io.data_std) ? nothing : output_io.data_std[:, keep]
    output_clean = CorrectionIO(t_clean, data_output_clean, data_std_output_clean)

    @info "Removed $(sum(.!keep)) out of $n_samples samples ($(round(sum(keep)/n_samples*100, digits=1))% kept)"
    return input_clean, output_clean
end

"""
    compute_input_preprocessing(data; normalize_x=true, margin=0.5)

Compute normalisation constants and a normalised‑space bounding box for input features.

# Arguments
- `data::AbstractMatrix`: input matrix of size `(d, N)` (features × samples).
- `normalize_x::Bool`: if `true`, subtract mean and divide by standard deviation.
- `margin::Real`: fraction of the minimum feature range added to the bounding box
  (applied *after* possible normalisation).

# Returns
A `NamedTuple` with fields:
- `μ`: mean vector of length `d` (zero if `normalize_x = false`).
- `σ`: standard deviation vector of length `d` (one if `normalize_x = false`).
- `LL_norm`: `2×d` matrix; first row = lower bound, second row = upper bound
  of the bounding box in the (possibly normalised) space.
- `mid_norm`: midpoint of the bounding box, `1×d`.
- `Lvec_norm`: half‑width of the bounding box, `1×d`.
"""
function compute_input_preprocessing(data::AbstractMatrix; normalize_x=true, margin=0.5)
    d = size(data, 1)

    if normalize_x
        μ = vec(mean(data, dims=2))
        σ = vec(std(data, dims=2))
        data_norm = (data .- μ) ./ σ
    else
        μ = zeros(d)
        σ = ones(d)
        data_norm = data
    end

    xmin_norm = vec(minimum(data_norm, dims=2))
    xmax_norm = vec(maximum(data_norm, dims=2))
    pm = margin * minimum(xmax_norm - xmin_norm)
    LL_norm = [xmin_norm' .- pm; xmax_norm' .+ pm]
    mid_norm = (LL_norm[1, :] .+ LL_norm[2, :]) ./ 2
    Lvec_norm = (LL_norm[2, :] .- LL_norm[1, :]) ./ 2

    return (; μ, σ, LL_norm, mid_norm, Lvec_norm)
end

"""
    compute_output_normalisation(data; normalize_y=true)

Compute normalisation constants for output variables.

# Arguments
- `data::AbstractMatrix`: output matrix of size `(d_out, N)`.
- `normalize_y::Bool`: if `true`, subtract mean and divide by standard deviation.

# Returns
A `NamedTuple` with fields:
- `μ`: mean vector (zero if `normalize_y = false`).
- `σ`: standard deviation vector (one if `normalize_y = false`).
"""
function compute_output_normalisation(data::AbstractMatrix; normalize_y=true)
    if normalize_y
        μ = vec(mean(data, dims=2))
        σ = vec(std(data, dims=2))
    else
        μ = zeros(size(data, 1))
        σ = ones(size(data, 1))
    end
    return (; μ, σ)
end

"""
    display_preprocessing(inp, outp; input_labels=nothing, output_labels=nothing)

Print a human‑readable summary of normalisation constants and the input bounding box
to the terminal.

# Arguments
- `inp`: named tuple from [`compute_input_preprocessing`](@ref) with fields `μ`, `σ`,
  `LL_norm`, `mid_norm`, `Lvec_norm`.
- `outp`: named tuple from [`compute_output_normalisation`](@ref) with fields `μ`, `σ`.
- `input_labels`: optional vector of strings naming each input feature (length `d`).
- `output_labels`: optional vector of strings naming each output variable (length `d_out`).
"""
function display_preprocessing(inp, outp; input_labels=nothing, output_labels=nothing)
    d_in = length(inp.μ)
    d_out = length(outp.μ)

    # Create default labels if not provided
    in_lbls = isnothing(input_labels) ? ["Input_$i" for i in 1:d_in] : input_labels
    out_lbls = isnothing(output_labels) ? ["Output_$i" for i in 1:d_out] : output_labels

    println("="^60)
    println("        INPUT NORMALISATION & BOUNDING BOX")
    println("="^60)
    println(rpad("Feature", 12), rpad("μ", 14), rpad("σ", 14),
        rpad("Lower (LL)", 14), rpad("Upper (LL)", 14),
        rpad("Mid", 14), rpad("Half-width", 14))
    println("-"^60)
    for i in 1:d_in
        @printf("%-12s %-14.6g %-14.6g %-14.6g %-14.6g %-14.6g %-14.6g\n",
            in_lbls[i], inp.μ[i], inp.σ[i],
            inp.LL_norm[1, i], inp.LL_norm[2, i],
            inp.mid_norm[i], inp.Lvec_norm[i])
    end
    println()

    println("="^60)
    println("        OUTPUT NORMALISATION")
    println("="^60)
    println(rpad("Variable", 12), rpad("μ", 14), rpad("σ", 14))
    println("-"^60)
    for i in 1:d_out
        @printf("%-12s %-14.6g %-14.6g\n", out_lbls[i], outp.μ[i], outp.σ[i])
    end
    println("="^60)
end

struct NoiseSpec
    pos_std::Union{Nothing,Float64,AbstractVector{Float64}}
    pos_bias::AbstractVector{Float64}
    att_std::Union{Nothing,Float64,AbstractVector{Float64}}
    att_bias::AbstractVector{Float64}
    tag::AbstractString

    function NoiseSpec(;
        pos_std::Union{Nothing,Float64,AbstractVector{Float64}}=nothing,
        pos_bias::Union{AbstractVector{Float64},NTuple{3,Float64}}=zeros(Float64, 3),
        att_std::Union{Nothing,Float64,AbstractVector{Float64}}=nothing,
        att_bias::Union{AbstractVector{Float64},NTuple{3,Float64}}=zeros(Float64, 3),
        tag::AbstractString="NoiseSpec"
    )
        pos_std_vec = isnothing(pos_std) ? nothing : (pos_std isa Float64 ? fill(pos_std, 3) : collect(Float64, pos_std))
        pos_bias_vec = collect(Float64, pos_bias)
        att_std_vec = isnothing(att_std) ? nothing : (att_std isa Float64 ? fill(att_std, 3) : collect(Float64, att_std))
        att_bias_vec = collect(Float64, att_bias)

        if !isnothing(pos_std_vec) && length(pos_std_vec) != 3
            throw(ArgumentError("pos_std must be a scalar or a 3-element vector, got length $(length(pos_std_vec))"))
        end
        if length(pos_bias_vec) != 3
            throw(ArgumentError("pos_bias must be a 3-element vector, got length $(length(pos_bias_vec))"))
        end
        if !isnothing(att_std_vec) && length(att_std_vec) != 3
            throw(ArgumentError("att_std must be a scalar or a 3-element vector, got length $(length(att_std_vec))"))
        end
        if length(att_bias_vec) != 3
            throw(ArgumentError("att_bias must be a 3-element vector, got length $(length(att_bias_vec))"))
        end

        new(pos_std_vec, pos_bias_vec, att_std_vec, att_bias_vec, tag)
    end
end

"""
    is_noiseless(spec::NoiseSpec) -> Bool

`true` when `spec` adds no randomness at all (both stds are `nothing` or all-zero),
so repeat draws would produce bit-identical runs. Used by
`run_online_correction_sweep` to run such specs once instead of once per seed.

Note a nonzero *bias* with zero std is still noiseless in this sense: the
perturbation is deterministic, so extra draws buy nothing.
"""
is_noiseless(spec::NoiseSpec)::Bool =
    (isnothing(spec.pos_std) || all(iszero, spec.pos_std)) &&
    (isnothing(spec.att_std) || all(iszero, spec.att_std))

"""
    function run_online_correction_sweep(
        aligned::OrderedDict{String,OrderedDict{Int,NamedTuple}},
        frame::ReferenceFrame,
        feature_type::FeatureType,
        hsgp_params::HsgpParameters,
        train_ratios::AbstractVector{<:Real},
        estimators::AbstractDict{<:AbstractString,<:Type},
        output_channels::Vector{Symbol};
        step_detector_factory::Type=StepDetector,
        estimator_alloc::Int=300,
        pos_std_vec::AbstractVector{<:Union{Nothing,Float64,AbstractVector{Float64}}}=[nothing],
        pos_bias_vec::AbstractVector{<:AbstractVector{Float64}}=[zeros(3)],
        att_std_vec::AbstractVector{<:Union{Nothing,Float64,AbstractVector{Float64}}}=[nothing],
        att_bias_vec::AbstractVector{<:AbstractVector{Float64}}=[zeros(3)],
    )::DataFrame

For every `(dataset_name, trial_id)` pair in `aligned` (as produced by
`collect_aligned_trajectories`), every `train_ratio` in `train_ratios`, and every
estimator type in `estimators`, run `hybrid_zupt_aided_insv2` and record the raw
outputs together with the resulting horizontal RMSE / RMSE-rate.

# Arguments
- `aligned`: `dataset_name => trial_id => NamedTuple` as returned by
  `collect_aligned_trajectories` (needs `inertial_updated`, `sim_config_updated`,
  `gt_traj_aligned`, `x_init`).
- `frame`, `feature_type`: passed straight to `hybrid_zupt_aided_insv2`.
- `hsgp_params`: `HsgpParameters` passed as `params=` to estimator constructors that
  need it (ignored via `kwargs...` by estimators that don't, e.g. `BaseEstimator`).
- `train_ratios`: vector of fractions in `[0,1]` controlling how much of the trial has
  ground truth available (`gt_available[n] = n <= floor(train_ratio*N)`). Order is
  preserved in `train_ratio_order`.
- `estimators`: `OrderedDict{String,Type}` name => estimator type, e.g.
  `OrderedDict("Joint HSGP" => JointHsgpEstimator, "Base" => BaseEstimator)`. Types are
  constructed as `T(estimator_alloc; params=hsgp_params, corrected_channels=output_channels)`.
  Order is preserved in `estimator_order`.
- `output_channels`: e.g. `[:pos_1, :pos_2, :yaw]`, forwarded as `corrected_channels`.
- `seeds`: one noise realisation per seed, per `(trial, train_ratio, noise_spec)`. Each
  realisation comes from its own `Xoshiro(seed)`, so a seed always means the same draw
  no matter what else the sweep contains or in what order it runs — add a trial, drop a
  noise spec, re-run one cell alone, and the rest is unchanged. A single seed (the
  default) gives one realisation per cell, so the spread across rows is walk-to-walk
  variability; more seeds add replicates of the *noise* on top, recorded in the `seed`
  column. Specs for which `is_noiseless` holds are run once regardless (the repeats
  would be identical), under the first seed.

  The seed loop sits outside the estimator loop, so every estimator in a cell sees the
  **same** realisation — that is what makes `paired_estimator_contrast` paired.
  Different trials at the same seed share one stream, and since `randn` is drawn as
  `3 x N` with `N` the trial length, shorter trials get a prefix of what longer ones
  get: draws are identical *within* a cell by design, and only partly independent
  *across* trials.
- `keep_artifacts`: keep the raw `zupt`/`step_seg`/`corr_traj`/`io_data`/`model` objects
  in the returned frame. They cost roughly **2.5 MB per row**, which a one-draw sweep
  can afford and a Monte-Carlo one cannot: 11 trials x 6 specs x 10 draws x 3 estimators
  is ~2000 rows, i.e. several GB of trajectories nothing downstream reads. Pass `false`
  when only the metric columns are wanted; the columns stay in place, filled with
  `nothing`.

# Returns
- `DataFrame` with columns:
  `dataset_name, trial_id, train_ratio, train_ratio_order, estimator, estimator_order,
   noise_spec_tag, noise_spec_order, seed,
   zupt, step_seg, corr_traj, io_data, model,
   rmse, rmse_rate, rmse_yaw`
  where `zupt`, `step_seg`, `corr_traj`, `io_data`, `model` hold the raw objects
  returned by `hybrid_zupt_aided_insv2` (`Any`-typed columns — no serialization).
  `estimator_order`/`train_ratio_order` are 1-based indices matching the iteration
  order of `estimators`/`train_ratios`, useful for plotting in a consistent order.
  Failed `(trial, train_ratio, estimator)` combinations are skipped with a `@warn`.
"""
function run_online_correction_sweep(
    aligned::OrderedDict{String,OrderedDict{Int,NamedTuple}},
    frame::ReferenceFrame,
    feature_type::FeatureType,
    hsgp_params::HsgpParameters,
    train_ratios::AbstractVector{<:Real},
    estimators::AbstractDict{<:AbstractString,<:Type},
    output_channels::Vector{Symbol};
    step_detector_factory::Type=StepDetector,
    estimator_alloc::Int=300,
    noise_specs::AbstractVector{NoiseSpec}=[NoiseSpec()], # Default noise is none at all
    seeds::AbstractVector{Int}=[123],
    keep_artifacts::Bool=true,
    # pos_std_vec::AbstractVector{<:Union{Nothing,Float64,AbstractVector{Float64}}}=[nothing],
    # pos_bias_vec::AbstractVector{<:AbstractVector{Float64}}=[zeros(3)],
    # att_std_vec::AbstractVector{<:Union{Nothing,Float64,AbstractVector{Float64}}}=[nothing],
    # att_bias_vec::AbstractVector{<:AbstractVector{Float64}}=[zeros(3)],
)::DataFrame

    isempty(seeds) && throw(ArgumentError("seeds must not be empty"))
    allunique(seeds) || throw(ArgumentError("seeds must be unique, got $seeds"))

    df = DataFrame(
        dataset_name=String[],
        dataset_order=Int[],
        trial_id=Int[],
        train_ratio=Float64[],
        train_ratio_order=Int[],
        estimator=String[],
        estimator_order=Int[],
        noise_spec_tag=String[],
        noise_spec_order=Int[],
        seed=Int[],
        pos_std=Any[],
        pos_bias=Any[],
        att_std=Any[],
        att_bias=Any[],
        zupt=Any[],
        step_seg=Any[],
        corr_traj=Any[],
        io_data=Any[],
        model=Any[],
        rmse=Float64[],
        rmse_rate=Float64[],
        rmse_yaw=Float64[],
    )

    n_ok = 0
    n_fail = 0

    for (dataset_order, (dataset_name, trials)) in enumerate(aligned)
        for (trial_id, res) in trials
            N = length(res.inertial_updated)

            for (train_ratio_order, train_ratio) in enumerate(train_ratios)
                n_train_cutoff = floor(Int, train_ratio * N)
                gt_available = [n <= n_train_cutoff for n in 1:N]

                for (noise_spec_order, noise_spec) in enumerate(noise_specs)

                    # A noiseless spec is deterministic, so extra seeds would only
                    # duplicate the same run (and would inflate its box with copies
                    # of one point). Run it once, under the first seed.
                    spec_seeds = is_noiseless(noise_spec) ? seeds[1:1] : seeds

                    for seed in spec_seeds

                        # Noisy GT is only used to drive the estimator (training/measurement
                        # updates); RMSE evaluation always uses the clean res.gt_traj_aligned.
                        #
                        # One Xoshiro per seed, drawn once per
                        # (trial, train_ratio, noise_spec, seed) and outside the estimator
                        # loop below: every estimator in this cell therefore sees the SAME
                        # realisation, which is what makes `paired_estimator_contrast` a
                        # paired comparison.
                        gt_traj_noisy = add_gaussian_noise(
                            res.gt_traj_aligned;
                            pos_std=noise_spec.pos_std, pos_bias=noise_spec.pos_bias,
                            att_std=noise_spec.att_std, att_bias=noise_spec.att_bias,
                            rng=Random.Xoshiro(seed),
                        )

                        for (estimator_order, (est_name, est_type)) in enumerate(estimators)
                            try
                                estimator = est_type(
                                    estimator_alloc;
                                    params=hsgp_params,
                                    corrected_channels=output_channels,
                                )

                                zupt, step_seg, corr_traj, io_data, model = hybrid_zupt_aided_insv2(
                                    res.inertial_updated,
                                    res.sim_config_updated,
                                    gt_traj_noisy,
                                    estimator;
                                    step_detector=step_detector_factory(),
                                    x_init=res.x_init,
                                    gt_available=gt_available,
                                    ref_frame=frame,
                                    feature_type=feature_type,
                                )

                                # RMSE evaluated against the clean ground truth
                                gt_step_traj_clean = res.gt_traj_aligned[step_seg]
                                N_step = length(gt_step_traj_clean)
                                n_step_cutoff = floor(Int, train_ratio * N_step)
                                _rmse = rmse(corr_traj[n_step_cutoff:end], gt_step_traj_clean[n_step_cutoff:end])[end]
                                _rmse_rate = _rmse / total_distance(gt_step_traj_clean[n_step_cutoff:end])
                                # Yaw is scored separately: `rmse` is horizontal position only,
                                # so a yaw-channel experiment is otherwise never measured on yaw.
                                _rmse_yaw = rmse_yaw(corr_traj[n_step_cutoff:end], gt_step_traj_clean[n_step_cutoff:end])[end]

                                push!(df, (
                                    dataset_name, dataset_order, trial_id, train_ratio, train_ratio_order,
                                    est_name, estimator_order,
                                    noise_spec.tag, noise_spec_order,
                                    seed,
                                    noise_spec.pos_std, noise_spec.pos_bias,
                                    noise_spec.att_std, noise_spec.att_bias,
                                    (keep_artifacts ? (zupt, step_seg, corr_traj, io_data, model) :
                                     (nothing, nothing, nothing, nothing, nothing))...,
                                    _rmse, _rmse_rate, _rmse_yaw,
                                ))
                                n_ok += 1
                            catch e
                                @warn "Skipping (dataset_name=$dataset_name, trial=$trial_id, train_ratio=$train_ratio, estimator=$est_name, noise=$(noise_spec.tag), seed=$seed)" exception = e
                                n_fail += 1
                            end
                        end
                    end
                end
            end
        end
    end

    @info "run_online_correction_sweep: $n_ok succeeded, $n_fail failed"
    return df
end

# ── Paired comparison of a noise sweep ────────────────────────────────────
#
# A noise sweep runs the SAME trials through every estimator, which makes the
# design paired. Boxing each estimator's metric separately throws that away and
# asks the reader to compare two clouds of ~10 points by eye, while walk-to-walk
# difficulty -- some walks are simply longer or twistier -- dominates the spread.
# Differencing within a trial first cancels that nuisance, leaving the effect.

"The columns that identify one walk under one GT-availability setting."
const _TRIAL_KEYS = [:dataset_name, :dataset_order, :trial_id, :train_ratio, :train_ratio_order]

function _require_cols(df::DataFrame, cols, who::AbstractString)
    missing_cols = [c for c in cols if !hasproperty(df, c)]
    isempty(missing_cols) && return nothing
    throw(ArgumentError("$who: DataFrame is missing column(s) $(missing_cols). \
                         Was it produced by run_online_correction_sweep?"))
end

"""
    paired_estimator_contrast(df; metric=:rmse_rate, reference_estimator="Base (no correction)") -> DataFrame

Per-trial change in `metric` relative to `reference_estimator`, for every noise spec.

Each row pairs one estimator against the reference on the **same**
`(dataset, trial, train_ratio, noise_spec, seed)` cell — the identical noise
realisation, which `run_online_correction_sweep` guarantees by drawing the noise
outside the estimator loop. The reference estimator itself is not in the output: it
is the zero line.

# Returns
`DataFrame` with the trial keys plus `estimator`, `noise_spec_tag`, `seed`, and:
- `value` — the estimator's metric,
- `ref_value` — the reference estimator's metric on that same cell,
- `delta = value - ref_value` — in the metric's own units,
- `rel_change_pct = 100 (value - ref_value) / |ref_value|` — the quantity
  `plot_paired_relative_change` and `plot_noise_paired_relative_change` display.

Negative means the estimator beat the reference on that trial.
"""
function paired_estimator_contrast(
    df::DataFrame;
    metric::Symbol=:rmse_rate,
    reference_estimator::AbstractString="Base (no correction)",
)::DataFrame
    _require_cols(df, vcat(_TRIAL_KEYS, [:estimator, :estimator_order, :noise_spec_tag,
            :noise_spec_order, :seed, metric]), "paired_estimator_contrast")

    key_cols = vcat(_TRIAL_KEYS, [:noise_spec_tag, :noise_spec_order, :seed])

    ref_rows = df[df.estimator .== reference_estimator, :]
    isempty(ref_rows) && throw(ArgumentError(
        "paired_estimator_contrast: no rows with estimator = \"$reference_estimator\". \
         Available: $(join(unique(df.estimator), ", "))"))
    ref = select(ref_rows, key_cols, metric => :ref_value)

    test = select(df[df.estimator .!= reference_estimator, :],
        key_cols, [:estimator, :estimator_order], metric => :value)

    out = innerjoin(test, ref, on=key_cols)
    isempty(out) && throw(ArgumentError(
        "paired_estimator_contrast: no cell has both \"$reference_estimator\" and \
         another estimator — nothing to pair."))

    ok = isfinite.(out.value) .& isfinite.(out.ref_value)
    n_bad = count(!, ok)
    n_bad > 0 && @warn "paired_estimator_contrast: dropping $n_bad pair(s) with non-finite metric values"
    out = out[ok, :]

    out.delta = out.value .- out.ref_value
    out.rel_change_pct = 100 .* out.delta ./ abs.(out.ref_value)
    sort!(out, [:dataset_order, :noise_spec_order, :estimator_order, :trial_id, :seed])
    return out
end
