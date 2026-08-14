
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
    rmse = rmse(corr_traj, gt_traj[step_seg])[end]
    rmse_rate = rmse / total_distance(gt_traj[step_seg])
    return rmse, rmse_rate
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
        corrector=String[],
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

"""
    run_online_correction_sweep(
        aligned::AbstractDict{<:AbstractString,<:AbstractDict{Int,<:NamedTuple}},
        frame::ReferenceFrame,
        feature_type::FeatureType,
        hsgp_params::HsgpParameters,
        train_ratios::AbstractVector{<:Real},
        estimators::AbstractDict{<:AbstractString,<:Type},
        output_channels::Vector{Symbol};
        step_detector_factory::Function=StepDetector,
        estimator_alloc::Int=300,
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

# Returns
- `DataFrame` with columns:
  `dataset_name, trial_id, train_ratio, train_ratio_order, estimator, estimator_order,
   zupt, step_seg, corr_traj, io_data, model,
   rmse, rmse_rate`
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
)::DataFrame

    df = DataFrame(
        dataset_name=String[],
        dataset_order=Int[],
        trial_id=Int[],
        train_ratio=Float64[],
        train_ratio_order=Int[],
        estimator=String[],
        estimator_order=Int[],
        zupt=Any[],
        step_seg=Any[],
        corr_traj=Any[],
        io_data=Any[],
        model=Any[],
        rmse=Float64[],
        rmse_rate=Float64[],
    )

    n_ok = 0
    n_fail = 0

    for (dataset_order, (dataset_name, trials)) in enumerate(aligned)
        for (trial_id, res) in trials
            N = length(res.inertial_updated)

            for (train_ratio_order, train_ratio) in enumerate(train_ratios)
                n_train_cutoff = floor(Int, train_ratio * N)
                gt_available = [n <= n_train_cutoff for n in 1:N]

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
                            res.gt_traj_aligned,
                            estimator;
                            step_detector=step_detector_factory(),
                            x_init=res.x_init,
                            gt_available=gt_available,
                            ref_frame=frame,
                            feature_type=feature_type,
                        )

                        gt_step_traj = res.gt_traj_aligned[step_seg]
                        N = length(gt_step_traj)
                        n_step_cutoff = floor(Int, train_ratio * N)
                        _rmse = rmse(corr_traj[n_step_cutoff:end], gt_step_traj[n_step_cutoff:end])[end]
                        _rmse_rate = _rmse / total_distance(gt_step_traj[n_step_cutoff:end])

                        push!(df, (
                            dataset_name, dataset_order, trial_id, train_ratio, train_ratio_order,
                            est_name, estimator_order,
                            zupt, step_seg, corr_traj, io_data, model,
                            _rmse, _rmse_rate,
                        ))
                        n_ok += 1
                    catch e
                        @warn "Skipping (dataset_name=$dataset_name, trial=$trial_id, train_ratio=$train_ratio, estimator=$est_name)" exception = e
                        n_fail += 1
                    end
                end
            end
        end
    end

    @info "run_online_correction_sweep: $n_ok succeeded, $n_fail failed"
    return df
end