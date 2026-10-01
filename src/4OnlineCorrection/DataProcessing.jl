
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
        ins_traj_aligned, gt_traj_aligned, _, _, inertial_updated, sim_config_updated = compute_aligned_ins_trajectory(
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
        _, step_seg, corr_traj, output_data = hybrid_zupt_aided_insv4(
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

function mahal_sqdistances(M::Matrix{Float64})
    d, n = size(M)
    μ = vec(mean(M, dims=2))
    C = M .- μ
    Σ = (C * C') ./ (n - 1) + 1e-8 * I
    L = cholesky(Symmetric(Σ)).U
    Y = L' \ C                      # L'⁻¹ * C
    sqdists = vec(sum(Y .^ 2, dims=1))   # D² per sample, χ²_d under a Gaussian
    return sqdists, d
end

"""
    mahal_keep(data, keep_fraction) -> BitVector

Keep-mask for the `keep_fraction` of samples with the smallest squared Mahalanobis
distance. Cuts at the empirical quantile rather than χ², so it retains exactly that
fraction even when the data are not Gaussian.
"""
function mahal_keep(data::Matrix{Float64}, keep_fraction::Float64)
    0 < keep_fraction <= 1 ||
        throw(ArgumentError("keep_fraction must be in (0,1], got $keep_fraction"))
    sqdists, _ = mahal_sqdistances(data)
    return sqdists .<= quantile(sqdists, keep_fraction)
end


"""
    remove_outliers(input_io, output_io; method="zscore", dims=:both,
                    threshold=3.0, keep_fraction=0.85) -> (input_clean, output_clean)

Drop outlier samples from both `CorrectionIO`s together, keeping them row-aligned.

- `method`: `"zscore"` (per channel, uses `threshold`) or `"mahalanobis"` (joint, uses
  `keep_fraction`, see [`mahal_keep`](@ref)).
- `dims`: `:input`, `:output` or `:both`; `:both` drops a sample flagged in either space.
"""
function remove_outliers(
    input_io::CorrectionIO, output_io::CorrectionIO;
    method::String="zscore",
    dims::Symbol=:both,
    threshold::Float64=3.0,
    keep_fraction::Float64=0.85,
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
            keep .&= mahal_keep(input_io.data, keep_fraction)
        end
        if dims == :output || dims == :both
            keep .&= mahal_keep(output_io.data, keep_fraction)
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
    compute_input_preprocessing(data; normalize_x=true, margin=0.5) -> NamedTuple

Normalisation constants and HSGP bounding box for a `(d, N)` input matrix. Returns
`μ`, `σ` (identity if `!normalize_x`), `LL_norm` (`2×d` lower/upper bounds in normalised
space, widened by `margin` × the smallest feature range), `mid_norm` and `Lvec_norm`
(box midpoint and half-width).
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
    compute_output_normalisation(data; normalize_y=true) -> (; μ, σ)

Per-channel mean and std of a `(d_out, N)` output matrix (zeros/ones if `!normalize_y`).
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

`true` when `spec` adds no randomness (stds `nothing` or zero; a bias alone is
deterministic), so the sweep runs it once instead of once per seed.
"""
is_noiseless(spec::NoiseSpec)::Bool =
    (isnothing(spec.pos_std) || all(iszero, spec.pos_std)) &&
    (isnothing(spec.att_std) || all(iszero, spec.att_std))

"""
    matched_gt_sigma_config(cfg::InsConfig, spec::NoiseSpec) -> InsConfig

Copy of `cfg` with `sigma_groundtruth = hypot(clean, injected)`: the R a filter should
get once `spec`'s noise is in its ground truth. This feeds the ground-truth stride
covariance, the mocap `Σy` and V4's initial corrector covariance. Only the yaw part of
the attitude noise enters channel 4.
"""
function matched_gt_sigma_config(cfg::InsConfig, spec::NoiseSpec)::InsConfig
    σ = sigma_groundtruth_array(cfg)
    inj = zeros(4)
    isnothing(spec.pos_std) || (inj[1:3] = spec.pos_std)
    isnothing(spec.att_std) || (inj[4] = spec.att_std[3])
    matched = deepcopy(cfg)
    matched.sigma_groundtruth = Tuple(sqrt.(σ .^ 2 .+ inj .^ 2))
    return matched
end

"""
    run_online_correction_sweep(aligned, frame, feature_type, hsgp_params, train_ratios,
                                estimators, output_channels; kwargs...) -> DataFrame

Run `correction_filter` (default `hybrid_zupt_aided_insv4`) for every trial in `aligned`
(from `collect_aligned_trajectories`) × `train_ratio` × noise spec × seed × estimator,
and record the RMSE metrics.

- `train_ratios`: ground truth is available for `n <= floor(train_ratio*N)`.
- `estimators`: name => type, constructed as `T(estimator_alloc; params=hsgp_params,
  corrected_channels=output_channels, estimator_kwargs...)`. Constructors swallow
  unknown keywords, so to compare two settings run one sweep per setting.
- `seeds`: each noise draw comes from its own `Xoshiro(seed)`, drawn outside the
  estimator loop so every estimator in a cell sees the same realisation (this is what
  `paired_estimator_contrast` pairs on). Noiseless specs run once.
- `match_gt_sigma`: give the filter the R matching the injected noise
  ([`matched_gt_sigma_config`](@ref)); `false` keeps the trial's clean R.
- `posyaw_measurement_update`: `false` gives the no-mocap dead-reckoning reference.
- `keep_artifacts`: keep the raw `zupt`/`step_seg`/`corr_traj`/`io_data`/`model` objects
  (~2.5 MB per row); `false` fills those columns with `nothing`.

Rows carry the trial keys, `*_order` columns (iteration order, for plotting), `seed`,
`gt_sigma_pos`/`gt_sigma_yaw` (the R given), the artifacts, and `rmse`, `rmse_rate`,
`rmse_yaw`. Failed runs are skipped with a `@warn`.
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
    estimator_kwargs::NamedTuple=(;),
    noise_specs::AbstractVector{NoiseSpec}=[NoiseSpec()], # Default noise is none at all
    seeds::AbstractVector{Int}=[123],
    keep_artifacts::Bool=true,
    correction_filter::Function=hybrid_zupt_aided_insv4,
    # Tell the filters how noisy the ground truth they are handed actually is
    # (`matched_gt_sigma_config`). `false` (the default, and what every existing
    # run used) leaves `sigma_groundtruth` at the trial's clean value whatever
    # noise is injected, so at `pos_std=1.0` every filter is handed an R that is
    # ~100x too tight.
    match_gt_sigma::Bool=false,
    # Forwarded to the filter. `false` runs without any mocap fix, i.e. the
    # dead-reckoning reference: what the corrector does when it ignores ground
    # truth entirely.
    posyaw_measurement_update::Bool=true,
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
        # The R the run was given, not the noise it was given: with
        # `match_gt_sigma=false` these stay at the trial's clean sigma_groundtruth
        # however much noise `pos_std`/`att_std` injected.
        gt_sigma_pos=Float64[],
        gt_sigma_yaw=Float64[],
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

                    # The R every estimator in this cell is given. Built once per
                    # spec so all of them share it exactly, like the noise draw
                    # below.
                    sim_config = match_gt_sigma ?
                                 matched_gt_sigma_config(res.sim_config_updated, noise_spec) :
                                 res.sim_config_updated
                    gt_sigma = sigma_groundtruth_array(sim_config)

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
                                    estimator_kwargs...,
                                )

                                zupt, step_seg, corr_traj, io_data, model = correction_filter(
                                    res.inertial_updated,
                                    sim_config,
                                    gt_traj_noisy,
                                    estimator;
                                    step_detector=step_detector_factory(),
                                    x_init=res.x_init,
                                    gt_available=gt_available,
                                    ref_frame=frame,
                                    feature_type=feature_type,
                                    posyaw_measurement_update=posyaw_measurement_update,
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
                                    gt_sigma[1], gt_sigma[4],
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

"""
    run_online_nees_sweep(aligned, frame, feature_type, hsgp_params, train_ratios,
        estimators, output_channels; estimator_alloc, correction_filter) -> DataFrame

Same runs as `run_online_correction_sweep` (clean ground truth), scored on consistency:
one row per footfall with the corrector's position NEES (3 dof) and yaw NEES (1 dof),
`phase` `"train"` while mocap is available and `"test"` after.
"""
function run_online_nees_sweep(
    aligned::OrderedDict{String,OrderedDict{Int,NamedTuple}},
    frame::ReferenceFrame,
    feature_type::FeatureType,
    hsgp_params::HsgpParameters,
    train_ratios::AbstractVector{<:Real},
    estimators::AbstractDict{<:AbstractString,<:Type},
    output_channels::Vector{Symbol};
    estimator_alloc::Int=300,
    correction_filter::Function=hybrid_zupt_aided_insv4,
)::DataFrame
    df = DataFrame(dataset_name=String[], dataset_order=Int[], trial_id=Int[],
        train_ratio=Float64[], train_ratio_order=Int[],
        estimator=String[], estimator_order=Int[],
        k=Int[], phase=String[], nees_pos=Float64[], nees_yaw=Float64[])

    for (dataset_order, (dataset_name, trials)) in enumerate(aligned)
        for (trial_id, res) in trials
            N = length(res.inertial_updated)
            for (train_ratio_order, train_ratio) in enumerate(train_ratios)
                n_train_cutoff = floor(Int, train_ratio * N)
                gt_available = [n <= n_train_cutoff for n in 1:N]

                for (estimator_order, (est_name, est_type)) in enumerate(estimators)
                    try
                        estimator = est_type(estimator_alloc;
                            params=hsgp_params, corrected_channels=output_channels)
                        d = CorrectorDiagnostics()
                        correction_filter(res.inertial_updated, res.sim_config_updated,
                            res.gt_traj_aligned, estimator;
                            x_init=res.x_init, gt_available=gt_available,
                            ref_frame=frame, feature_type=feature_type, diagnostics=d)

                        npos = corrector_nees_series(d, res.gt_traj_aligned).pos
                        nyaw = nees_yaw_series(d, res.gt_traj_aligned).yaw
                        for (i, k) in enumerate(d.k)
                            push!(df, (dataset_name, dataset_order, trial_id,
                                train_ratio, train_ratio_order, est_name, estimator_order,
                                k, k <= n_train_cutoff ? "train" : "test", npos[i], nyaw[i]))
                        end
                    catch e
                        @warn "Skipping (dataset_name=$dataset_name, trial=$trial_id, train_ratio=$train_ratio, estimator=$est_name)" exception = e
                    end
                end
            end
        end
    end
    return df
end

"""
    nees_summary(df) -> DataFrame

Per run and phase of a `run_online_nees_sweep` frame: footfall count, ANEES, median NEES
and the fraction inside the 95% χ² envelope, for position and yaw.
"""
function nees_summary(df::DataFrame)::DataFrame
    inside(v, dof) = consistency_ratio(v, quantile(Chisq(dof), 0.025), quantile(Chisq(dof), 0.975))
    return combine(groupby(df, [:dataset_name, :dataset_order, :trial_id, :train_ratio,
            :train_ratio_order, :estimator, :estimator_order, :phase]),
        nrow => :n,
        :nees_pos => mean => :anees_pos,
        :nees_pos => median => :median_nees_pos,
        :nees_pos => (v -> inside(v, 3)) => :inside_pos,
        :nees_yaw => mean => :anees_yaw,
        :nees_yaw => median => :median_nees_yaw,
        :nees_yaw => (v -> inside(v, 1)) => :inside_yaw)
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
    paired_estimator_contrast(df; metric=:rmse_rate, reference_estimator="ZUPT only",
                              train_ratios=nothing, noise_spec_tags=nothing) -> DataFrame

Per-trial change in `metric` relative to `reference_estimator`, paired on the same
`(dataset, trial, train_ratio, noise_spec, seed)` cell. `train_ratios` and
`noise_spec_tags` filter rows (not the pairing); a value missing from `df` is an error.

Returns the trial keys plus `estimator`, `noise_spec_tag`, `seed`, `value`, `ref_value`,
`delta` and `rel_change_pct`. Negative means the estimator beat the reference.
"""
function paired_estimator_contrast(
    df::DataFrame;
    metric::Symbol=:rmse_rate,
    reference_estimator::AbstractString="ZUPT only",
    train_ratios::Union{Nothing,AbstractVector{<:Real}}=nothing,
    noise_spec_tags::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
)::DataFrame
    _require_cols(df, vcat(_TRIAL_KEYS, [:estimator, :estimator_order, :noise_spec_tag,
            :noise_spec_order, :seed, metric]), "paired_estimator_contrast")

    if !isnothing(train_ratios)
        absent = setdiff(train_ratios, unique(df.train_ratio))
        isempty(absent) || throw(ArgumentError(
            "paired_estimator_contrast: train_ratio(s) $(absent) are not in the frame. \
             Available: $(sort(unique(df.train_ratio)))"))
        df = df[in.(df.train_ratio, Ref(train_ratios)), :]
    end

    if !isnothing(noise_spec_tags)
        absent = setdiff(noise_spec_tags, unique(df.noise_spec_tag))
        isempty(absent) || throw(ArgumentError(
            "paired_estimator_contrast: noise spec tag(s) $(absent) are not in the frame. \
             Available: $(join(unique(df.noise_spec_tag), ", "))"))
        df = df[in.(df.noise_spec_tag, Ref(noise_spec_tags)), :]
    end

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

# ── Learning curve: how much online mocap does the correction need? ───────
#
# `run_online_correction_sweep` varies `train_ratio`, which moves two things at once:
# it scores each cell on `corr_traj[floor(r*N_s):end]`, so more training also buys a
# shorter, later evaluation window. A ZUPT-INS drifts with open-loop distance, so its
# RMSE falls with `train_ratio` whether or not anything was learned, and the columns of
# that figure cannot be compared with each other. See notes/011.
#
# Here the evaluation window is a fixed number of strides at the END of the walk, and
# the budget is the window of ground-truth strides immediately before it: recency and
# the scored strides are both held fixed, and only the amount of mocap varies.
#
#   stride:  1 ............ ks-b .... ks | ks+1 ...... N
#   run:     .   not run   . [start on mocap ...................]
#   mocap:                  [==== b ====] |   none (test)
#   score:                                 [=== n_test ==]
#
# Each run starts at `ks-b` on the mocap pose rather than at stride 1. Run from stride 1,
# the open-loop prefix leaks into the test window twice: the "ZUPT only" filter's yaw
# fixes remove only part of the prefix heading error (its attitude covariance is
# overconfident), and the correctors put the whole prefix drift into `β` at the first
# fix. Either way the budget would be "the prefix plus b strides", not b strides.
#
# V4 has no separate train/anchor switch: `β` lives in the corrector's error state and
# the mocap pose update is what learns it (JointStrideEstimators.jl), so the budget IS
# the mocap. Budget 0 starts on mocap at the split and never sees another fix: the
# open-loop reference over exactly the scored strides. The heading entering the test
# window still depends on `b`, so the `"ZUPT only"` reference is re-run at every budget
# and `learning_curve_contrast` pairs within a budget rather than assuming one reference
# per trial.

"The part of an aligned trial from sample `s0` on, started from the INS state there."
function _trial_from(res::NamedTuple, s0::Int)::NamedTuple
    return (;
        inertial_updated=res.inertial_updated[s0:end],
        gt_traj_aligned=res.gt_traj_aligned[s0:end],
        sim_config_updated=res.sim_config_updated,
        x_init=initial_state(res.ins_traj_aligned, s0),
    )
end

"""
    run_online_learning_curve(aligned, frame, feature_type, hsgp_params, budgets,
                              estimators, output_channels; n_test_strides, ...) -> DataFrame

For every trial × budget × estimator, run `correction_filter` with mocap over exactly
`budget` strides before the split and score on the last `n_test_strides` strides.

Budget `b` starts on the mocap pose at `step_seg[k_split - b]` (`b+1` fixes bounding
`b` strides); budget 0 starts at the split open loop. A budget longer than a trial's
pre-split prefix is skipped for that trial, not clamped.
"""
function run_online_learning_curve(
    aligned::OrderedDict{String,OrderedDict{Int,NamedTuple}},
    frame::ReferenceFrame,
    feature_type::FeatureType,
    hsgp_params::HsgpParameters,
    budgets::AbstractVector{Int},
    estimators::AbstractDict{<:AbstractString,<:Type},
    output_channels::Vector{Symbol};
    n_test_strides::Int,
    step_detector_factory::Type=StepDetector,
    estimator_alloc::Int=300,
    estimator_kwargs::NamedTuple=(;),
    correction_filter::Function=hybrid_zupt_aided_insv4,
    keep_artifacts::Bool=false,
)::DataFrame

    isempty(budgets) && throw(ArgumentError("budgets must not be empty"))
    allunique(budgets) || throw(ArgumentError("budgets must be unique, got $budgets"))
    all(>=(0), budgets) || throw(ArgumentError("budgets must be non-negative, got $budgets"))
    n_test_strides > 0 ||
        throw(ArgumentError("n_test_strides must be positive, got $n_test_strides"))

    df = DataFrame(
        dataset_name=String[],
        dataset_order=Int[],
        trial_id=Int[],
        train_strides=Int[],
        train_strides_order=Int[],
        estimator=String[],
        estimator_order=Int[],
        n_strides=Int[],
        k_split=Int[],
        n_test_strides=Int[],
        test_distance_m=Float64[],
        zupt=Any[],
        step_seg=Any[],
        corr_traj=Any[],
        io_data=Any[],
        model=Any[],
        rmse=Float64[],
        rmse_rate=Float64[],
        rmse_yaw=Float64[],
        final_pos_err=Float64[],
    )

    n_ok = 0
    n_fail = 0
    n_short = 0

    for (dataset_order, (dataset_name, trials)) in enumerate(aligned)
        for (trial_id, res) in trials
            N = length(res.inertial_updated)

            # The segmentation depends only on the ZUPT detector and the IMU stream, not
            # on the corrector or on what ground truth is available, so one throwaway run
            # fixes the stride grid every cell of this trial is built on. Each cell below
            # asserts it got that same grid back.
            _, step_seg_ref, _, _, _ = correction_filter(
                res.inertial_updated,
                res.sim_config_updated,
                res.gt_traj_aligned,
                BaseEstimator(estimator_alloc);
                step_detector=step_detector_factory(),
                x_init=res.x_init,
                gt_available=zeros(Bool, N),
                ref_frame=frame,
                feature_type=feature_type,
            )

            n_strides = length(step_seg_ref)
            k_split = n_strides - n_test_strides
            if k_split < 2
                @warn "trial $trial_id has $n_strides strides, too few for a \
                       $n_test_strides-stride test window; skipping"
                continue
            end

            split_sample = step_seg_ref[k_split]
            gt_step = res.gt_traj_aligned[step_seg_ref]
            test_ks = (k_split+1):n_strides
            test_distance = total_distance(gt_step[test_ks])

            for (budget_order, budget) in enumerate(budgets)
                first_k = k_split - budget
                if first_k < 1
                    n_short += 1
                    @info "trial $trial_id: $(k_split - 1) strides before the split, \
                           budget $budget skipped"
                    continue
                end
                # The run starts at the window, on mocap. Ground truth is read only at
                # footfall samples, so marking the closed sample range
                # [s0, split_sample] gives exactly `budget` strides with mocap at both ends.
                s0 = step_seg_ref[first_k]
                sub = _trial_from(res, s0)
                gt_available = [n <= split_sample - s0 + 1 for n in 1:(N-s0+1)]

                for (estimator_order, (est_name, est_type)) in enumerate(estimators)
                    # Only the filter run is guarded: a trial that diverges should cost one
                    # cell, but a segmentation mismatch or a scoring failure below is a bug
                    # in the design of the sweep and must not be swallowed as a skipped cell.
                    result = nothing
                    try
                        estimator = est_type(
                            estimator_alloc;
                            params=hsgp_params,
                            corrected_channels=output_channels,
                            estimator_kwargs...,
                        )

                        result = correction_filter(
                            sub.inertial_updated,
                            sub.sim_config_updated,
                            sub.gt_traj_aligned,
                            estimator;
                            step_detector=step_detector_factory(),
                            x_init=sub.x_init,
                            gt_available=gt_available,
                            ref_frame=frame,
                            feature_type=feature_type,
                        )
                    catch e
                        @warn "Skipping (dataset_name=$dataset_name, trial=$trial_id, \
                               budget=$budget, estimator=$est_name)" exception = e
                        n_fail += 1
                    end
                    isnothing(result) && continue

                    zupt, step_seg, corr_traj, io_data, model = result
                    step_seg .+ (s0 - 1) == step_seg_ref[first_k:end] || error(
                        "segmentation moved between runs of trial $trial_id \
                         (budget $budget: $(length(step_seg)) strides from stride $first_k, \
                          against $(n_strides - first_k + 1)): the fixed evaluation \
                          window is not the same window in every cell.")

                    test_ks_run = test_ks .- (first_k - 1)
                    _rmse = rmse(corr_traj[test_ks_run], gt_step[test_ks])[end]
                    _rmse_yaw = rmse_yaw(corr_traj[test_ks_run], gt_step[test_ks])[end]
                    _final = norm(corr_traj.pos[1:2, end] .- gt_step.pos[1:2, end])

                    push!(df, (
                        dataset_name, dataset_order, trial_id,
                        budget, budget_order,
                        est_name, estimator_order,
                        n_strides, k_split, length(test_ks), test_distance,
                        (keep_artifacts ? (zupt, step_seg, corr_traj, io_data, model) :
                         (nothing, nothing, nothing, nothing, nothing))...,
                        _rmse, _rmse / test_distance, _rmse_yaw, _final,
                    ))
                    n_ok += 1
                end
            end
        end
    end

    @info "run_online_learning_curve: $n_ok succeeded, $n_fail failed, $n_short budget(s) \
           skipped as longer than the trial's pre-split prefix"
    return df
end

"The columns that identify one cell of a learning-curve sweep."
const _LC_TRIAL_KEYS = [:dataset_name, :dataset_order, :trial_id,
    :train_strides, :train_strides_order]

"""
    learning_curve_contrast(df; metric=:rmse, reference_estimator="ZUPT only") -> DataFrame

Per-trial change in `metric` against `reference_estimator`, paired within one budget.
The reference is re-run per budget (its covariance depends on the mocap window) and
its spread across budgets is reported.

Returns the trial keys plus `estimator`, `estimator_order`, `value`, `ref_value`,
`delta` and `rel_change_pct`. Negative means the estimator beat the baseline.
"""
function learning_curve_contrast(
    df::DataFrame;
    metric::Symbol=:rmse,
    reference_estimator::AbstractString="ZUPT only",
)::DataFrame
    _require_cols(df, vcat(_LC_TRIAL_KEYS, [:estimator, :estimator_order, metric]),
        "learning_curve_contrast")

    ref_rows = df[df.estimator .== reference_estimator, :]
    isempty(ref_rows) && throw(ArgumentError(
        "learning_curve_contrast: no rows with estimator = \"$reference_estimator\". \
         Available: $(join(unique(df.estimator), ", "))"))

    # How much the reference moves across budgets, per trial: the invariant the fixed
    # evaluation window rests on, reported rather than asserted.
    spread = combine(groupby(ref_rows, [:dataset_name, :trial_id]),
        metric => (v -> (maximum(v) - minimum(v)) / abs(median(v))) => :rel_spread)
    @info "learning_curve_contrast: \"$reference_estimator\" spread across budgets \
           (median $(round(100 * median(spread.rel_spread); digits=2))%, \
           max $(round(100 * maximum(spread.rel_spread); digits=2))% \
           on trial $(spread.trial_id[argmax(spread.rel_spread)]))"

    ref = select(ref_rows, _LC_TRIAL_KEYS, metric => :ref_value)
    test = select(df[df.estimator .!= reference_estimator, :],
        _LC_TRIAL_KEYS, [:estimator, :estimator_order], metric => :value)

    out = innerjoin(test, ref, on=_LC_TRIAL_KEYS)
    isempty(out) && throw(ArgumentError(
        "learning_curve_contrast: no cell has both \"$reference_estimator\" and \
         another estimator — nothing to pair."))

    ok = isfinite.(out.value) .& isfinite.(out.ref_value)
    n_bad = count(!, ok)
    n_bad > 0 && @warn "learning_curve_contrast: dropping $n_bad pair(s) with non-finite metric values"
    out = out[ok, :]

    out.delta = out.value .- out.ref_value
    out.rel_change_pct = 100 .* out.delta ./ abs.(out.ref_value)
    sort!(out, [:dataset_order, :train_strides_order, :estimator_order, :trial_id])
    return out
end
