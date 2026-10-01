"""
    multi_track_training_analysis(data_dir, estimators, train_labels, test_labels, params;
                                  order_seeds=[1], noise_spec=nothing, kwargs...) -> DataFrame

Add training tracks one at a time and, after each, test the frozen model on every test
track, for every estimator: does more (noisy) training data buy back what the noise costs?

- Each seed in `order_seeds` accumulates its own permutation `shuffle(Xoshiro(s), ids)`;
  `train_labels` only defines the set.
- A track's training-GT noise comes from `Xoshiro(1000 * train_id)`, identical across
  estimators and repeats, and the training runs get the matching R
  ([`matched_gt_sigma_config`](@ref)). Test tracks keep clean ground truth and R.
- `test_tr_ratio`: mocap fraction on the test walks (`0` = start pose only).

Rows carry `seed`, `train_set_order` (tracks accumulated) and `train_set`/`train_ids`.
Baseline rows (`train_set == "Base"`, `seed === missing`) are `BaseEstimator` through the
same `correction_filter`, once per test track. Their name defaults to `"ZUPT only"`,
which keys its colour in `_METHOD_COLOR_INDICES`; renaming it turns the baseline grey.
"""
function multi_track_training_analysis(
    data_dir::AbstractString,
    estimators::AbstractDict{<:AbstractString,<:Any},
    train_labels::AbstractDict{<:Integer,<:AbstractString},
    test_labels::AbstractDict{<:Integer,<:AbstractString},
    params::HsgpParameters;
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw],
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
    test_tr_ratio::Float64=0.1,
    train_tr_ratio::Float64=1.0,
    noise_spec::Union{Nothing,NoiseSpec}=nothing,
    order_seeds::AbstractVector{Int}=[1],
    base_estimator_name::AbstractString="ZUPT only",
    estimator_kwargs::NamedTuple=(;),
    correction_filter::Function=hybrid_zupt_aided_insv4,
)::DataFrame

    isempty(order_seeds) && throw(ArgumentError("order_seeds must not be empty"))
    allunique(order_seeds) || throw(ArgumentError("order_seeds must be unique, got $order_seeds"))

    # Results container
    results = DataFrame(
        estimator=String[],
        estimator_order=Union{Int,Missing}[],
        seed=Union{Int,Missing}[],   # which repeat; `missing` on the untrained baseline
        train_set=String[],          # e.g. "6" or "6,3" or "Base"
        train_set_order=Union{Int,Missing}[],
        train_ids=Union{String,Missing}[],  # comma‑separated list or missing
        test_id=Int[],
        test_order=Int[],
        test_name=String[],
        rmse=Float64[],
        rmse_rate=Float64[]
    )

    # ---------- Pre‑cache test tracks ----------
    test_cache = Dict{Int,Tuple}()
    for (test_order, (test_id, test_name)) in enumerate(test_labels)
        try
            ins_traj_aligned, gt_traj_aligned, _, _, inertial_updated, sim_config_updated =
                compute_aligned_ins_trajectory(data_dir, test_id)

            x_init = vcat(
                ins_traj_aligned.pos[:, 1],
                ins_traj_aligned.vel[:, 1],
                matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
            )
            N = length(inertial_updated)
            test_cache[test_id] = (inertial_updated, sim_config_updated, gt_traj_aligned, x_init, N)

            # The untrained baseline, through the SAME filter as every trained row:
            # `BaseEstimator` propagates the raw stride, so this is that filter's own
            # uncorrected run and the only difference from a trained row is the
            # correction model. Everything else below mirrors the test path in the main
            # loop exactly -- same `gt_available` mask, same clean ground truth, same
            # scoring on the filter's own `step_seg`.
            gt_available_base = [n <= max(0, floor(Int, test_tr_ratio * N)) for n in 1:N]
            estimator_base = BaseEstimator(300; params=params, corrected_channels=corrected_channels)
            _, step_seg, corr_traj, _, _ = correction_filter(
                inertial_updated, sim_config_updated, gt_traj_aligned, estimator_base;
                x_init=x_init,
                gt_available=gt_available_base,
                ref_frame=frame,
                feature_type=feature_type,
            )

            gt_step_traj = gt_traj_aligned[step_seg]
            n_test_cutoff = max(1, floor(Int, test_tr_ratio * length(gt_step_traj)))
            _rmse = rmse(corr_traj[n_test_cutoff:end], gt_step_traj[n_test_cutoff:end])[end]
            _rmse_rate = _rmse / total_distance(gt_step_traj[n_test_cutoff:end])

            push!(results, (
                base_estimator_name,
                missing,          # no estimator order
                missing,          # no seed: untrained, so noise- and order-independent
                "Base",           # train_set
                0,                # train_set_order
                missing,          # train_ids
                test_id,
                test_order,
                test_name,
                _rmse,
                _rmse_rate
            ))
        catch e
            @warn "Skipping test trial $test_id in $data_dir" exception=e
        end
    end

    # ---------- Pre‑cache train tracks ----------
    # The corrupted ground truth and the R that goes with it are built here, once per
    # track: the noise is keyed on `train_id` alone, so neither depends on the repeat,
    # the estimator or the track's place in the permutation. What the cache holds is
    # therefore exactly what the training runs are handed.
    train_cache = Dict{Int,Tuple}()
    for (train_id, _) in train_labels
        try
            ins_traj_aligned, gt_traj_aligned, _, _, inertial_updated, sim_config_updated =
                compute_aligned_ins_trajectory(data_dir, train_id)

            x_init = vcat(
                ins_traj_aligned.pos[:, 1],
                ins_traj_aligned.vel[:, 1],
                matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
            )
            N_train = length(inertial_updated)

            # Seeded by `train_id` alone: not by the estimator, so the estimators stay
            # paired, and not by the repeat or by the track's position in the
            # permutation, so a track's noise does not change when the order does.
            if !isnothing(noise_spec)
                gt_traj_aligned = add_gaussian_noise(
                    gt_traj_aligned;
                    pos_std=noise_spec.pos_std,
                    pos_bias=noise_spec.pos_bias,
                    att_std=noise_spec.att_std,
                    att_bias=noise_spec.att_bias,
                    rng=Random.Xoshiro(1000 * train_id)
                )
                # ... and the filter is told about it, rather than being handed an R
                # that still describes the clean mocap.
                sim_config_updated = matched_gt_sigma_config(sim_config_updated, noise_spec)
            end

            train_cache[train_id] = (inertial_updated, sim_config_updated, gt_traj_aligned, x_init, N_train)
        catch e
            @warn "Skipping train trial $train_id in $data_dir" exception=e
        end
    end

    # ---------- Main loops ----------
    # One repeat per seed. The seed fixes the accumulation order and nothing else, so every
    # estimator in a repeat sees the same tracks in the same order with the same noise.
    declared_train_ids = collect(keys(train_labels))

    for seed in order_seeds
        train_order = Random.shuffle(Random.Xoshiro(seed), declared_train_ids)

        for (estimator_order, (estimator_name, estimator_factory)) in enumerate(estimators)
            init_model = nothing          # cumulative model
            used_train_ids = Int[]        # tracks used so far
            train_set_step = 1

            for train_id in train_order
                # If this train track was not cached, skip it – but we cannot continue accumulating?
                if !haskey(train_cache, train_id)
                    @warn "Train trial $train_id not cached; stopping incremental training for estimator $estimator_name"
                    break
                end

                # Retrieve cached train data: the ground truth is already corrupted and
                # the config already carries the matching R (see the pre-cache above).
                inertial_train, sim_config_train, gt_for_training, x_init_train, N_train = train_cache[train_id]

                # Training mask – only first `train_tr_ratio` fraction has GT
                n_train_cutoff = floor(Int, train_tr_ratio * N_train)
                gt_available_train = [n <= n_train_cutoff for n in 1:N_train]

                # Train on this track, continuing from previous model
                try
                    estimator_train = estimator_factory(300; params=params, corrected_channels=corrected_channels, estimator_kwargs...)
                    _, _, _, _, init_model = correction_filter(
                        inertial_train, sim_config_train, gt_for_training, estimator_train;
                        x_init=x_init_train,
                        gt_available=gt_available_train,
                        ref_frame=frame,
                        feature_type=feature_type,
                        init_model=init_model
                    )
                catch e
                    @warn "Training failed on train $train_id for estimator $estimator_name (seed $seed)" exception=e
                    # Stop adding more tracks for this estimator
                    break
                end

                # Add this track to the set of used training IDs
                push!(used_train_ids, train_id)
                train_set_str = join(string.(used_train_ids), ",")

                # Test on all test tracks with the updated model
                for (test_order, (test_id, test_name)) in enumerate(test_labels)
                    if !haskey(test_cache, test_id)
                        continue
                    end

                    inertial_test, sim_config_test, gt_traj_test, x_init_test, N_test = test_cache[test_id]
                    n_test_cutoff = max(1, floor(Int, test_tr_ratio * N_test))
                    gt_available_test = [n <= n_test_cutoff-1 for n in 1:N_test]

                    try
                        estimator_test = estimator_factory(300; params=params, corrected_channels=corrected_channels, estimator_kwargs...)
                        _, step_seg, corr_traj, _, _ = correction_filter(
                            inertial_test, sim_config_test, gt_traj_test, estimator_test;
                            x_init=x_init_test,
                            gt_available=gt_available_test,
                            ref_frame=frame,
                            feature_type=feature_type,
                            init_model=init_model
                        )

                        # Compute RMSE on the part where GT is not available
                        gt_step_traj = gt_traj_test[step_seg]
                        N = length(gt_step_traj)
                        n_test_cutoff_local = max(1, floor(Int, test_tr_ratio * N))
                        _rmse = rmse(corr_traj[n_test_cutoff_local:end], gt_step_traj[n_test_cutoff_local:end])[end]
                        _rmse_rate = _rmse / total_distance(gt_step_traj[n_test_cutoff_local:end])

                        push!(results, (
                            estimator_name,
                            estimator_order,
                            seed,
                            train_set_str,
                            train_set_step,
                            train_set_str,   # or we could keep a separate column for IDs
                            test_id,
                            test_order,
                            test_name,
                            _rmse,
                            _rmse_rate
                        ))
                    catch e
                        @warn "Testing failed (estimator=$estimator_name, seed=$seed, train_set=$train_set_str, test=$test_id)" exception=e
                    end
                end

                train_set_step += 1
            end
        end
    end

    return results
end
