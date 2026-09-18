"""
    multi_track_training_analysis(data_dir, estimators, train_labels, test_labels, params;
                                  order_seeds=[1], noise_spec=nothing, kwargs...) -> DataFrame

Accumulate the training tracks one at a time, re-testing the frozen model on every test
track after each addition, for every estimator in `estimators`. The question the output is
built to answer is whether *more* (noisy) training data buys back the performance the noise
costs, so the quantity of interest is the trend against `train_set_order` — the number of
tracks accumulated — with the untrained `"Base"` rows as the reference.

The whole accumulation is repeated once per seed in `order_seeds`, and repeat `s` walks its
own random permutation `shuffle(Xoshiro(s), ids)` of the training tracks. WAS: the tracks
were accumulated in the declared order of `train_labels`, one arbitrary permutation, so the
curve against training-set size also encoded which track happened to come next. That order
is now drawn, not chosen: `train_labels` defines the *set* of training tracks and their
declared order no longer affects any result.

The training-GT noise is deliberately *not* redrawn per repeat. A track's realisation comes
from `Xoshiro(1000 * train_id)`, keyed on the track and nothing else, which has two
consequences worth keeping:

- every estimator, and every repeat, trains on bit-identical noisy tracks, so estimators
  stay paired and the only thing varying between repeats is the order;
- at the final step, where all repeats have trained on the same set, any remaining spread is
  order-sensitivity in the fit alone. That is the check that the shuffling measures what it
  should. The flip side is that the figure carries one noise realisation per track, so it
  marginalises over order but not over noise.

Rows carry `seed` (which repeat), `train_set_order` (how many tracks had been accumulated)
and `train_set`/`train_ids` (that repeat's ordered ids — the only record of the permutation
it drew). Baseline rows (`train_set == "Base"`) are untrained, hence noise- and
order-independent: they are computed once and carry `seed === missing`. Their estimator
name defaults to `"ZUPT only"` so it keys into `_METHOD_COLOR_INDICES`
(`Plotting/OfflineCorrection.jl`) and matches the other Section 5 figures; rename it and
the baseline silently drops to the fallback grey.
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
    # Filter that runs the correction: `hybrid_zupt_aided_insv2` (absolute-state
    # update) or `hybrid_zupt_aided_insv3` (stride-level, notes/013-014).
    correction_filter::Function=hybrid_zupt_aided_insv2,
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
            ins_traj_aligned, gt_traj_aligned, _, segs, inertial_updated, sim_config_updated =
                compute_aligned_ins_trajectory(data_dir, test_id)

            x_init = vcat(
                ins_traj_aligned.pos[:, 1],
                ins_traj_aligned.vel[:, 1],
                matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
            )
            N = length(inertial_updated)
            test_cache[test_id] = (inertial_updated, sim_config_updated, gt_traj_aligned, x_init, N)

            # Compute base RMSE (without training)
            step_traj = ins_traj_aligned[segs]
            gt_step_traj = gt_traj_aligned[segs]
            n_test_cutoff = floor(Int, test_tr_ratio * length(gt_step_traj))
            _rmse = rmse(step_traj[n_test_cutoff:end], gt_step_traj[n_test_cutoff:end])[end]
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

                # Retrieve cached train data
                inertial_train, sim_config_train, gt_traj_train, x_init_train, N_train = train_cache[train_id]

                # Prepare noisy ground truth if requested. Seeded by `train_id` alone: not by
                # the estimator, so the estimators stay paired, and not by the repeat or by
                # the track's position in the permutation, so a track's noise does not change
                # when the order does.
                gt_for_training = if isnothing(noise_spec)
                    gt_traj_train
                else
                    add_gaussian_noise(
                        gt_traj_train;
                        pos_std=noise_spec.pos_std,
                        pos_bias=noise_spec.pos_bias,
                        att_std=noise_spec.att_std,
                        att_bias=noise_spec.att_bias,
                        rng=Random.Xoshiro(1000 * train_id)
                    )
                end

                # Training mask – only first `train_tr_ratio` fraction has GT
                n_train_cutoff = floor(Int, train_tr_ratio * N_train)
                gt_available_train = [n <= n_train_cutoff for n in 1:N_train]

                # Train on this track, continuing from previous model
                try
                    estimator_train = estimator_factory(300; params=params, corrected_channels=corrected_channels)
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
                    n_test_cutoff = floor(Int, test_tr_ratio * N_test)
                    gt_available_test = [n <= n_test_cutoff for n in 1:N_test]

                    try
                        estimator_test = estimator_factory(300; params=params, corrected_channels=corrected_channels)
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
                        n_test_cutoff_local = floor(Int, test_tr_ratio * N)
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
