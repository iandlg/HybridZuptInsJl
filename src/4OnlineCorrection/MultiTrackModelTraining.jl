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
    base_estimator::Tuple{<:AbstractString,<:Any}=("ZUPT INS", BaseEstimator),
    rng::Random.AbstractRNG=Xoshiro(123)
)::DataFrame

    # Results container
    results = DataFrame(
        estimator=String[],
        estimator_order=Union{Int,Missing}[],
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
                base_estimator[1],
                missing,          # no estimator order
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
    for (estimator_order, (estimator_name, estimator_factory)) in enumerate(estimators)
        init_model = nothing          # cumulative model
        used_train_ids = Int[]        # tracks used so far
        train_set_step = 1

        for (train_id, train_name) in train_labels
            # If this train track was not cached, skip it – but we cannot continue accumulating?
            if !haskey(train_cache, train_id)
                @warn "Train trial $train_id not cached; stopping incremental training for estimator $estimator_name"
                break
            end

            # Retrieve cached train data
            inertial_train, sim_config_train, gt_traj_train, x_init_train, N_train = train_cache[train_id]

            # Prepare noisy ground truth if requested
            gt_for_training = if isnothing(noise_spec)
                gt_traj_train
            else
                add_gaussian_noise(
                    gt_traj_train;
                    pos_std=noise_spec.pos_std,
                    pos_bias=noise_spec.pos_bias,
                    att_std=noise_spec.att_std,
                    att_bias=noise_spec.att_bias,
                    rng=rng
                )
            end

            # Training mask – only first `train_tr_ratio` fraction has GT
            n_train_cutoff = floor(Int, train_tr_ratio * N_train)
            gt_available_train = [n <= n_train_cutoff for n in 1:N_train]

            # Train on this track, continuing from previous model
            try
                estimator_train = estimator_factory(300; params=params, corrected_channels=corrected_channels)
                _, _, _, _, init_model = hybrid_zupt_aided_insv2(
                    inertial_train, sim_config_train, gt_for_training, estimator_train;
                    x_init=x_init_train,
                    gt_available=gt_available_train,
                    ref_frame=frame,
                    feature_type=feature_type,
                    init_model=init_model
                )
            catch e
                @warn "Training failed on train $train_id for estimator $estimator_name" exception=e
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
                    _, step_seg, corr_traj, _, _ = hybrid_zupt_aided_insv2(
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
                    @warn "Testing failed (estimator=$estimator_name, train_set=$train_set_str, test=$test_id)" exception=e
                end
            end

            train_set_step += 1
        end
    end

    return results
end