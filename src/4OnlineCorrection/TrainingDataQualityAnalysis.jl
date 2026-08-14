"""
    training_data_quality_analysis(
        data_dir::AbstractString,
        estimators::AbstractDict{<:AbstractString},
        train_labels::AbstractDict{<:Integer,<:AbstractString},
        test_labels::AbstractDict{<:Integer,<:AbstractString},
        params::HsgpParameters;
        corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw],
        frame::ReferenceFrame=BODY,
        feature_type::FeatureType=THREED_STEP,
        test_tr_ratio::Float64=0.1,
        train_tr_ratio::Float64=1.0
    )::DataFrame

For each estimator in `estimators`, and for each `train_id`, train a corrector on that
track, extract the resulting `(β, Σβ)` via `get_model`, then apply it frozen to each
`test_id` and record horizontal RMSE / RMSE-rate.

The input dictionaries are ordered by iteration. The returned `DataFrame` contains
`estimator_order`, `train_order`, and `test_order` columns so plotting functions can
respect the same order.
"""
function training_data_quality_analysis(
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
    base_estimator::Tuple{<:AbstractString,<:Any}=("ZUPT INS", BaseEstimator),
)::DataFrame

    results = DataFrame(
        estimator=String[],
        estimator_order=Optional{Int}[],
        train_id=Optional{Int}[],
        train_order=Optional{Int}[],
        train_name=Optional{String}[],
        test_id=Int[],
        test_order=Int[],
        test_name=String[],
        rmse=Float64[],
        rmse_rate=Float64[]
    )

    # ---- Pre-align & cache test tracks, preserving test_labels order ----
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

            step_traj = ins_traj_aligned[segs]
            gt_step_traj = gt_traj_aligned[segs]

            _rmse = rmse(step_traj, gt_step_traj)[end]
            _rmse_rate = _rmse / total_distance(gt_step_traj)

            push!(results, (
                base_estimator[1],  # estimator
                nothing,                 # estimator_order
                nothing,                 # train_id
                nothing,                 # train_order
                nothing,                 # train_name
                test_id,                 # test_id
                test_order,              # test_order
                test_name,               # test_name
                _rmse,
                _rmse_rate
            ))
        catch e
            @warn "Skipping test trial $test_id in $data_dir" exception = e
        end
    end

    # ---- Pre-align & cache train tracks ----
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
            @warn "Skipping train trial $train_id in $data_dir" exception = e
        end
    end

    for (estimator_order, (estimator_name, estimator_factory)) in enumerate(estimators)
        for (train_order, (train_id, train_name)) in enumerate(train_labels)
            haskey(train_cache, train_id) || continue

            inertial_train, sim_config_train, gt_traj_train, x_init_train, N_train = train_cache[train_id]

            init_model = nothing
            try
                n_train_cutoff = floor(Int, train_tr_ratio * N_train)
                gt_available_train = [n <= n_train_cutoff for n in 1:N_train]

                estimator_train = estimator_factory(300; params=params, corrected_channels=corrected_channels)
                _, _, _, _, init_model = hybrid_zupt_aided_insv2(
                    inertial_train, sim_config_train, gt_traj_train, estimator_train;
                    x_init=x_init_train, gt_available=gt_available_train,
                    ref_frame=frame, feature_type=feature_type
                )
            catch e
                @warn "Skipping train trial $train_id ($estimator_name) in $data_dir" exception = e
                continue
            end

            for (test_order, (test_id, test_name)) in enumerate(test_labels)
                haskey(test_cache, test_id) || continue

                inertial_updated, sim_config_updated, gt_traj_aligned, x_init, N = test_cache[test_id]
                n_test_cutoff = floor(Int, test_tr_ratio * N)
                gt_available_test = [n <= n_test_cutoff for n in 1:N]

                try
                    estimator_test = estimator_factory(300; params=params, corrected_channels=corrected_channels)
                    _, step_seg, corr_traj, _, _ = hybrid_zupt_aided_insv2(
                        inertial_updated, sim_config_updated, gt_traj_aligned, estimator_test;
                        x_init=x_init, gt_available=gt_available_test,
                        ref_frame=frame, feature_type=feature_type,
                        init_model=init_model
                    )

                    # Truncate the trajectories to get RMSE when gt is unavailable
                    gt_step_traj = gt_traj_aligned[step_seg]
                    N = length(gt_step_traj)
                    n_test_cutoff = floor(Int, test_tr_ratio * N)
                    _rmse = rmse(corr_traj[n_test_cutoff:end], gt_step_traj[n_test_cutoff:end])[end]
                    _rmse_rate = _rmse / total_distance(gt_step_traj[n_test_cutoff:end])

                    push!(results, (
                        estimator_name,
                        estimator_order,
                        train_id,
                        train_order,
                        train_name,
                        test_id,
                        test_order,
                        test_name,
                        _rmse,
                        _rmse_rate
                    ))
                catch e
                    @warn "Skipping (estimator=$estimator_name, train=$train_id, test=$test_id) in $data_dir" exception = e
                end
            end
        end
    end

    return results
end