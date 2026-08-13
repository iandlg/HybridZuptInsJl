"""
    training_data_quality_analysis(
        data_dir::AbstractString,
        train_ids::Vector{Int},
        test_ids::Vector{Int},
        params::HsgpParameters;
        hsgp_estimator_factories::Vector{<:Type}=[JointHsgpEstimator],
        corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw],
        frame::ReferenceFrame=BODY,
        feature_type::FeatureType=THREED_STEP,
        train_labels::Union{Nothing,AbstractDict{Int,String}}=nothing,
        test_labels::Union{Nothing,AbstractDict{Int,String}}=nothing
    )::DataFrame

For each estimator factory in `hsgp_estimator_factories`, and for each `train_id`,
train a corrector on that track (ground truth available throughout), extract the
resulting `(β, Σβ)` via `get_model`, then apply it *frozen* (no further learning —
`gt_available` all `false`) to each `test_id` and record the horizontal RMSE /
RMSE-rate on that test track.

# Returns
`DataFrame` with columns
`estimator, train_id, train_name, test_id, test_name, rmse, rmse_rate`.
"""
function training_data_quality_analysis(
    data_dir::AbstractString,
    train_ids::Vector{Int},
    test_ids::Vector{Int},
    params::HsgpParameters;
    hsgp_estimator_factories::Vector{<:Type}=[JointHsgpEstimator],
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw],
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP,
    train_labels::Union{Nothing,AbstractDict{Int,String}}=nothing,
    test_labels::Union{Nothing,AbstractDict{Int,String}}=nothing
)::DataFrame

    results = DataFrame(
        estimator=String[],
        train_id=Int[], train_name=String[],
        test_id=Int[], test_name=String[],
        rmse=Float64[], rmse_rate=Float64[]
    )

    # ---- Pre-align & cache test tracks (reused across every estimator/train_id) ----
    test_cache = Dict{Int,Tuple}()
    for tid in test_ids
        try
            ins_traj_aligned, gt_traj_aligned, _, segs, inertial_updated, sim_config_updated =
                compute_aligned_ins_trajectory(data_dir, tid)
            x_init = vcat(
                ins_traj_aligned.pos[:, 1],
                ins_traj_aligned.vel[:, 1],
                matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
            )
            N = length(inertial_updated)
            test_cache[tid] = (inertial_updated, sim_config_updated, gt_traj_aligned, x_init, N)


            step_traj = ins_traj_aligned[segs]
            gt_step_traj = gt_traj_aligned[segs]

            _rmse = rmse(step_traj, gt_step_traj)[end]
            _rmse_rate = _rmse / total_distance(gt_step_traj)

            push!(results, ("ZUPT INS", 0, train_name, test_id, test_name, _rmse, _rmse_rate))

        catch e
            @warn "Skipping test trial $tid in $data_dir" exception = e
        end
    end

    # ---- Pre-align & cache train tracks (reused across every estimator) ----
    train_cache = Dict{Int,Tuple}()
    for train_id in train_ids
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

    for hsgp_estimator_factory in hsgp_estimator_factories
        estimator_name = string(hsgp_estimator_factory)

        for train_id in train_ids
            haskey(train_cache, train_id) || continue
            train_name = isnothing(train_labels) ? "trial_$train_id" : get(train_labels, train_id, "trial_$train_id")
            inertial_train, sim_config_train, gt_traj_train, x_init_train, N_train = train_cache[train_id]

            β_Σβ_0 = nothing
            try
                gt_available_train = fill(true, N_train)
                estimator_train = hsgp_estimator_factory(300, params; corrected_channels=corrected_channels)
                _, _, _, _, β_Σβ_0 = hybrid_zupt_aided_insv2(
                    inertial_train, sim_config_train, gt_traj_train, estimator_train;
                    x_init=x_init_train, gt_available=gt_available_train,
                    ref_frame=frame, feature_type=feature_type
                )
            catch e
                @warn "Skipping train trial $train_id ($estimator_name) in $data_dir" exception = e
                continue
            end

            for test_id in test_ids
                haskey(test_cache, test_id) || continue
                test_name = isnothing(test_labels) ? "trial_$test_id" : get(test_labels, test_id, "trial_$test_id")
                inertial_updated, sim_config_updated, gt_traj_aligned, x_init, N = test_cache[test_id]
                gt_available_test = fill(false, N)

                try
                    estimator_test = hsgp_estimator_factory(300, params; corrected_channels=corrected_channels)
                    _, step_seg, corr_traj, _, _ = hybrid_zupt_aided_insv2(
                        inertial_updated, sim_config_updated, gt_traj_aligned, estimator_test;
                        x_init=x_init, gt_available=gt_available_test,
                        ref_frame=frame, feature_type=feature_type,
                        β_Σβ_0=β_Σβ_0
                    )

                    gt_step_traj = gt_traj_aligned[step_seg]
                    _rmse = rmse(corr_traj, gt_step_traj)[end]
                    _rmse_rate = _rmse / total_distance(gt_step_traj)

                    push!(results, (estimator_name, train_id, train_name, test_id, test_name, _rmse, _rmse_rate))
                catch e
                    @warn "Skipping (estimator=$estimator_name, train=$train_id, test=$test_id) in $data_dir" exception = e
                end
            end
        end
    end

    return results
end