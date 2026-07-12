
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
    corrector::AbstractCorrector=DefaultCorrector(300)
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
    correctors::AbstractDict{<:AbstractString,<:AbstractCorrector};
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
    valid::Dict{Int,Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}
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