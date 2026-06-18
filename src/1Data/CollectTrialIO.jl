"""
    collect_trial_io(
        data_dir::AbstractString,
        trial_id::Int;
        frame::ReferenceFrame=BODY,
        feature_type::FeatureType=THREED_STEP
    )::Union{Tuple{CorrectionIO, Matrix{Float64}}, Nothing}

Load, align and extract training IO for a single trial.
Returns `nothing` if the trial fails (logs a warning).
"""
function collect_trial_io(
    data_dir::AbstractString,
    trial_id::Int;
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
)::Union{Tuple{CorrectionIO,Matrix{Float64},Trajectory,Trajectory,Vector{Int}},Nothing}
    try
        ins_traj, gt_traj, _, segs, _, _ = compute_aligned_ins_trajectory(data_dir, trial_id)
        outputs, features = compute_training_io(ins_traj, gt_traj, segs;
            ref_frame=frame, feature_type=feature_type)
        return outputs, features, ins_traj, gt_traj, segs
    catch e
        @warn "Skipping trial $trial_id in $data_dir" exception = e
        return nothing
    end
end

# """
#     collect_dataset(
#         data_dir, trial_ids; frame=BODY, feature_type=THREED_STEP)::Vector{Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}}}

# Collect and concatenate training IO across multiple trials.
# Failed trials are skipped with a warning.
# """
# function collect_dataset(
#     data_dir::AbstractString,
#     trial_ids::AbstractVector{Int};
#     frame::ReferenceFrame=BODY,
#     feature_type::FeatureType=THREED_STEP
# )::Vector{Tuple{CorrectionIO,Matrix{Float64},Trajectory,Trajectory,Vector{Int}}}

#     results = map(trial_ids) do id
#         collect_trial_io(data_dir, id; frame=frame, feature_type=feature_type)
#     end

#     valid = filter(!isnothing, results)
#     isempty(valid) && error("No trials loaded successfully from $data_dir")

#     n_loaded = length(valid)
#     n_total = length(trial_ids)
#     n_loaded < n_total && @warn "$( n_total - n_loaded) / $n_total trials failed and were skipped"
#     @info "Loaded $n_loaded / $n_total trials"
#     return valid
# end

"""
    split_single_trial(
        train_frac::Float64,
        output::CorrectionIO,
        features::Matrix{Float64},
        rng::AbstractRNG = Random.default_rng()
    )::(train_output, test_output, train_feat, test_feat)

Randomly split one trial into training and test portions.
Returns subsets with time indices sorted to keep temporal order.
"""
function split_single_trial(
    output::CorrectionIO,
    features::Matrix{Float64};
    train_frac::Float64=0.8,
    rng::Random.AbstractRNG=Random.Xoshiro(123)
)
    n_steps = length(output.t)
    n_train = round(Int, n_steps * train_frac)
    # Ensure at least one sample in each set (if trial has ≥2 steps)
    if n_steps >= 2
        n_train = clamp(n_train, 1, n_steps - 1)
    else
        # Single step trial: put it entirely in training (or entirely in test)?
        # Putting in training is safer; will leave test empty for this trial.
        n_train = n_steps
    end

    perm = Random.randperm(rng, n_steps)
    train_idx = sort(perm[1:n_train])
    test_idx = sort(perm[n_train+1:end])

    train_output = output[train_idx]
    test_output = output[test_idx]
    train_feat = features[:, train_idx]
    test_feat = features[:, test_idx]

    return train_output, test_output, train_feat, test_feat
end

"""
    split_dataset_train_test(
        outputs::Vector{CorrectionIO},
        features::Vector{Matrix{Float64}};
        train_frac::Float64 = 0.8,
        rng::AbstractRNG = Random.default_rng()
    ) -> (train_output, test_output, train_features, test_features)

Split each trial randomly according to `train_frac` and then concatenate
all training portions into a single `CorrectionIO` / feature matrix,
and similarly for test portions.

Returns four objects:
- `train_output` : `CorrectionIO`
- `test_output`  : `CorrectionIO`
- `train_features`: `Matrix{Float64}` (n_features × N_train)
- `test_features` : `Matrix{Float64}` (n_features × N_test)
"""
function split_dataset_train_test(
    outputs::Vector{CorrectionIO},
    features::Vector{Matrix{Float64}};
    train_frac::Float64=0.8,
    rng::Random.AbstractRNG=Random.Xoshiro(123)
)::Tuple{CorrectionIO,CorrectionIO,Matrix{Float64},Matrix{Float64}}

    @assert length(outputs) == length(features) "Outputs and features must have the same number of trials"

    train_output_parts = CorrectionIO[]
    test_output_parts = CorrectionIO[]
    train_feat_parts = Matrix{Float64}[]
    test_feat_parts = Matrix{Float64}[]

    for (out_io, feat) in zip(outputs, features)
        t_out, v_out, t_feat, v_feat = split_single_trial(
            out_io, feat; train_frac=train_frac, rng=rng
        )
        push!(train_output_parts, t_out)
        push!(test_output_parts, v_out)
        push!(train_feat_parts, t_feat)
        push!(test_feat_parts, v_feat)
    end

    # Concatenate all parts
    train_output = concatenate_io(train_output_parts)
    test_output = concatenate_io(test_output_parts)

    train_features = reduce(hcat, train_feat_parts)
    test_features = reduce(hcat, test_feat_parts)

    return train_output, test_output, train_features, test_features
end
