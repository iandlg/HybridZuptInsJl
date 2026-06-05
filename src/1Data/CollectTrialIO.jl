"""
    collect_trial_io(
        data_dir::AbstractString,
        trial_id::Int;
        frame::ReferenceFrame=BODY,
        feature_type::FeatureType=THREED_STEP
    ) -> Union{Tuple{CorrectionIO, Matrix{Float64}}, Nothing}

Load, align and extract training IO for a single trial.
Returns `nothing` if the trial fails (logs a warning).
"""
function collect_trial_io(
    data_dir::AbstractString,
    trial_id::Int;
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
)::Union{Tuple{CorrectionIO,Matrix{Float64}},Nothing}
    try
        ins_traj, gt_traj, _, segs, _, _ = compute_aligned_ins_trajectory(data_dir, trial_id)
        outputs, features = compute_training_io(ins_traj, gt_traj, segs;
            ref_frame=frame, feature_type=feature_type)
        return outputs, features
    catch e
        @warn "Skipping trial $trial_id in $data_dir" exception = e
        return nothing
    end
end

"""
    collect_dataset(
        data_dir::AbstractString,
        trial_ids::AbstractVector{Int};
        frame::ReferenceFrame=BODY,
        feature_type::FeatureType=THREED_STEP
    ) -> Tuple{CorrectionIO, Matrix{Float64}}

Collect and concatenate training IO across multiple trials.
Failed trials are skipped with a warning.
"""
function collect_dataset(
    data_dir::AbstractString,
    trial_ids::AbstractVector{Int};
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
)::Tuple{CorrectionIO,Matrix{Float64}}

    results = map(trial_ids) do id
        collect_trial_io(data_dir, id; frame=frame, feature_type=feature_type)
    end

    valid = filter(!isnothing, results)
    isempty(valid) && error("No trials loaded successfully from $data_dir")

    n_loaded = length(valid)
    n_total = length(trial_ids)
    n_loaded < n_total && @warn "$( n_total - n_loaded) / $n_total trials failed and were skipped"
    @info "Loaded $n_loaded / $n_total trials"

    outputs_list = first.(valid)
    features_list = last.(valid)

    return outputs_list, features_list
end

"""
    concatenate_io(ios::Vector{CorrectionIO}) -> CorrectionIO

Combine multiple `CorrectionIO` objects into one.
Time stamps are replaced by a simple 1:total_steps index to guarantee
strictly increasing order after concatenation.
All inputs must have the same number of data channels and consistent
`data_std` presence (all must have it or all must lack it).
"""
function concatenate_io(ios::Vector{CorrectionIO})
    isempty(ios) && error("Cannot concatenate an empty vector of CorrectionIO")
    n_chan = size(first(ios).data, 1)
    total_steps = sum(io -> length(io.t), ios)
    t = collect(1.0:total_steps)               # Float64 to match existing

    # Check consistency
    has_std = !isnothing(first(ios).data_std)
    for io in ios
        size(io.data, 1) == n_chan || error("Channel count mismatch in concatenated IO")
        if has_std != !isnothing(io.data_std)
            error("Inconsistent data_std: all CorrectionIO must either have or lack data_std")
        end
    end

    all_data = reduce(hcat, [io.data for io in ios])
    all_std = has_std ? reduce(hcat, [io.data_std for io in ios]) : nothing

    return CorrectionIO(t, all_data, all_std)
end

"""
    split_single_trial(train_frac::Float64, output::CorrectionIO, features::Matrix{Float64},
                      rng::AbstractRNG = Random.default_rng()) ->
                      (train_output, test_output, train_feat, test_feat)

Randomly split one trial into training and test portions.
Returns subsets with time indices sorted to keep temporal order.
"""
function split_single_trial(
    output::CorrectionIO,
    features::Matrix{Float64};
    train_frac::Float64=0.8,
    rng::AbstractRNG=Random.default_rng()
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

    perm = randperm(rng, n_steps)
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
    rng::AbstractRNG=Random.default_rng()
)
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
