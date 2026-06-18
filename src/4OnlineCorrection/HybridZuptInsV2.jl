function hybrid_zupt_aided_insv2(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    corrector::AbstractCorrector;
    step_detector::AbstractSegmentDetector=StepDetector(),
    x_init::Vector{Float64}=zeros(9),
    gt_available::Vector{Bool}=zeros(Bool, length(gt_traj)),
    ref_frame::ReferenceFrame=HEADING,
    feature_type::FeatureType=THREED_STEP
)
    is_compatible(inertial, gt_traj) ||
        throw(ArgumentError("TimeSeries need to be aligned."))

    u = inertial.u
    N = size(u, 2)
    Ts = simdata.Ts
    g_vec = simdata.g isa Float64 ? [0.0, 0.0, simdata.g] : collect(simdata.g)

    zupt, _ = detect_zupt(u, simdata)

    Q, R_meas, H = init_filter(simdata)
    I9 = Matrix{Float64}(I, 9, 9)

    x = zeros(9, N)
    quat = zeros(4, N)
    dx = zeros(9, N)
    dx_timeupd = zeros(9, N)
    dx_smooth = zeros(9, N)
    P = zeros(9, 9, N)
    P_timeupd = zeros(9, 9, N)
    P_smooth = zeros(9, 9, N)
    F_store = zeros(9, 9, N)
    ΔP = zeros(Float64, 9, 9)
    mat66 = zeros(Float64, 6, 6)

    P[1:3, 1:3, 1] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
    P[4:6, 4:6, 1] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
    P[7:9, 7:9, 1] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

    x[:, 1] = x_init
    quat[:, 1] = matrix_to_quat(euler_to_matrix(x_init[7:9]))

    initialize_corrector!(corrector;
        t=inertial.t[1],
        pos_init=x_init[1:3],
        quat_init=quat[:, 1],
        Σpq_init=P[[1:3; 7:9], [1:3; 7:9], 1],
    )

    # n_train_cutoff = floor(Int, train_ratio * N)
    # gt_available = [n <= n_train_cutoff for n in 1:N]
    # @show typeof(gt_available)

    seg_start = 2
    seg_end = N
    step_seg = Int[1]

    io_data = Dict{String,CorrectionIO}(
        "input" => CorrectionIO(FEATURE_DIMS[feature_type], true),
        "target" => CorrectionIO(4, true),
        "prediction" => CorrectionIO(4, true),
    )

    while true
        # ------------------- Step Covariance Reset -------------------------
        ΔP .= 0.0

        # ------------------- ZUPT aided INS Loop -------------------------
        for n in seg_start:seg_end
            x[:, n], quat[:, n] = navigation_equations(
                x[:, n-1], u[:, n], quat[:, n-1], Ts, g_vec)
            F_store[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)

            dx[:, n] = F_store[:, :, n] * dx[:, n-1]
            P[:, :, n] = F_store[:, :, n] * P[:, :, n-1] * F_store[:, :, n]' + G * Q * G'
            ΔP = F_store[:, :, n] * ΔP * F_store[:, :, n]' + G * Q * G'
            dx_timeupd[:, n] = dx[:, n]
            P_timeupd[:, :, n] = P[:, :, n]

            if zupt[n]
                S = H * P[:, :, n] * H' + R_meas
                ΔS = H * ΔP * H' + R_meas
                K = P[:, :, n] * H' / S
                ΔK = ΔP * H' / ΔS
                dx[:, n] = dx[:, n] - K * (dx[4:6, n] - x[4:6, n])
                P[:, :, n] = (I9 - K * H) * P[:, :, n]
                ΔP = (I9 - ΔK * H) * ΔP
            end

            P[:, :, n] = (P[:, :, n] + P[:, :, n]') / 2
            ΔP = (ΔP + ΔP') / 2


            if update!(step_detector, zupt[n])
                push!(step_seg, n)
                seg_end = n
                break
            end
        end

        dx_smooth[:, seg_end] = dx[:, seg_end]
        P_smooth[:, :, seg_end] = P[:, :, seg_end]

        for n in seg_end-1:-1:seg_start
            A = P[:, :, n] * F_store[:, :, n]' / P_timeupd[:, :, n+1]
            dx_smooth[:, n] = dx[:, n] + A * (dx_smooth[:, n+1] - dx_timeupd[:, n+1])
            P_smooth[:, :, n] = P[:, :, n] +
                                A * (P_smooth[:, :, n+1] - P_timeupd[:, :, n+1]) * A'
            P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n]') / 2
        end

        compensate_internal_states!(
            view(x, :, seg_start:seg_end),
            -dx_smooth[:, seg_start:seg_end],
            view(quat, :, seg_start:seg_end)
        )

        dx[:, seg_end] .= 0.0
        P[1:2, 9, seg_end] .= 0.0
        P[9, 1:2, seg_end] .= 0.0

        if seg_end != N
            seg_start = seg_end + 1
            seg_end = N
        else
            break # Break early as to not repeat next part on last index
        end

        prev_step = step_seg[end-1]
        curr_step = step_seg[end]

        # using ΔP means we assume i-1 known
        # means no additive uncertainty p_i and pi-1
        # and no rotation uncertainty q_i-1
        mat66 .= 0.0
        mat66[1:3, 1:3] = quat_to_matrix(quat[:, prev_step])'
        mat66[4:6, 4:6] = Matrix{Float64}(I, 3, 3) # Body frame noise from INS

        @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------ " maxlog = 5
        # Update slow kalman
        dynamic_update!(corrector;
            t=inertial.t[curr_step],
            Δp=mat66[1:3, 1:3] * (x[1:3, curr_step] - x[1:3, prev_step]),
            Δq=quat_multiply(quat_conjugate(quat[:, prev_step]), quat[:, curr_step]),
            Σpq=(mat66 * ΔP[[1:3; 7:9], [1:3; 7:9]] * mat66')
        )

        # Compute stride error and estimated stride 
        stride_err, Σ_err, ins_stride, Σ_ins_stride, R_aug_wl = stride_error(ref_frame;
            R_wb=(quat_to_matrix(corrector.quat[:, corrector.i-1]), quat_to_matrix(corrector.quat[:, corrector.i])),
            Δp=corrector.pos[:, corrector.i] - corrector.pos[:, corrector.i-1],
            Σ_ΔpΔθ3=ΔP[[1:3; 9], [1:3; 9]],
            R_wb_gt=(gt_traj.R_nb[:, :, prev_step], gt_traj.R_nb[:, :, curr_step]),
            Δp_gt=gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step],
            Σ_ΔpΔθ3_gt=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
        )

        # Compute feature
        feature, Σ_feature = compute_feature(feature_type;
            ins_stride=ins_stride, Σ_ins_stride=Σ_ins_stride,
            ΔT=inertial.t[curr_step] - inertial.t[prev_step]
        )

        # Save for plotting
        append_io!(io_data["target"], inertial.t[prev_step], stride_err, sqrt.(diag(Σ_err)))
        append_io!(io_data["input"], inertial.t[prev_step], feature, sqrt.(diag(Σ_feature)))

        if gt_available[curr_step] && gt_available[prev_step]
            # @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------"
            # @info "Prev and curr GT available"
            stride_measurement_update!(corrector;
                feature_type=feature_type,
                stride_err=stride_err, Σ_err=Σ_err,
                feature=feature, Σ_feature=Σ_feature, R_aug_wl=R_aug_wl,
            )
            relinearize!(corrector)

            posyaw_measurement_update!(corrector;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
            )

        elseif gt_available[curr_step]
            # @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------"
            # @info "Curr GT available"
            posyaw_measurement_update!(corrector;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
            )
        else
            # @info "No GT available"
            pred, var_pred = learned_measurement_update!(corrector;
                feature_type=feature_type,
                feature=feature, Σ_feature=Σ_feature, R_aug_wl=R_aug_wl)
            if !isnothing(pred) && !isnothing(var_pred)
                append_io!(io_data["prediction"], inertial.t[prev_step], pred, sqrt.(var_pred))
            end

        end
        relinearize!(corrector)
    end

    return zupt, step_seg, get_trajectory(corrector), io_data
end


"""
    collect_trial_io_online(data_dir, trial_id; frame=BODY, feature_type=THREED_STEP)::Optional{Tuple{CorrectionIO, Matrix{Float64}}}

Load, align and extract training IO for a single trial.
Returns `nothing` if the trial fails (logs a warning).
"""
function collect_trial_io_online(
    data_dir::AbstractString,
    trial_id::Int;
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
)::Union{Tuple{CorrectionIO,CorrectionIO,Trajectory,Trajectory,Vector{Int}},Nothing}
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

        gt_available = [n <= 0 for n in 1:N]
        default_corr = HybridZuptInsJl.DefaultCorrector(round(Int, N / 60))
        _, step_seg, def_corr_traj, output_data = HybridZuptInsJl.hybrid_zupt_aided_insv2(
            inertial_updated, sim_config_updated, gt_traj_aligned, default_corr;
            x_init=x_init, gt_available=gt_available, ref_frame=frame, feature_type=feature_type)
        return output_data["target"], output_data["input"], def_corr_traj, gt_traj_aligned, step_seg
    catch e
        @warn "Skipping trial $trial_id in $data_dir" exception = e
        return nothing
    end
end

function collect_dataset(
    data_dir::AbstractString,
    trial_ids::AbstractVector{Int};
    frame::ReferenceFrame=BODY,
    feature_type::FeatureType=THREED_STEP
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

"""
    to_dataframe(valid::Vector) -> DataFrame

Collect every raw time-series sample for each input/output channel across all trials.
Returns a long-form DataFrame with columns:
  `trial_id`  - trial index (1-based)
  `io_type`   - `:input` or `:output`
  `channel`   - channel index
  `t`         - timestamp of the sample (from `CorrectionIO.t`)
  `value`     - individual sample value
"""
function to_dataframe(
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

# """
#     from_dataframe(df)::CorrectionIO

# Takes df from a single trial and outputs input output CorrectionIO objects
# """
# function from_dataframe(df::Dataframe)::CorrectionIO

# end