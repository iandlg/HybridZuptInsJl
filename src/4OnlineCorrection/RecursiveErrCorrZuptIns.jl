# ── EstimatorLoop.jl ────────────────────────────────────────────────────────
function run_recusive_zupt_ins(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    estimator::AbstractEstimator;
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
    Q, R_meas, _H = init_filter(simdata)

    quat_init = matrix_to_quat(euler_to_matrix(x_init[7:9]))
    P_init = zeros(9, 9)
    P_init[1:3, 1:3] = Diagonal(sigma_initial_pos_array(simdata) .^ 2)
    P_init[4:6, 4:6] = Diagonal(sigma_initial_vel_array(simdata) .^ 2)
    P_init[7:9, 7:9] = Diagonal(sigma_initial_att_array(simdata) .^ 2)

    initialize_estimator!(estimator;
        t=inertial.t[1], x_init=x_init, quat_init=quat_init, P_init=P_init)

    seg_start = 2
    seg_end = N
    step_seg = Int[1]

    io_data = Dict{String,CorrectionIO}(
        "input" => CorrectionIO(FEATURE_DIMS[feature_type], true),
        "target" => CorrectionIO(4, true),
        "prediction" => CorrectionIO(4, true),
    )

    while true
        # ------------------- ZUPT aided INS Loop -------------------------
        for n in seg_start:seg_end
            time_update!(estimator; t=inertial.t[n], u=u[:, n], Ts=Ts, g_vec=g_vec, Q=Q)

            if zupt[n]
                zupt_measurement_update!(estimator; R_meas=R_meas)
            end

            if update!(step_detector, zupt[n])
                push!(step_seg, n)
                seg_end = n
                break
            end
        end

        # ------------------- RTS smoothing over the segment ---------------
        smooth_segment!(estimator; seg_start=seg_start, seg_end=seg_end)

        if seg_end != N
            seg_start = seg_end + 1
            seg_end = N
        else
            break   # break early, mirrors old early exit before the stride block
        end

        prev_step = step_seg[end-1]
        curr_step = step_seg[end]

        @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------" maxlog = 5

        # --- Compute stride error & covariance directly from estimator state ---
        stride_err, Σ_err, ins_stride, Σ_ins_stride, R_aug_wl = stride_error(ref_frame;
            R_wb=(get_quat_matrix(estimator, estimator.i - 1), get_quat_matrix(estimator, estimator.i)),
            Δp=get_pos(estimator, estimator.i) - get_pos(estimator, estimator.i - 1),
            Σ_ΔpΔθ3=get_posatt_cov(estimator),
            R_wb_gt=(gt_traj.R_nb[:, :, prev_step], gt_traj.R_nb[:, :, curr_step]),
            Δp_gt=gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step],
            Σ_ΔpΔθ3_gt=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
        )

        feature, Σ_feature = compute_feature(feature_type;
            ins_stride=ins_stride, Σ_ins_stride=Σ_ins_stride,
            ΔT=inertial.t[curr_step] - inertial.t[prev_step]
        )

        append_io!(io_data["target"], inertial.t[prev_step], stride_err, sqrt.(diag(Σ_err)))
        append_io!(io_data["input"], inertial.t[prev_step], feature, sqrt.(diag(Σ_feature)))

        if gt_available[curr_step] && gt_available[prev_step]
            stride_measurement_update!(estimator;
                feature_type=feature_type,
                stride_err=stride_err, Σ_err=Σ_err,
                feature=feature, Σ_feature=Σ_feature, R_aug_wl=R_aug_wl)
            relinearize!(estimator)

            posyaw_measurement_update!(estimator;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2))

        elseif gt_available[curr_step]
            posyaw_measurement_update!(estimator;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2))

        else
            pred, var_pred = learned_measurement_update!(estimator;
                feature_type=feature_type,
                feature=feature, Σ_feature=Σ_feature, R_aug_wl=R_aug_wl)
            if !isnothing(pred) && !isnothing(var_pred)
                append_io!(io_data["prediction"], inertial.t[prev_step], pred, sqrt.(var_pred))
            end
        end
        relinearize!(estimator)
    end

    return zupt, step_seg, get_trajectory(estimator), io_data
end