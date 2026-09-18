"""
    hybrid_zupt_aided_insv4(inertial, simdata, gt_traj, corrector; ...)

`hybrid_zupt_aided_insv3` with the stride model moved into the corrector's
state (notes/015). Same signature, same return tuple, same `io_data` keys.

V3 learns the stride error in a separate filter and hands its prediction to the
corrector as per-stride process noise, so the model's uncertainty is white from
one stride to the next. Here `β` is part of the corrector's error state and the
correction `y = y₀ + Φ(z) β` is a term of the propagation, so:

- there is one model and it is the same in both halves; ground truth only adds
  a position/yaw measurement, which learns `β` through the cross-covariance the
  propagation builds. No stride measurement, which would count the same mocap
  twice;
- in the test half `β`'s uncertainty accumulates as a correlated bias.

`corrector` is a `AbstractJointStrideEstimator`, or any other estimator, which
then propagates the raw stride (the "ZUPT only" baseline). The target and the
feature are built from the inner INS only, as in V3 (notes/014).
"""
function hybrid_zupt_aided_insv4(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    corrector::AbstractEstimator;
    step_detector::AbstractSegmentDetector=StepDetector(),
    x_init::Vector{Float64}=zeros(9),
    gt_available::Vector{Bool}=zeros(Bool, length(gt_traj)),
    ref_frame::ReferenceFrame=HEADING,
    feature_type::FeatureType=THREED_STEP,
    init_model::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}=nothing,
    posyaw_measurement_update::Bool=true,
    diagnostics::Optional{CorrectorDiagnostics}=nothing
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

    # With mocap at the start, the corrector starts on it. Otherwise the offset
    # between `x_init` (a whole-walk alignment) and the mocap pose at k=1 --
    # 0.045 rad of yaw on ANG2 13, 26σ of the initial yaw prior -- is first
    # observed after one stride, and the flat prior on β absorbs it as bias.
    Σ_gt = Diagonal(sigma_groundtruth_array(simdata) .^ 2)
    pos_init, quat_init = x_init[1:3], quat[:, 1]
    Σpq_init = P[[1:3; 7:9], [1:3; 7:9], 1]
    if gt_available[1] && posyaw_measurement_update
        pos_init = gt_traj.pos[:, 1]
        δψ = wrap_pi(matrix_to_euler(gt_traj.R_nb[:, :, 1])[3] - x_init[9])
        quat_init = quat_multiply(quat_exp([0.0, 0.0, δψ]), quat[:, 1])
        Σpq_init[[1, 2, 3, 6], [1, 2, 3, 6]] = Σ_gt
    end

    initialize_corrector!(corrector;
        t=inertial.t[1],
        pos_init=pos_init,
        quat_init=quat_init,
        Σpq_init=Σpq_init,
        init_model=init_model
    )

    seg_start = 2
    seg_end = N
    step_seg = Int[1]
    name = split(string(typeof(corrector)), ".")[end]

    @info " ##### Processing $name #####"
    has_params = hasfield(typeof(corrector), :params)

    io_data = Dict{String,CorrectionIO}(
        "input" => CorrectionIO(FEATURE_DIMS[feature_type], true),
        "target" => CorrectionIO(4, true),
        "prediction" => CorrectionIO(4, true),
        "input_norm" => CorrectionIO(FEATURE_DIMS[feature_type], true),
        "target_norm" => CorrectionIO(4, true),
        "prediction_norm" => CorrectionIO(4, true),
        "residual" => CorrectionIO(4, false)
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

        for n in (seg_end-1):-1:seg_start
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
            break
        end

        prev_step = step_seg[end-1]
        curr_step = step_seg[end]

        # INS position error is nav-frame, INS attitude error is body-frame
        # (`state_matrix`), so this puts ε_p in b_i and leaves ε_q in b_{i+1}.
        R_ins_prev = quat_to_matrix(quat[:, prev_step])
        mat66 .= 0.0
        mat66[1:3, 1:3] = R_ins_prev'
        mat66[4:6, 4:6] = Matrix{Float64}(I, 3, 3)

        @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------ " maxlog = 5

        Δp_stride = mat66[1:3, 1:3] * (x[1:3, curr_step] - x[1:3, prev_step])
        Δq_stride = quat_multiply(quat_conjugate(quat[:, prev_step]), quat[:, curr_step])
        Σ_stride = mat66 * ΔP[[1:3; 7:9], [1:3; 7:9]] * mat66'

        stride_err, Σ_err, ins_stride, Σ_ins_stride, R_aug_wl_ins = stride_error(ref_frame;
            R_wb=(R_ins_prev, quat_to_matrix(quat[:, curr_step])),
            Δp=x[1:3, curr_step] - x[1:3, prev_step],
            Σ_ΔpΔθ3=ΔP[[1:3; 9], [1:3; 9]],
            R_wb_gt=(gt_traj.R_nb[:, :, prev_step], gt_traj.R_nb[:, :, curr_step]),
            Δp_gt=gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step],
            Σ_ΔpΔθ3_gt=Σ_gt
        )

        feature, Σ_feature = compute_feature(feature_type;
            ins_stride=ins_stride, Σ_ins_stride=Σ_ins_stride,
            ΔT=inertial.t[curr_step] - inertial.t[prev_step]
        )

        append_io!(io_data["target"], inertial.t[prev_step], stride_err, sqrt.(diag(Σ_err)))
        append_io!(io_data["input"], inertial.t[prev_step], feature, sqrt.(diag(Σ_feature)))

        if has_params
            feat_norm, Σ_feat_norm = normalize_feature!(feature_type;
                feature=copy(feature), Σ_feature=copy(Σ_feature),
                input_stats=corrector.params.input_stats, mid_norm=corrector.params.mid_norm)
            σ_out = corrector.params.output_stats[2]
            target_norm = (stride_err .- corrector.params.output_stats[1]) ./ σ_out
            Σ_err_norm = Diagonal(1 ./ σ_out) * Σ_err * Diagonal(1 ./ σ_out)
            append_io!(io_data["target_norm"], inertial.t[prev_step], target_norm, sqrt.(diag(Σ_err_norm)))
            append_io!(io_data["input_norm"], inertial.t[prev_step], feat_norm, sqrt.(diag(Σ_feat_norm)))
        end

        # R^{bh}: the stride's local frame into the INS body frame at t_i, from
        # INS quantities only, so it is a known input and has no Jacobian.
        R_bh = R_ins_prev' * R_aug_wl_ins[1:3, 1:3]

        predicted = propagate_stride!(corrector;
            t=inertial.t[curr_step], Δp=Δp_stride, Δq=Δq_stride, Σpq=Σ_stride,
            R_bh=R_bh, ins_stride=ins_stride, ref_frame=ref_frame,
            feature_type=feature_type, feature=feature)

        if gt_available[curr_step] && posyaw_measurement_update
            if !isnothing(predicted) && gt_available[prev_step]
                append_io!(io_data["residual"], inertial.t[prev_step], stride_err - predicted[1])
            end
            posyaw_measurement_update!(corrector;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Σ_gt
            )
        elseif !isnothing(predicted)
            append_io!(io_data["prediction"], inertial.t[prev_step],
                predicted[1], sqrt.(diag(predicted[2])))
        end
        relinearize!(corrector)

        isnothing(diagnostics) || record_corrector!(diagnostics, corrector;
            k=curr_step, t=inertial.t[curr_step])
    end

    return zupt, step_seg, get_trajectory(corrector), io_data, get_model(corrector)
end
