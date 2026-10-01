"""
    hybrid_zupt_aided_insv3(inertial, simdata, gt_traj, corrector; ...)

`hybrid_zupt_aided_insv2` with the GP correction moved from the absolute state
onto the stride. Same signature, same return tuple, same `io_data` keys, same
`CorrectorDiagnostics` hook, so every existing consumer works on either.

V2 applies the GP's prediction as a Kalman measurement on the corrector's
*absolute* error state, which shrinks the absolute `Σ` at every test-phase
footfall although the GP only knows about the stride (notes/003 §3.8). Here the
GP corrects the stride and the stride covariance, and the corrected stride then
propagates the absolute state, so no measurement update touches `Σ` in the test
half (notes/013).

The stride error and the feature are built from the inner ZUPT-INS's own
attitude and position, so the GP's input and target depend on the INS and the
ground truth only. Building them from the corrector's attitude instead -- as the
first version did -- lets the corrector's drifting roll/pitch leak into the
training targets, which flipped the sign of the learned yaw bias on some walks
(notes/014). The corrector's orientation enters exactly once, in
`correct_stride`, to put the corrected local stride back into the world.

The ground-truth branches keep V2's updates: mocap *is* exogenous absolute
information, and 002 Result 3 measured that its shrink is earned.
"""
function hybrid_zupt_aided_insv3(
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
    # Opt-in per-footfall record of the corrector's own state and Σ. See
    # `CorrectorDiagnostics` in SingleFilterDiagnostics.jl.
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

    initialize_corrector!(corrector;
        t=inertial.t[1],
        pos_init=x_init[1:3],
        quat_init=quat[:, 1],
        Σpq_init=P[[1:3; 7:9], [1:3; 7:9], 1],
        init_model=init_model
    )

    # n_train_cutoff = floor(Int, train_ratio * N)
    # gt_available = [n <= n_train_cutoff for n in 1:N]
    # @show typeof(gt_available)

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

        # ------------------- Raw stride, before the state moves ----------------
        # V2 called `dynamic_update!` here and read the stride back off the
        # corrector afterwards. Here the propagation waits until the GP has
        # corrected the stride, so the correction lands on the stride rather than
        # on the absolute state.
        Δp_stride = mat66[1:3, 1:3] * (x[1:3, curr_step] - x[1:3, prev_step])
        Δq_stride = quat_multiply(quat_conjugate(quat[:, prev_step]), quat[:, curr_step])
        Σ_stride = mat66 * ΔP[[1:3; 7:9], [1:3; 7:9]] * mat66'

        q_prev = corrector.quat[:, corrector.i]
        R_prev = quat_to_matrix(q_prev)

        # Stride error and stride in the inner INS's own local frame: the target
        # and the feature never see the corrector's state.
        stride_err, Σ_err, ins_stride, Σ_ins_stride, _ = stride_error(ref_frame;
            R_wb=(quat_to_matrix(quat[:, prev_step]), quat_to_matrix(quat[:, curr_step])),
            Δp=x[1:3, curr_step] - x[1:3, prev_step],
            Σ_ΔpΔθ3=ΔP[[1:3; 9], [1:3; 9]],
            R_wb_gt=(gt_traj.R_nb[:, :, prev_step], gt_traj.R_nb[:, :, curr_step]),
            Δp_gt=gt_traj.pos[:, curr_step] - gt_traj.pos[:, prev_step],
            Σ_ΔpΔθ3_gt=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
        )

        # The corrector's local→world map, used only to put the corrected stride
        # back into the world.
        R_aug_wl = stride_local(ref_frame; R_wb=R_prev, ΔpΔθ3=zeros(4))[3]

        # Compute feature
        feature, Σ_feature = compute_feature(feature_type;
            ins_stride=ins_stride, Σ_ins_stride=Σ_ins_stride,
            ΔT=inertial.t[curr_step] - inertial.t[prev_step]
        )

        # Save for plotting
        append_io!(io_data["target"], inertial.t[prev_step], stride_err, sqrt.(diag(Σ_err)))
        append_io!(io_data["input"], inertial.t[prev_step], feature, sqrt.(diag(Σ_feature)))

        if has_params
            feat_norm = deepcopy(feature)
            Σ_feat_norm = deepcopy(Σ_feature)
            normalize_feature!(feature_type;
                feature=feat_norm,
                Σ_feature=Σ_feat_norm,
                input_stats=corrector.params.input_stats,
                mid_norm=corrector.params.mid_norm
            )

            target_norm = deepcopy(stride_err)
            target_norm = (target_norm .- corrector.params.output_stats[1]) ./ corrector.params.output_stats[2]
            Σ_err_norm = deepcopy(Σ_err)
            Σ_err_norm = Diagonal(1 ./ corrector.params.output_stats[2]) * Σ_err_norm * Diagonal(1 ./ corrector.params.output_stats[2])

            append_io!(io_data["target_norm"], inertial.t[prev_step], target_norm, sqrt.(diag(Σ_err_norm)))
            append_io!(io_data["input_norm"], inertial.t[prev_step], feat_norm, sqrt.(diag(Σ_feat_norm)))
        end

        if gt_available[curr_step] && gt_available[prev_step]
            # Mocap at both ends: the stride is a training target, and the
            # absolute update that follows is genuinely exogenous information.
            # Unchanged from V2, deliberately -- see notes/013 §0.
            dynamic_update!(corrector;
                t=inertial.t[curr_step], Δp=Δp_stride, Δq=Δq_stride, Σpq=Σ_stride)

            residual, residual_var = stride_measurement_update!(corrector;
                feature_type=feature_type,
                stride_err=stride_err, Σ_err=Σ_err,
                feature=feature, Σ_feature=Σ_feature, R_aug_wl=R_aug_wl,
            )

            if posyaw_measurement_update
                relinearize!(corrector)

                posyaw_measurement_update!(corrector;
                    curr_pos=gt_traj.pos[:, curr_step],
                    curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                    Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
                )
            end
            if !isnothing(residual)
                append_io!(io_data["residual"], inertial.t[prev_step], residual)
            end

        elseif gt_available[curr_step] && posyaw_measurement_update
            dynamic_update!(corrector;
                t=inertial.t[curr_step], Δp=Δp_stride, Δq=Δq_stride, Σpq=Σ_stride)

            posyaw_measurement_update!(corrector;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2)
            )
        else
            # No ground truth: the GP corrects the stride and the stride
            # covariance, and the corrected stride then propagates the absolute
            # state. No measurement update touches Σ here, so the GP can no
            # longer credit the filter with absolute information it never
            # received. notes/013 §1.4-1.6.
            predicted = predict_stride_error(corrector;
                feature_type=feature_type, feature=feature, Σ_feature=Σ_feature,
                include_noise=true)

            if isnothing(predicted)
                dynamic_update!(corrector;
                    t=inertial.t[curr_step], Δp=Δp_stride, Δq=Δq_stride, Σpq=Σ_stride)
            else
                pred, Σ_pred = predicted
                Δp_corr, Δq_corr, Σ_corr = correct_stride(;
                    q_prev=q_prev, Δp=Δp_stride, Δq=Δq_stride, Σpq=Σ_stride,
                    s_l=ins_stride, pred=pred, Σ_pred=Σ_pred, R_aug_wl=R_aug_wl,
                    mask=corrector.correction_mask)

                dynamic_update!(corrector;
                    t=inertial.t[curr_step], Δp=Δp_corr, Δq=Δq_corr, Σpq=Σ_corr)

                append_io!(io_data["prediction"], inertial.t[prev_step], pred, sqrt.(diag(Σ_pred)))
            end
        end
        relinearize!(corrector)

        # Recorded after relinearize!, so this is the posterior: the state and Σ
        # left by whichever path this footfall took -- mocap in the train half,
        # the corrected propagation in the test half.
        isnothing(diagnostics) || record_corrector!(diagnostics, corrector;
            k=curr_step, t=inertial.t[curr_step])
    end

    return zupt, step_seg, get_trajectory(corrector), io_data, get_model(corrector)
end
