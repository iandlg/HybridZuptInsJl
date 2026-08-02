function hybrid_zupt_aided_insv2(
    inertial::InertialData,
    simdata::InsConfig,
    gt_traj::Trajectory,
    corrector::AbstractEstimator;
    step_detector::AbstractSegmentDetector=StepDetector(),
    x_init::Vector{Float64}=zeros(9),
    gt_available::Vector{Bool}=zeros(Bool, length(gt_traj)),
    ref_frame::ReferenceFrame=HEADING,
    feature_type::FeatureType=THREED_STEP,
    β_Σβ_0::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}=nothing
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
        β_Σβ_0=β_Σβ_0
    )

    # n_train_cutoff = floor(Int, train_ratio * N)
    # gt_available = [n <= n_train_cutoff for n in 1:N]
    # @show typeof(gt_available)

    seg_start = 2
    seg_end = N
    step_seg = Int[1]
    @info " ##### Processing $(typeof(corrector)) #####"
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
            # @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------"
            # @info "Prev and curr GT available"
            residual, residual_var = stride_measurement_update!(corrector;
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
            if !isnothing(residual)
                append_io!(io_data["residual"], inertial.t[prev_step], residual)
            end

        elseif gt_available[curr_step]
            # @info "----- Footfall n°$(length(step_seg)) detected : k=$curr_step ------"
            # @info "Curr GT available"
            posyaw_measurement_update!(corrector;
                curr_pos=gt_traj.pos[:, curr_step],
                curr_θ3=matrix_to_euler(gt_traj.R_nb[:, :, curr_step])[3],
                Σy=Diagonal(sigma_groundtruth_array(simdata) .^ 2) .* 0.5e1
            )
        else
            # @info "No GT available"
            pred, var_pred, pred_norm, var_pred_norm = learned_measurement_update!(corrector;
                feature_type=feature_type,
                feature=feature, Σ_feature=Σ_feature, R_aug_wl=R_aug_wl)
            if !isnothing(pred) && !isnothing(var_pred)
                append_io!(io_data["prediction"], inertial.t[prev_step], pred, sqrt.(var_pred))
            end
            if !isnothing(pred_norm) && !isnothing(var_pred_norm)
                append_io!(io_data["prediction_norm"], inertial.t[prev_step], pred_norm, sqrt.(var_pred_norm))
            end

        end
        relinearize!(corrector)
    end

    return zupt, step_seg, get_trajectory(corrector), io_data, get_β_Σβ(corrector)
end

