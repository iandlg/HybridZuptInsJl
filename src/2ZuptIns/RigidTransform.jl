"""
    wrapped_min_residuals(ins_vals, gt_vals) -> Vector

Element-wise `min(|d|, |d+2π|, |d-2π|)` with `d = ins_vals - gt_vals`.
"""
function wrapped_min_residuals(ins_vals::AbstractVector{T}, gt_vals::AbstractVector{T}) where T<:Real
    diff = ins_vals - gt_vals
    res = min.(abs.(diff), abs.(diff .+ 2π), abs.(diff .- 2π))
    return res
end

"""
    transform_position(ins_traj, gt_traj, calib_idxs) -> (aligned_traj, R_opt, t_opt)

Fit a rigid transform (yaw rotation plus a fixed 180° roll for the IMU mounting, and a
translation) mapping `ins_traj` onto the time-aligned `gt_traj` over `calib_idxs`, and
apply it to positions, velocities and orientations.
"""
function transform_position(ins_traj::Trajectory, gt_traj::Trajectory,
    calib_idxs::Vector{Int})

    function residuals!(r::Vector{Float64}, x::Vector{Float64})
        R = euler_to_matrix([π, 0.0, x[1]])
        t = x[2:4]
        diff = gt_traj.pos[:, calib_idxs] .- (R * ins_traj.pos[:, calib_idxs] .+ t)
        r .= vec(diff)   # in-place: LeastSquaresOptim wants mutating f!
    end

    n_res = 3 * length(calib_idxs)
    x0 = zeros(4)

    res = LeastSquaresOptim.optimize!(
        LeastSquaresOptim.LeastSquaresProblem(
            x=copy(x0),
            (f!)=residuals!,
            output_length=n_res,
        ),
        LeastSquaresOptim.LevenbergMarquardt()
    )

    x = res.minimizer
    R_opt = euler_to_matrix([π, 0.0, x[1]])
    t_opt = x[2:4]

    new_pos = R_opt * ins_traj.pos .+ t_opt        # broadcasts correctly
    new_vel = isnothing(ins_traj.vel) ? nothing : R_opt * ins_traj.vel

    N = size(ins_traj.R_nb, 3)
    new_R_nb = similar(ins_traj.R_nb)
    for i in 1:N
        new_R_nb[:, :, i] = R_opt * ins_traj.R_nb[:, :, i]
    end

    return Trajectory(ins_traj.t, new_pos, new_R_nb, new_vel), R_opt, t_opt
end

"""
    euler_mse(angles, ins_traj, gt_traj, zupt, calib_idxs) -> Vector{Float64}

Wrapped Euler residuals of `ins_traj` rotated by `angles` (extrinsic ZYX) against
`gt_traj`: roll and pitch over the ZUPT samples, yaw over `calib_idxs`, concatenated.
"""
function euler_mse(angles::Vector{Float64}, ins_traj::Trajectory,
    gt_traj::Trajectory, zupt::BitVector,
    calib_idxs::Vector{Int})

    R_trial = euler_to_matrix(angles)
    N = size(ins_traj.R_nb, 3)
    R_rotated = Array{Float64}(undef, 3, 3, N)
    for i in 1:N
        R_rotated[:, :, i] = ins_traj.R_nb[:, :, i] * R_trial
    end

    ins_euler = matrix_to_euler(R_rotated)    # (3, N)
    gt_euler = matrix_to_euler(gt_traj.R_nb) # (3, N)

    roll_res = wrapped_min_residuals(ins_euler[1, zupt], gt_euler[1, zupt])
    pitch_res = wrapped_min_residuals(ins_euler[2, zupt], gt_euler[2, zupt])
    yaw_res = wrapped_min_residuals(ins_euler[3, calib_idxs], gt_euler[3, calib_idxs])

    return vcat(roll_res, pitch_res, yaw_res)
end

"""
    transform_orientation(ins_traj, gt_traj, zupt, initial_value, calib_idxs) -> (aligned_traj, R_opt)

Fit the body-frame rotation minimising [`euler_mse`](@ref) from `initial_value` and apply
it to the orientations (positions and velocities unchanged).
"""
function transform_orientation(ins_traj::Trajectory, gt_traj::Trajectory,
    zupt::BitVector, initial_value::Vector{Float64},
    calib_idxs::Vector{Int})

    # n_res: zupt contributes roll+pitch (2×), calib_idxs contributes yaw (1×)
    n_zupt = sum(zupt)
    n_res = 2 * n_zupt + length(calib_idxs)

    # In-place residual wrapper required by LeastSquaresOptim
    function residuals!(r::Vector{Float64}, angles::Vector{Float64})
        r .= euler_mse(angles, ins_traj, gt_traj, zupt, calib_idxs)
    end

    res = LeastSquaresOptim.optimize!(
        LeastSquaresOptim.LeastSquaresProblem(
            x=copy(initial_value),  # use caller's initial guess, not zeros
            (f!)=residuals!,
            output_length=n_res,
        ),
        LeastSquaresOptim.LevenbergMarquardt()
    )

    opt_angles = res.minimizer          # Vector{Float64} of length 3 (roll, pitch, yaw)
    R_opt = euler_to_matrix(opt_angles)

    # Apply rotation to all orientation matrices
    N = size(ins_traj.R_nb, 3)
    new_R_nb = similar(ins_traj.R_nb)
    for i in 1:N
        new_R_nb[:, :, i] = ins_traj.R_nb[:, :, i] * R_opt
    end

    aligned_traj = Trajectory(ins_traj.t, ins_traj.pos, new_R_nb, ins_traj.vel)
    return aligned_traj, R_opt
end

"""
    compute_aligned_ins_trajectory(data_path, trial_id; sim_config=InsConfig(),
        orientation_offset=zeros(3)) -> (ins_traj, gt_traj, zupt, segs, inertial, sim_config)

Load a trial, run the smoothed ZUPT-INS and align it to ground truth. Returns the
aligned INS and ground-truth trajectories (on the IMU time axis), the ZUPT mask, the
step-end indices, the rotated inertial data and the config with gravity rotated to match.
"""
function compute_aligned_ins_trajectory(
    data_path::AbstractString,
    trial_id::Int;
    sim_config::InsConfig=InsConfig(),
    orientation_offset::AbstractVector=zeros(3),
    inertial::InertialData=InertialData(data_path, trial_id),
    gt_traj::Trajectory=Trajectory(data_path, trial_id),
    fs_resample=200.0
)
    src = resolve_source(data_path)

    # Preprocessing sequence 
    inertial_trunc, gt_traj_aligned = preprocess(src, inertial, gt_traj; fs_resample=fs_resample)

    # # Truncate to overlapping time window and align ground truth to IMU timestamps
    # inertial_trunc, gt_traj_trunc = truncate_to_overlap(inertial, gt_traj)

    # Compute INS trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial_trunc, sim_config)

    # Calibration window: first point that exceeds the calibration distance from start.
    # Distances are computed from the first position (column 1) to all points,
    # then restricted to the step-end indices `segs`.
    pos_start = ins_traj.pos[:, 1:1]      # 3×1 matrix
    distances = sqrt.(sum((pos_start .- ins_traj.pos) .^ 2, dims=1))[:]   # length N vector
    dist_at_segs = distances[segs]        # distances only at step ends
    idx_in_segs = findfirst(>(sim_config.calibration_distance_m), dist_at_segs)
    if isnothing(idx_in_segs)
        error("No step end exceeds the calibration distance of $(sim_config.calibration_distance_m) m")
    end
    b = segs[idx_in_segs]   # index in the original array

    # Calibration indices: all indices up to `b` that are **not** ZUPT frames
    calib_idxs = [i for i in 1:b if !zupt[i]]

    # Rigidly align position and orientation to ground truth
    ins_traj_aligned, R_nprime_n, _ = transform_position(ins_traj, gt_traj_aligned, calib_idxs)
    ins_traj_aligned, R_b_bprime = transform_orientation(ins_traj_aligned, gt_traj_aligned,
        zupt, orientation_offset, calib_idxs)

    # Update inertial data: rotate accelerometer and gyroscope readings
    u_rotated = vcat(R_b_bprime' * inertial_trunc.u[1:3, :],
        R_b_bprime' * inertial_trunc.u[4:6, :])
    inertial_updated = InertialData(inertial_trunc.t, u_rotated)

    # Rotate gravity vector according to the position alignment
    sim_config_updated = deepcopy(sim_config)
    sim_config_updated.g = R_nprime_n * [0.0, 0.0, sim_config.g]

    return ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated
end

"The INS state `[pos; vel; euler]` at sample `n` of `traj`, as the filters take it for `x_init`."
initial_state(traj::Trajectory, n::Int=1)::Vector{Float64} =
    vcat(traj.pos[:, n], traj.vel[:, n], matrix_to_euler(traj.R_nb[:, :, n]))

"""
    collect_aligned_trajectories(data_dict; kwargs...) -> Dict{String,Dict{Int,NamedTuple}}

[`compute_aligned_ins_trajectory`](@ref) for every `data_path => trial_ids` pair, as
`data_path => trial_id => (; ins_traj_aligned, gt_traj_aligned, zupt, segs,
inertial_updated, sim_config_updated, x_init)`. Failing trials are skipped with a `@warn`.
"""
function collect_aligned_trajectories(
    data_dict::AbstractDict{String,Tuple{String,Vector{Int}}};
    kwargs...
)::OrderedDict{String,OrderedDict{Int,NamedTuple}}

    results = OrderedDict{String,Dict{Int,NamedTuple}}()

    for (dataset_name, (data_path, trial_ids)) in data_dict
        trial_results = OrderedDict{Int,NamedTuple}()
        failures = Int[]

        for trial_id in trial_ids
            try
                ins_traj_aligned, gt_traj_aligned, zupt, segs,
                inertial_updated, sim_config_updated =
                    compute_aligned_ins_trajectory(data_path, trial_id; kwargs...)

                x_init = initial_state(ins_traj_aligned)

                trial_results[trial_id] = (;
                    ins_traj_aligned=ins_traj_aligned,
                    gt_traj_aligned=gt_traj_aligned,
                    zupt=zupt,
                    segs=segs,
                    inertial_updated=inertial_updated,
                    sim_config_updated=sim_config_updated,
                    x_init=x_init,
                )
            catch e
                @warn "Skipping trial $trial_id in $data_path" exception = e
                push!(failures, trial_id)
            end
        end

        if isempty(trial_results)
            @warn "No trials loaded successfully from $data_path"
        else
            n_loaded = length(trial_results)
            n_total = length(trial_ids)
            if !isempty(failures)
                @warn "$(length(failures)) / $n_total trials failed and were skipped in $data_path. Failed IDs: $failures"
            end
            @info "Loaded $n_loaded / $n_total trials from $data_path"
        end

        results[dataset_name] = trial_results
    end

    return results
end