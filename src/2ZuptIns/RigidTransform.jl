"""
    wrapped_min_residuals(ins_vals::AbstractVector{T}, gt_vals::AbstractVector{T}) where T<:Real

Compute the minimum absolute angular residual between two vectors, accounting for the
±2π periodic ambiguity of angles. For each pair of elements, the residual is
`min(|d|, |d+2π|, |d-2π|)` where `d = ins_vals - gt_vals`.

# Arguments
- `ins_vals`: Estimated angles (radians).
- `gt_vals`: Ground‑truth angles (radians), same length as `ins_vals`.

# Returns
- Vector of wrapped residuals of the same length.
"""
function wrapped_min_residuals(ins_vals::AbstractVector{T}, gt_vals::AbstractVector{T}) where T<:Real
    diff = ins_vals - gt_vals
    res = min.(abs.(diff), abs.(diff .+ 2π), abs.(diff .- 2π))
    return res
end

"""
    transform_position(ins_traj::Trajectory, gt_traj::Trajectory, calib_idxs::Vector{Int})

Estimate a rigid transformation (rotation + translation) that aligns the inertial‑navigation
trajectory with the ground‑truth reference frame. The rotation is constrained to a yaw change
plus a fixed 180° roll (to mirror the IMU mounting), and a 3D translation.

# Arguments
- `ins_traj`: Trajectory computed from inertial data.
- `gt_traj`: Ground‑truth trajectory (already temporally aligned with `ins_traj`).
- `calib_idxs`: Indices of the time window used for calibration (typically the first few metres).

# Returns
- `aligned_traj`: `Trajectory` with positions, velocities and orientations rotated/translated
  to match the ground‑truth frame.
- `R_opt`: 3×3 rotation matrix that maps INS‑frame coordinates to GT‑frame coordinates.
- `t_opt`: 3‑element translation vector.
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
    euler_mse(angles::Vector{Float64}, ins_traj::Trajectory, gt_traj::Trajectory,
              zupt::BitVector, calib_idxs::Vector{Int}) -> Vector{Float64}

Compute the per‑sample Euler‑angle residuals between an inertial trajectory (rotated by
`angles`) and the ground truth. Roll and pitch residuals are evaluated only on zero‑velocity
(ZUPT) frames; yaw residuals only on the calibration window. The residuals are the minimum
absolute difference wrapped to the interval [‑π, π].

# Arguments
- `angles`: (roll, pitch, yaw) in radians, applied as an extrinsic ZYX rotation to the
  orientation matrices of `ins_traj`.
- `ins_traj`: Inertial trajectory (original orientation estimates).
- `gt_traj`: Ground‑truth trajectory (reference Euler angles).
- `zupt`: Boolean vector, `true` where the foot is stationary (used for roll/pitch).
- `calib_idxs`: Indices of the calibration window (used for yaw).

# Returns
- A 1‑D vector concatenating the per‑frame roll residual (zupt length), pitch residual
  (zupt length) and yaw residual (length of `calib_idxs`).
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
    transform_orientation(ins_traj::Trajectory, gt_traj::Trajectory, zupt::BitVector,
                          initial_value::Vector{Float64}, calib_idxs::Vector{Int}) ->
                          (Trajectory, Matrix{Float64})

Optimise a full 3‑axis rotation (roll, pitch, yaw) that aligns the inertial orientation
estimates with the ground truth. The cost function minimises the wrapped Euler‑angle errors
over ZUPT frames (roll/pitch) and a calibration window (yaw).

# Arguments
- `ins_traj`: Inertial trajectory (original orientation matrices `R_nb`).
- `gt_traj`: Ground‑truth trajectory (reference orientations).
- `zupt`: Boolean vector, `true` at zero‑velocity frames (used for roll/pitch).
- `initial_value`: Initial guess for (roll, pitch, yaw) in radians.
- `calib_idxs`: Indices of the calibration window (used for yaw).

# Returns
- `aligned_traj`: `Trajectory` with orientations rotated by the optimal transformation
  (positions and velocities unchanged).
- `R_opt`: The optimal 3×3 rotation matrix that maps the INS body frame to the GT body frame.
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
    compute_aligned_ins_trajectory(data_path, trial_id::Int;
                                   sim_config::INSConfig=INSConfig(),
                                   orientation_offset::AbstractVector=zeros(3)) ->
                                   (Trajectory, Trajectory, BitVector, Vector{Int}, InertialData, INSConfig)

Load inertial and ground truth data, compute an INS trajectory, and align it to the ground truth.

# Arguments
- `data_path`: Path to the data directory (string or `AbstractString`).
- `trial_id`: Trial/session identifier passed to the CSV loaders.
- `sim_config`: INS configuration. Defaults to `INSConfig()`.
- `orientation_offset`: 3‑element orientation offset for `transform_orientation`. Defaults to zeros.

# Returns
- `ins_traj_aligned`: The INS trajectory aligned to ground truth.
- `gt_traj_aligned`: The ground truth trajectory aligned to the IMU time axis.
- `zupt`: ZUPT detection signal (boolean vector, `true` where stationary).
- `segs`: Segmentation output from `smoothed_zupt_aided_ins` (vector of step‑end indices).
- `inertial`: Updated inertial data after applying the estimated rotation.
- `sim_config`: Updated simulation config (gravity rotated by the position alignment).
"""
function compute_aligned_ins_trajectory(
    data_path::AbstractString,
    trial_id::Int;
    sim_config::InsConfig=InsConfig(),
    orientation_offset::AbstractVector=zeros(3),
    inertial::InertialData=InertialData(data_path, trial_id),
    gt_traj::Trajectory=Trajectory(data_path, trial_id)
)
    # Truncate to overlapping time window and align ground truth to IMU timestamps
    inertial_trunc, gt_traj_trunc = truncate_to_overlap(inertial, gt_traj)
    gt_traj_aligned = temporal_alignment(gt_traj_trunc, inertial_trunc.t)

    # Compute INS trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial_trunc, sim_config)

    # Calibration window: first point that exceeds the calibration distance from start.
    # Distances are computed from the first position (column 1) to all points,
    # then restricted to the step‑end indices `segs`.
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