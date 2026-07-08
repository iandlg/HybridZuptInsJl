
"""
    synchronize(imu::InertialData, traj::Trajectory; fs_resample=200.0)

Synchronize an IMU and a trajectory using the norm of angular rates.
The trajectory's timestamps are shifted by the estimated lag so that
its angular velocity aligns with the IMU gyroscope. Both outputs are
trimmed to the overlapping time window after alignment.

# Arguments
- `imu`: InertialData object (gyro in rows 4-6).
- `traj`: Trajectory object (must have `R_nb`).
- `fs_resample`: common sampling rate (Hz) for GCC-PHAT.

# Returns
- `imu_sync::InertialData`: IMU trimmed to overlap.
- `traj_sync::Trajectory`: trajectory with shifted time, trimmed to overlap.
- `lag::Float64`: estimated time shift (seconds) added to `traj.t`.
"""
function synchronize(imu::InertialData, traj::Trajectory; fs_resample=200.0)
    # 1. IMU gyro norm
    imu_gyro = imu.u[4:6, :]
    imu_norm = sqrt.(sum(abs2, imu_gyro, dims=1))[:]

    # 2. Trajectory angular velocity (3×(N-2)) and its time points
    ω_traj = angular_velocity_from_rotations(traj.t, traj.R_nb)  # size (3, N-2)
    t_ω = traj.t[2:(end-1)]                    # corresponding time vector (length N-2)
    traj_norm = sqrt.(sum(abs2, ω_traj, dims=1))[:]

    # 3. Common time grid covering the overlap of IMU and trajectory angular velocity
    t_start = max(imu.t[1], t_ω[1])
    t_stop = min(imu.t[end], t_ω[end])
    t_start < t_stop || error("No overlap between IMU and trajectory angular velocity")
    t_grid = collect(t_start:(1/fs_resample):t_stop)

    # 4. Resample norms onto common grid
    imu_resampled = resample_to_grid(imu.t, imu_norm, t_grid)
    traj_resampled = resample_to_grid(t_ω, traj_norm, t_grid)

    # 5. Estimate lag via GCC-PHAT
    lag, cc, lags = gcc_phat(imu_resampled, traj_resampled, fs_resample)

    # 6. Shift the whole trajectory by the estimated lag
    traj_shifted = Trajectory(traj.t .+ lag, traj.pos, traj.R_nb, traj.vel)

    # 7. Trim both to the new overlapping time window
    t_overlap_start = max(imu.t[1], traj_shifted.t[1])
    t_overlap_end = min(imu.t[end], traj_shifted.t[end])
    t_overlap_start < t_overlap_end || error("No overlap after time shift")

    mask_imu = (imu.t .>= t_overlap_start) .& (imu.t .<= t_overlap_end)
    mask_traj = (traj_shifted.t .>= t_overlap_start) .& (traj_shifted.t .<= t_overlap_end)

    imu_sync = imu[mask_imu]
    traj_sync = traj_shifted[mask_traj]

    return imu_sync, traj_sync, lag
end

"""
    fit_constant_rotation(ω_imu, ω_opti)

ω_imu, ω_opti: (3, N) matrices of body-frame angular velocity samples,
time-synced and same length.

Returns R such that ω_opti ≈ R * ω_imu (least-squares, in the SO(3) sense).
"""
function fit_constant_rotation(ω_imu::AbstractMatrix{T}, ω_opti::AbstractMatrix{T}) where T
    @assert size(ω_imu) == size(ω_opti)
    # Filter out low motion samples
    mask_imu = sqrt.(sum(abs2, ω_imu, dims=1))[:] .> 0.5
    mask_opti = sqrt.(sum(abs2, ω_opti, dims=1))[:] .> 0.5
    mask = mask_imu .& mask_opti
    # Cross-covariance matrix
    H = ω_imu[:, mask] * ω_opti[:, mask]'          # (3,3): sum over samples of ω_imu * ω_opti'
    F = svd(H)
    d = sign(det(F.V * F.U'))
    D = Diagonal([1.0, 1.0, d])   # ensures proper rotation (det = +1)
    R = F.V * D * F.U'
    return R
end

function preprocess(::DCSC, imu::InertialData, gt::Trajectory; fs_resample=200.0, kwargs...)
    # Compute lag between both timeseries
    imu_sync, gt_sync, lag = synchronize(imu, gt; fs_resample=fs_resample)

    # Interpolate ground truth trajectory
    gt_aligned = temporal_alignment(gt_sync, imu_sync.t)
    ω_opti = angular_velocity_from_rotations(gt_aligned)
    ω_imu = imu.u[4:6, 2:(end-1)]

    # Find rotation between IMU and Optitrack rigid object
    R = fit_constant_rotation(ω_imu, ω_opti)
    mats = similar(gt_aligned.R_nb)
    for i in axes(mats, 3)
        mats[:, :, i] = gt_aligned.R_nb[:, :, i] * R
    end
    gt_aligned = HybridZuptInsJl.Trajectory(gt_aligned.t, gt_aligned.pos, mats)
    return imu_sync, gt_aligned
end


function preprocess(::Union{ANG,ANG2}, imu::InertialData, gt::Trajectory; kwargs...)
    inertial_trunc, gt_traj_trunc = truncate_to_overlap(imu, gt)
    gt_traj_aligned = temporal_alignment(gt_traj_trunc, inertial_trunc.t)
    return inertial_trunc, gt_traj_aligned
end

