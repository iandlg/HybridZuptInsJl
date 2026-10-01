"""
Evaluation-only consistency diagnostics for the correction filters (nothing here is
used by the filters themselves): `StepDiagnostics` (V1, per step), `CorrectorDiagnostics`
(V4 corrector, per footfall), NEES series and ZUPT gain series.
"""


# =====================================================================
# Numerics helpers
# =====================================================================

"""Rotation matrix -> rotation vector (log map), numerically guarded."""
function rotmat_to_rotvec(R::AbstractMatrix)
    c = clamp((tr(R) - 1) / 2, -1.0, 1.0)
    theta = acos(c)
    if theta < 1e-8
        return [R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2]] / 2
    end
    s = sin(theta)
    return (theta / (2s)) * [R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2]]
end

# =====================================================================
# Per-step record
# =====================================================================

Base.@kwdef struct StepDiagnostics
    k::Vector{Int} = Int[]                 # sample index (curr_step)
    t::Vector{Float64} = Float64[]

    # --- filter-internal consistency -------------------------------------
    innov::Vector{Vector{Float64}} = Vector{Float64}[]     # nu = y_meas - H*dx_prior
    innov_norm::Vector{Vector{Float64}} = Vector{Float64}[]     # nu ./ sqrt.(diag(S))
    nis::Vector{Float64} = Float64[]             # nu' S^-1 nu,  dof = length(nu)

    # --- GP calibration (independent of the filter) ----------------------
    pred_err_stride::Vector{Vector{Float64}} = Vector{Float64}[] # y_hat - y_true_stride
    pred_nis_stride::Vector{Float64} = Float64[]         # e' y_cov^-1 e
    pred_err_state::Vector{Vector{Float64}} = Vector{Float64}[] # y_hat - true absolute state error

    # --- independence assumption ----------------------------------------
    state_err::Vector{Vector{Float64}} = Vector{Float64}[]     # [gt.pos - x; wrapped yaw err]

    # --- bookkeeping ------------------------------------------------------
    trace_S::Vector{Float64} = Float64[]
    trace_ycov::Vector{Float64} = Float64[]
    trace_P_pre::Vector{Float64} = Float64[]
    trace_P_post::Vector{Float64} = Float64[]

    # --- ZUPT epochs (velocity consistency, needs no ground truth) --------
    zupt_k::Vector{Int} = Int[]
    zupt_nis::Vector{Float64} = Float64[]
    zupt_K_att::Vector{Float64} = Float64[]   # ||K[7:9,:]||, ZUPT->attitude authority
    zupt_P_att::Vector{Float64} = Float64[]   # tr(P[7:9,7:9]) at the ZUPT epoch
    zupt_K_pos::Vector{Float64} = Float64[]   # ||K[1:3,:]||, ZUPT->position authority
    zupt_P_pos::Vector{Float64} = Float64[]   # tr(P[1:3,1:3]) at the ZUPT epoch
    zupt_dpos::Vector{Float64} = Float64[]    # ||position correction applied by this ZUPT||
end

Base.length(d::StepDiagnostics) = length(d.k)

"""
    record_step!(diagnostics; ...)

Called once per corrected footfall. `y_meas` is what the filter consumed,
`y_true_stride` the true stride error through the same `R_aug_wl`, `e_state_true` the
true absolute state error in measurement space.
"""
function record_step!(diagnostics::StepDiagnostics;
    k::Int, t::Float64,
    y_meas::AbstractVector, S::AbstractMatrix,
    y_hat::AbstractVector, y_cov::AbstractMatrix,
    y_true_stride::AbstractVector, e_state_true::AbstractVector,
    P_pre::AbstractMatrix, P_post::AbstractMatrix)

    nu = collect(y_meas)                     # prior dx is zero by construction
    sd = sqrt.(abs.(diag(S)))

    push!(diagnostics.k, k)
    push!(diagnostics.t, t)
    push!(diagnostics.innov, nu)
    push!(diagnostics.innov_norm, nu ./ sd)
    push!(diagnostics.nis, mahalanobis(nu, S))

    e_stride = collect(y_hat) .- collect(y_true_stride)
    push!(diagnostics.pred_err_stride, e_stride)
    push!(diagnostics.pred_nis_stride, mahalanobis(e_stride, y_cov))
    push!(diagnostics.pred_err_state, collect(y_hat) .- collect(e_state_true))

    push!(diagnostics.state_err, collect(e_state_true))
    push!(diagnostics.trace_S, tr(S))
    push!(diagnostics.trace_ycov, tr(y_cov))
    push!(diagnostics.trace_P_pre, tr(P_pre))
    push!(diagnostics.trace_P_post, tr(P_post))
    return diagnostics
end

# =====================================================================
# Per-footfall record, V4 corrector
# =====================================================================

"""
Per-footfall record of a corrector's state and `[pos; att]` covariance, filled by
`hybrid_zupt_aided_insv4` when passed `diagnostics=`. `k` is the IMU sample index.
"""
Base.@kwdef struct CorrectorDiagnostics
    k::Vector{Int} = Int[]
    t::Vector{Float64} = Float64[]
    pos::Vector{Vector{Float64}} = Vector{Float64}[]
    quat::Vector{Vector{Float64}} = Vector{Float64}[]
    Σ::Vector{Matrix{Float64}} = Matrix{Float64}[]
end

Base.length(d::CorrectorDiagnostics) = length(d.k)

"""
    record_corrector!(d, c; k, t)

Snapshot (a copy of) the corrector's posterior state and `Σ` at one footfall; call it
after `relinearize!`.
"""
function record_corrector!(d::CorrectorDiagnostics, c::AbstractEstimator; k::Int, t::Float64)
    push!(d.k, k)
    push!(d.t, t)
    push!(d.pos, collect(c.pos[:, c.i]))
    push!(d.quat, collect(c.quat[:, c.i]))
    push!(d.Σ, Matrix(c.Σ[1:6, 1:6]))
    return d
end

"""
    corrector_nees_series(d, gt_traj; att_convention=:left, include_vel=false)

Per-footfall NEES of a corrector against ground truth, via [`nees_series`](@ref) with `k`
relabelled to IMU sample indices. Defaults to `:left` because the correctors perturb
attitude on the left (`quat_exp(δθ) * q`).
"""
function corrector_nees_series(d::CorrectorDiagnostics, gt_traj;
    att_convention::Symbol=:left, include_vel::Bool=false)

    n = length(d)
    n == 0 && error("No footfalls recorded; pass `diagnostics=` to hybrid_zupt_aided_insv4.")

    x = zeros(9, n)
    P = zeros(9, 9, n)
    quat = zeros(4, n)
    for i in 1:n
        x[1:3, i] = d.pos[i]
        quat[:, i] = d.quat[i]
        P[1:3, 1:3, i] = d.Σ[i][1:3, 1:3]
        P[7:9, 7:9, i] = d.Σ[i][4:6, 4:6]
    end

    nees = nees_series(x, P, quat, gt_traj[d.k];
        ks=1:n, att_convention=att_convention, include_vel=include_vel)
    return (; nees..., k=copy(d.k))
end

"""
    nees_yaw_series(d, gt_traj; att_convention=:left) -> (k, yaw, lower, upper, dof=1)

Yaw-only NEES of a corrector per footfall (`Σ[6,6]`, `Chisq(1)` envelope). Roll and pitch
are unobserved, so the 3-dof attitude NEES would hide the yaw channel (notes/015 §2.5).
"""
function nees_yaw_series(d::CorrectorDiagnostics, gt_traj; att_convention::Symbol=:left)
    n = length(d)
    n == 0 && error("No footfalls recorded; pass `diagnostics=` to the filter.")

    gt = gt_traj[d.k]
    nyaw = Float64[]
    for i in 1:n
        R_est = quat_to_matrix(d.quat[i])
        R_gt = gt.R_nb[:, :, i]
        R_err = att_convention === :right ? R_est' * R_gt : R_gt * R_est'
        eψ = rotmat_to_rotvec(R_err)[3]
        push!(nyaw, mahalanobis([eψ], view(d.Σ[i], 6:6, 6:6)))
    end

    lo, hi = quantile(Chisq(1), 0.025), quantile(Chisq(1), 0.975)
    return (k=copy(d.k), yaw=nyaw, lower=lo, upper=hi, dof=1)
end

# =====================================================================
# 1. NEES  (state consistency)
# =====================================================================

"""
    nees_series(x, P, quat, gt_traj; ks, att_convention=:right, include_vel=false)

Per-sample NEES of the position, velocity and attitude blocks separately, with the
two-sided 95% χ² bounds (dof = 3). `att_convention` picks the attitude error matching
the filter's perturbation: `:right` = `logmap(R_ins' * R_gt)`, `:left` =
`logmap(R_gt * R_ins')`.
"""
function nees_series(x::AbstractMatrix, P::AbstractArray{<:Real,3},
    quat::AbstractMatrix, gt_traj;
    ks=1:size(x, 2), att_convention::Symbol=:right,
    include_vel::Bool=false)

    npos = Float64[];
    natt = Float64[]
    nvel = include_vel ? Float64[] : nothing

    for k in ks
        ep = x[1:3, k] .- gt_traj.pos[:, k]
        push!(npos, mahalanobis(ep, view(P, 1:3, 1:3, k)))

        R_ins = quat_to_matrix(quat[:, k])
        R_gt = gt_traj.R_nb[:, :, k]
        R_err = att_convention === :right ? R_ins' * R_gt : R_gt * R_ins'
        push!(natt, mahalanobis(rotmat_to_rotvec(R_err), view(P, 7:9, 7:9, k)))

        if include_vel
            ev = x[4:6, k] .- gt_traj.vel[:, k]
            push!(nvel, mahalanobis(ev, view(P, 4:6, 4:6, k)))
        end
    end

    lo, hi = quantile(Chisq(3), 0.025), quantile(Chisq(3), 0.975)
    return (k=collect(ks), pos=npos, vel=nvel, att=natt,
        lower=lo, upper=hi, dof=3)
end

"""Fraction of samples falling inside the 95% NEES envelope. 0.95 == consistent."""
consistency_ratio(nees::AbstractVector, lo::Real, hi::Real) =
    count(v -> lo <= v <= hi, nees) / length(nees)

# =====================================================================
# 2. Whiteness of the innovation sequence
# =====================================================================

"""
    zupt_gain_series(diagnostics; from_k=1, to_k=typemax(Int))

Per-ZUPT-epoch position gain `K_pos` (`‖K[1:3,:]‖`), position covariance, applied
correction `dpos` and its cumulative sum, with attitude counterparts as a control,
over epochs `from_k <= k <= to_k`.
"""
function zupt_gain_series(diagnostics::StepDiagnostics; from_k::Int=1,
    to_k::Int=typemax(Int))
    isempty(diagnostics.zupt_k) &&
        error("No ZUPT epochs recorded; the ZUPT branch fills these.")
    length(diagnostics.zupt_K_pos) == length(diagnostics.zupt_k) ||
        error("ZUPT gain fields not recorded for this run.")

    sel = findall(k -> from_k <= k <= to_k, diagnostics.zupt_k)
    isempty(sel) && error("No ZUPT epochs in k = $from_k:$to_k.")

    dpos = diagnostics.zupt_dpos[sel]
    return (k=diagnostics.zupt_k[sel],
        P_pos=diagnostics.zupt_P_pos[sel],
        K_pos=diagnostics.zupt_K_pos[sel],
        dpos=dpos, cum_dpos=cumsum(dpos),
        P_att=diagnostics.zupt_P_att[sel],
        K_att=diagnostics.zupt_K_att[sel],
        mean_K_pos=mean(diagnostics.zupt_K_pos[sel]),
        mean_P_pos=mean(diagnostics.zupt_P_pos[sel]),
        total_dpos=sum(dpos), n=length(sel))
end

# =====================================================================
# 3. Error summaries
# =====================================================================

function rmse_summary(x::AbstractMatrix, quat::AbstractMatrix, gt_traj;
    ks=1:size(x, 2), include_vel::Bool=false)
    ep = [norm(x[1:3, k] .- gt_traj.pos[:, k]) for k in ks]
    ey = [abs(wrap_pi(matrix_to_euler(gt_traj.R_nb[:, :, k])[3] -
                      matrix_to_euler(quat_to_matrix(quat[:, k]))[3])) for k in ks]
    ev = include_vel ? [norm(x[4:6, k] .- gt_traj.vel[:, k]) for k in ks] : nothing
    return (pos=sqrt(mean(abs2, ep)),
        vel=include_vel ? sqrt(mean(abs2, ev)) : nothing,
        yaw=sqrt(mean(abs2, ey)), final_pos=ep[end])
end

