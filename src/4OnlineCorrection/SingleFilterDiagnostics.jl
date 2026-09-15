"""
Consistency diagnostics for the ZUPT-aided INS + HSGP correction filter.

Provides:
  * `StepDiagnostics`  – per-step record filled inside the V1 filter loop
  * `CorrectorDiagnostics` – per-footfall record of a V2 corrector's own state and Σ,
    with `corrector_nees_series` / `pos_cov_trace` reading it
  * `nees_series`      – post-hoc NEES per state block, with chi-square bounds
  * `autocorr`         – innovation whiteness test with significance bands
  * `noise_state_correlation` – direct test of the Kalman independence assumption
  * `inflation_sweep`  – RMSE / ANEES as a function of measurement-noise inflation

Nothing here is used by the filter itself; all quantities are evaluation-only.
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

Called once per corrected footfall. `y_meas` is what the filter actually
consumed, `y_true_stride` the true stride error mapped through the same
`R_aug_wl`, `e_state_true` the true absolute state error in measurement space.
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
# Per-footfall record, decoupled corrector (V2)
# =====================================================================

"""
Per-footfall record of an `AbstractEstimator`'s own state and covariance, filled
by `hybrid_zupt_aided_insv2` when it is handed one.

The V2 correctors keep `Σ` as a *single* 6×6 matrix over `[pos(1:3); att(4:6)]`
and overwrite it every footfall, so after a run only the final value survives.
Anything that wants the series -- NEES, the covariance trace -- has to be
recorded while the filter runs, which is what this is for.

`k` is the IMU sample index of the footfall, so every series indexes the same
axis as the V1 diagnostics and the `split_k` divider in the plots.
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

Snapshot the corrector at one footfall. Call it *after* `relinearize!`, so the
record is the posterior -- state and `Σ` after whichever update that footfall
took (mocap in the train half, GP in the test half).

`Σ` is copied rather than aliased: the next `dynamic_update!` writes the same
matrix in place.
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
    corrector_nees_series(d, gt_traj; att_convention, include_vel)

NEES of a decoupled corrector against ground truth, per footfall. An adapter
onto [`nees_series`](@ref) rather than a second implementation of it: the 6×6
`[pos; att]` covariance is embedded into the 9-state layout that function
expects (position 1:3, attitude 7:9, velocity block left at zero and never
read), and ground truth is subsampled to the footfall samples with `gt_traj[d.k]`.

The returned `k` is relabelled to the IMU sample indices, so the result drops
straight into `plot_nees_comparison` alongside a `split_k` divider.

`att_convention` defaults to `:left` here, NOT to `nees_series`'s `:right`:
`relinearize!` on the decoupled correctors applies `quat_exp(δθ) * q`, a left
perturbation, so the attitude error matching their `Σ[4:6,4:6]` is
`logmap(R_gt * R_est')`. Position NEES is unaffected either way.
"""
function corrector_nees_series(d::CorrectorDiagnostics, gt_traj;
    att_convention::Symbol=:left, include_vel::Bool=false)

    n = length(d)
    n == 0 && error("No footfalls recorded; pass `diagnostics=` to hybrid_zupt_aided_insv2.")

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
    pos_cov_trace(d)

`tr(Σ[1:3,1:3])` per footfall -- the position uncertainty the corrector reports,
and the matrix `corrector_nees_series` scores its position error against. Read
the two together: a trace that falls while NEES rises is a shrink the estimator
has not earned.
"""
pos_cov_trace(d::CorrectorDiagnostics) =
    (k=copy(d.k), trace=[tr(S[1:3, 1:3]) for S in d.Σ])

# =====================================================================
# 1. NEES  (state consistency)
# =====================================================================

"""
    nees_series(x, P, quat, gt_traj; ks, att_convention)

Per-sample NEES for the position / velocity / attitude blocks separately.
Blocks are kept apart on purpose: pooling them hides which one diverges and
mixes units with wildly different scales.

`att_convention` selects the attitude error definition matching the filter's
perturbation model. `:right` uses logmap(R_ins' * R_gt), `:left` uses
logmap(R_gt * R_ins'). Given the `d(theta3)/d(delta_theta)_right` Jacobian in
the measurement matrix, `:right` is the one to use.

Returns a NamedTuple of vectors plus the two-sided 95% chi-square bounds for a
single run (dof = 3).
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

"""
    anees(nees_runs; dof)

Average NEES over independent Monte-Carlo runs. `nees_runs` is a vector of
equal-length NEES vectors (one per run). Bounds tighten as 1/sqrt(N), which is
what makes this far more convincing than a single-run plot.
"""
function anees(nees_runs::Vector{<:AbstractVector}; dof::Int=3)
    N = length(nees_runs)
    m = mean(hcat(nees_runs...), dims=2)[:]
    lo = quantile(Chisq(dof * N), 0.025) / N
    hi = quantile(Chisq(dof * N), 0.975) / N
    return (anees=m, lower=lo, upper=hi, n_runs=N, dof=dof)
end

"""Fraction of samples falling inside the 95% NEES envelope. 0.95 == consistent."""
consistency_ratio(nees::AbstractVector, lo::Real, hi::Real) =
    count(v -> lo <= v <= hi, nees) / length(nees)

# =====================================================================
# 2. Whiteness of the innovation sequence
# =====================================================================

"""
    autocorr(v, maxlag)

Normalised sample autocorrelation, lags 0..maxlag. For a correctly specified
Kalman update the innovation sequence is white, so lags >= 1 sit inside
+/- 1.96/sqrt(n). Persistent positive correlation is the signature of a
measurement that carries state error.
"""
function autocorr(v::AbstractVector{<:Real}, maxlag::Int)
    n = length(v)
    mu = mean(v)
    d = v .- mu
    c0 = sum(abs2, d) / n
    c0 == 0 && return zeros(maxlag + 1)
    return [sum(d[1:(n-l)] .* d[(1+l):n]) / (n * c0) for l in 0:maxlag]
end

"""
    whiteness_test(diagnostics; maxlag, component)

`component = 0` uses the scalar NIS sequence; `component = i` uses the i-th
normalised innovation. Returns the ACF, the significance band, and a
Ljung-Box p-value (small p == reject whiteness).
"""
function whiteness_test(diagnostics::StepDiagnostics; maxlag::Int=15, component::Int=0)
    v = component == 0 ? diagnostics.nis : [nu[component] for nu in diagnostics.innov_norm]
    n = length(v)
    n <= maxlag + 2 && error("Not enough steps ($n) for maxlag=$maxlag.")

    rho = autocorr(v, maxlag)
    band = 1.96 / sqrt(n)
    Q = n * (n + 2) * sum(rho[l+1]^2 / (n - l) for l in 1:maxlag)
    pval = ccdf(Chisq(maxlag), Q)

    return (lags=0:maxlag, acf=rho, band=band,
        ljung_box=Q, pvalue=pval, n=n)
end

"""
    zupt_consistency(diagnostics)

Velocity consistency without a velocity reference. At a ZUPT epoch the true
velocity is zero by assumption, so the ZUPT innovation is a measurement whose
expected value is known exactly. Mean NIS should sit near 3.

Well above 3 means the velocity block of P is over-confident by the time ZUPTs
arrive -- which is what you would expect if a correlated position/yaw update
has already shrunk P too far and the coupling has propagated. Well below 3
means R_meas is set too loose. Either way this substitutes for the velocity
NEES when OptiTrack gives you no velocity.
"""
function zupt_consistency(diagnostics::StepDiagnostics)
    n = length(diagnostics.zupt_nis)
    n == 0 && error("No ZUPT epochs recorded; add the record hook to the ZUPT branch.")
    m = mean(diagnostics.zupt_nis)
    lo = quantile(Chisq(3 * n), 0.025) / n
    hi = quantile(Chisq(3 * n), 0.975) / n
    return (mean_nis=m, expected=3.0, lower=lo, upper=hi,
        consistent=lo <= m <= hi, n=n)
end

"""
    zupt_gain_series(diagnostics; from_k, to_k)

Per-ZUPT-epoch view of how much position-correction authority the ZUPT actually
has. Position is never directly observed in a ZUPT-aided INS: the only channel
that walks back the error accumulated during the swing phase is the
position<->velocity cross-covariance, through the position rows of the ZUPT gain

    K[1:3, :] = P[1:3, 4:6] * S^-1,    S = P[4:6,4:6] + R_meas

so `K_pos` is the quantity a GP covariance update starves when it shrinks the
absolute `P[1:3,1:3]`. `dpos` is the position correction that gain actually
delivered at each epoch, and `cum_dpos` its running total -- the integrated
shortfall is what shows up as position RMSE.

`from_k`/`to_k` restrict to the epochs in `from_k <= k <= to_k`. That window is
how one run is split into its two phases: the train half, where the mocap update
shrinks `P`, and the test half, where the GP correction does (or does not).
Attitude counterparts are returned alongside as a control: they should be much
less affected.
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

"""
    nis_summary(diagnostics; dof)

Mean NIS with its expected value and 95% interval. Mean NIS >> dof means the
innovation covariance S is understated, i.e. the correction is over-trusted.
"""
function nis_summary(diagnostics::StepDiagnostics; dof::Int=4)
    n = length(diagnostics.nis)
    m = mean(diagnostics.nis)
    lo = quantile(Chisq(dof * n), 0.025) / n
    hi = quantile(Chisq(dof * n), 0.975) / n
    return (mean_nis=m, expected=float(dof), lower=lo, upper=hi,
        consistent=lo <= m <= hi, n=n)
end

# =====================================================================
# 3. Independence assumption, measured directly
# =====================================================================

"""
    noise_state_correlation(diagnostics)

The Kalman update assumes E[v * dx'] = 0, where v is the measurement noise and
dx the state error. Here v is approximated by the prediction error and dx by
the true state error, both in measurement space. A per-component correlation
significantly different from zero is direct evidence the assumption is violated.

`band` is the ~95% significance threshold for zero correlation.
"""
function noise_state_correlation(diagnostics::StepDiagnostics)
    n = length(diagnostics)
    E = hcat(diagnostics.pred_err_state...)     # p x n
    X = hcat(diagnostics.state_err...)          # p x n
    p = size(E, 1)
    rho = [cor(E[i, :], X[i, :]) for i in 1:p]
    return (rho=rho, band=1.96 / sqrt(n), n=n,
        significant=abs.(rho) .> 1.96 / sqrt(n))
end

# =====================================================================
# 4. Error summaries
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

# =====================================================================
# 5. Measurement-noise inflation sweep
# =====================================================================

"""
    inflation_sweep(inertial, simdata, gt_traj, params; factors, kwargs...)

Runs the single-filter configuration with y_cov scaled by each factor and
reports RMSE + consistency. Monotone improvement toward the two-filter result
as the factor grows shows the degradation is an over-confidence effect, and
that the two-filter design is the factor -> infinity limit taken exactly rather
than by tuning.

Pass `kwargs` straight through to `hybrid_zupt_aided_ins` (ref_frame,
feature_type, gt_available, ...).
"""
function inflation_sweep(inertial, simdata, gt_traj, params;
    factors=[1.0, 1e1, 1e2, 1e3, 1e4], kwargs...)

    rows = NamedTuple[]

    # reference: two-filter (correction applied, covariance untouched)
    _, traj2, _, _, _, _, _, _, _, d2, q2, x2, P2 = hybrid_zupt_aided_ins(
        inertial, simdata, gt_traj, params; cov_update=false, kwargs...)
    r2 = rmse_summary(x2, q2, gt_traj)
    nd = nees_series(x2, P2, q2, gt_traj)
    push!(rows, (factor=NaN, mode="two-filter",
        rmse_pos=r2.pos, rmse_yaw=r2.yaw,
        mean_nees_pos=mean(nd.pos),
        inside_95=consistency_ratio(nd.pos, nd.lower, nd.upper),
        mean_nis=length(d2) > 0 ? mean(d2.nis) : NaN))

    for f in factors
        _, _, _, _, _, _, _, _, _, dg, qq, xx, PP = hybrid_zupt_aided_ins(
            inertial, simdata, gt_traj, params;
            cov_update=true, R_inflation=f, kwargs...)
        r = rmse_summary(xx, qq, gt_traj)
        nn = nees_series(xx, PP, qq, gt_traj)
        push!(rows, (factor=f, mode="single-filter",
            rmse_pos=r.pos, rmse_yaw=r.yaw,
            mean_nees_pos=mean(nn.pos),
            inside_95=consistency_ratio(nn.pos, nn.lower, nn.upper),
            mean_nis=length(dg) > 0 ? mean(dg.nis) : NaN))
    end
    return rows
end

"""Pretty-print the sweep table."""
function print_sweep(rows)
    # metric_symbol_ascii, not metric_symbol: this is terminal output and there
    # is no subscript ψ in Unicode, so the figures' form cannot be printed.
    @printf("%-14s %10s %10s %10s %12s %10s\n",
        "mode", "factor", metric_symbol_ascii(:rmse), metric_symbol_ascii(:rmse_yaw),
        "mean NEES", "in 95%")
    for r in rows
        @printf("%-14s %10.1e %10.4f %10.4f %12.2f %9.1f%%\n",
            r.mode, r.factor, r.rmse_pos, r.rmse_yaw,
            r.mean_nees_pos, 100 * r.inside_95)
    end
end