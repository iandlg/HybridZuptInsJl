### Does correcting the stride instead of the absolute state make the filter honest?
###
### V2 applies the GP's stride-error prediction as a Kalman measurement on the
### corrector's absolute error state, so the absolute `Σ` shrinks at every
### test-phase footfall. V3 corrects the stride and the stride covariance and
### propagates with those, so `Σ` is never shrunk by the GP. notes/013.
###
### Run both on the same walk with the same estimator configuration and read
### three things together:
###   (a) RMSE -- is the correction still being applied at full strength?
###   (b) position NEES -- was the uncertainty honest?
###   (c) tr(Σ[1:3,1:3]) -- which direction did the uncertainty move?
###
### (b) and (c) come from the same matrix, so the pair is one statement: a trace
### that falls while NEES rises is over-confidence.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
using OrderedCollections, Statistics, Printf, LinearAlgebra

## 1. HSGP hyperparameters
m = 200
hsgp_p_key = 42
params, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 2. Track -- the same walk 7_decoupled_consistency.jl reports on
data_key = "ANG2"
trial_id = 14                                   # = TEST_IDS["ANG2"], the held-out walk
output_channels = [:pos_1, :pos_2, :yaw]

ins_traj_aligned, gt_traj, zupt, segs, inertial, simdata =
    HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir(data_key), trial_id)

x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
)

train_ratio = 0.3
N = length(inertial)
n_train_cutoff = floor(Int, train_ratio * N)
k0 = n_train_cutoff + 1
n_alloc = 300

## 3. The frame round-trip, before trusting any number below
# notes/013 §1.7. A GP that predicts nothing and claims exactly the INS stride's
# uncertainty must leave `correct_stride` returning its input, otherwise the
# w -> l -> w -> b round trip is wrong and every covariance downstream is too.
let
    q_prev = HybridZuptInsJl.matrix_to_quat(
        HybridZuptInsJl.euler_to_matrix([0.21, -0.37, 1.1]))
    Δq = HybridZuptInsJl.matrix_to_quat(
        HybridZuptInsJl.euler_to_matrix([0.05, 0.02, 0.11]))
    Δp = [0.63, -0.12, 0.04]
    L = randn(6, 6)
    Σpq = L * L' + 1e-3I

    R_prev = HybridZuptInsJl.quat_to_matrix(q_prev)
    q_raw = HybridZuptInsJl.quat_multiply(q_prev, Δq)
    Δθ3 = HybridZuptInsJl.wrap_pi(
        HybridZuptInsJl.matrix_to_euler(HybridZuptInsJl.quat_to_matrix(q_raw))[3] -
        HybridZuptInsJl.matrix_to_euler(R_prev)[3])

    for frame in (HybridZuptInsJl.HEADING, HybridZuptInsJl.BODY)
        G = zeros(6, 6)
        G[1:3, 1:3] = R_prev
        G[4:6, 4:6] = HybridZuptInsJl.quat_to_matrix(q_raw)
        Σ_inc = G * Σpq * G'

        s_l, Σ_l, R_aug_wl = HybridZuptInsJl.stride_local(frame;
            R_wb=R_prev, ΔpΔθ3=[R_prev * Δp; Δθ3],
            Σ_ΔpΔθ3=Matrix(Σ_inc[[1:3; 6], [1:3; 6]]))

        # The covariance assertion takes the full mask on purpose. With a
        # partial one the cross-block zeroing of §1.4/§1.5 is real and mixes
        # through A into every position component, so there is no block left
        # that the round trip has to preserve. The vector assertions hold for
        # either mask.
        for mask in ([1, 2, 3, 4], [1, 2, 4])
            Σ_pred = zeros(4, 4)
            Σ_pred[mask, mask] = Σ_l[mask, mask]

            Δp_c, Δq_c, Σ_c = HybridZuptInsJl.correct_stride(;
                q_prev=q_prev, Δp=Δp, Δq=Δq, Σpq=Σpq,
                pred=zeros(4), Σ_pred=Σ_pred, R_aug_wl=R_aug_wl, mask=mask)

            @assert norm(Δp_c - Δp) < 1e-12 "$frame/$mask: Δp off by $(norm(Δp_c - Δp))"
            @assert norm(Δq_c - Δq) < 1e-12 "$frame/$mask: Δq off by $(norm(Δq_c - Δq))"
            if length(mask) == 4
                e = norm(Σ_c[1:3, 1:3] - Σpq[1:3, 1:3]) / norm(Σpq[1:3, 1:3])
                @assert e < 1e-10 "$frame: Σ position block off by $e"
            end
            @assert isposdef(Symmetric(Σ_c)) "$frame/$mask: Σpq_corr not positive definite"
        end

        # The check above passes for any orthogonal R_aug_wl, right frame or
        # not. This one does not: it pins the frame to the one the GP was
        # trained in, via the stride `compute_feature` actually sees.
        @assert norm(HybridZuptInsJl.stride_local(frame;
            R_wb=R_prev, ΔpΔθ3=[R_prev * Δp; Δθ3])[1] - s_l) < 1e-12 "$frame: stride mismatch"
    end
    @info "notes/013 §1.7 degenerate check passed for HEADING and BODY"
end

## 4. Run both filters on the same walk, same estimator configuration
runners = OrderedDict{String,Function}(
    "V2 absolute" => HybridZuptInsJl.hybrid_zupt_aided_insv2,
    "V3 stride" => HybridZuptInsJl.hybrid_zupt_aided_insv3,
)

nees_runs = OrderedDict{String,NamedTuple}()
cov_runs = OrderedDict{String,NamedTuple}()
diags = OrderedDict{String,HybridZuptInsJl.CorrectorDiagnostics}()
ios = OrderedDict{String,Any}()

@printf("%-14s %-6s %9s %9s %11s %10s %8s %12s %7s\n",
    "filter", "phase", "RMSE[m]", "yaw[rad]", "mean NEES", "med NEES",
    "in 95%", "tr(Σ_pp)", "n_foot")

for (name, run) in runners
    corrector = HybridZuptInsJl.DecoupledHsgpEstimator(
        n_alloc; params=params, corrected_channels=output_channels)
    diag = HybridZuptInsJl.CorrectorDiagnostics()

    _, _, _, io, _ = run(inertial, simdata, gt_traj, corrector;
        x_init=x_init,
        gt_available=[n <= n_train_cutoff for n in 1:N],
        ref_frame=FRAME, feature_type=FEATURE_TYPE,
        diagnostics=diag)

    diags[name] = diag
    ios[name] = io
    nees_runs[name] = HybridZuptInsJl.corrector_nees_series(diag, gt_traj)
    cov_runs[name] = HybridZuptInsJl.pos_cov_trace(diag)

    nees = nees_runs[name]
    gt_foot = gt_traj[diag.k]
    x_foot = hcat(diag.pos...)
    quat_foot = hcat(diag.quat...)
    trace = cov_runs[name].trace

    for (phase, idx) in (("train", findall(<(k0), diag.k)),
        ("test", findall(>=(k0), diag.k)))
        isempty(idx) && continue

        x9 = zeros(9, length(diag.k))
        x9[1:3, :] = x_foot
        rmse = HybridZuptInsJl.rmse_summary(x9, quat_foot, gt_foot; ks=idx)

        @printf("%-14s %-6s %9.4f %9.4f %11.2f %10.2f %7.1f%% %12.3e %7d\n",
            phase == "train" ? name : "", phase,
            rmse.pos, rmse.yaw,
            mean(view(nees.pos, idx)), median(view(nees.pos, idx)),
            100 * HybridZuptInsJl.consistency_ratio(
                view(nees.pos, idx), nees.lower, nees.upper),
            mean(view(trace, idx)), length(idx))
    end
end

## 5. Is the GP still predicting the same thing?
# The two filters relinearise differently, so the predictions cannot be
# identical -- but if V3's are a different *shape*, the reordering broke the
# feature, not the covariance.
let
    p2 = ios["V2 absolute"]["prediction"]
    p3 = ios["V3 stride"]["prediction"]
    n = min(size(p2.data, 2), size(p3.data, 2))
    @printf("\nGP prediction, test half (%d footfalls compared)\n", n)
    @printf("%-8s %12s %12s %12s\n", "channel", "V2 mean", "V3 mean", "rel |Δ|")
    for (i, ch) in enumerate(["pos_1", "pos_2", "pos_3", "yaw"])
        a = view(p2.data, i, 1:n)
        b = view(p3.data, i, 1:n)
        all(iszero, a) && all(iszero, b) && continue
        @printf("%-8s %12.5f %12.5f %12.4f\n", ch, mean(a), mean(b),
            norm(b .- a) / max(norm(a), eps()))
    end
end

## 6. Where does the reported stride covariance come from?
# notes/013 §1.3 keeps an input-noise term whose Σ_feature is fed in raw while
# ∂y∂z is a normalised-space Jacobian. In V2 that only mis-scaled a measurement
# noise; in V3 it is process noise that never gets pulled back. Print the three
# terms before deciding whether it matters.
let
    corrector = HybridZuptInsJl.DecoupledHsgpEstimator(
        n_alloc; params=params, corrected_channels=output_channels)
    HybridZuptInsJl.hybrid_zupt_aided_insv3(inertial, simdata, gt_traj, corrector;
        x_init=x_init, gt_available=[n <= n_train_cutoff for n in 1:N],
        ref_frame=FRAME, feature_type=FEATURE_TYPE)

    io = ios["V3 stride"]
    feat = io["input"]
    d = size(feat.data, 1)
    k = size(feat.data, 2)
    mask = corrector.correction_mask
    scale = Diagonal(params.output_stats[2][mask])

    acc = zeros(3, length(mask))
    for j in 1:k
        feature = collect(feat.data[:, j])
        Σ_feature = Matrix(Diagonal(collect(feat.data_std[:, j]) .^ 2))
        HybridZuptInsJl.normalize_feature!(FEATURE_TYPE;
            feature=feature, Σ_feature=Σ_feature,
            input_stats=params.input_stats, mid_norm=params.mid_norm)
        for output_d in eachindex(mask)
            for input_d in 1:params.d
                corrector.∂y∂z[output_d, input_d:input_d] =
                    HybridZuptInsJl.calc_eigenvectors_dx(
                        reshape(feature, 1, params.d), params.LL,
                        corrector.per_dim_eigvals, input_d) *
                    corrector.β[((output_d-1)*m+1):(output_d*m)]
            end
        end
        kron!(corrector.Φ, I(length(mask)),
            HybridZuptInsJl.calc_eigenvectors(
                reshape(feature, 1, params.d), params.LL, corrector.per_dim_eigvals))

        acc[1, :] += diag(scale * (corrector.Φ * corrector.Σβ * corrector.Φ') * scale)
        acc[2, :] += diag(scale * (corrector.∂y∂z * Σ_feature * corrector.∂y∂z') * scale)
        acc[3, :] += diag(scale * Diagonal(corrector.σ_n .^ 2) * scale)
    end
    acc ./= k

    @printf("\nMean per-stride variance by term, denormalised (%d footfalls)\n", k)
    @printf("%-8s %13s %13s %13s\n", "channel", "Φ Σβ Φ'", "∂y∂z Σz ∂y∂z'", "σ_n²")
    for (j, orig) in enumerate(mask)
        @printf("%-8s %13.3e %13.3e %13.3e\n",
            ["pos_1", "pos_2", "pos_3", "yaw"][orig], acc[1, j], acc[2, j], acc[3, j])
    end
end

## 7. Figures
const SECTION = "9_StrideLevelCorrection"
run_label = "$data_key trial $trial_id, train_ratio=$train_ratio — train: mocap, test: GP"

results_figure() do
    HybridZuptInsJl.plot_nees_comparison(
        nees_runs; block=:pos, yscale=log10, split_k=k0,
        legend_position=:lt,
        title="Corrector position NEES, $run_label",
        save_path=stamped(SECTION, "nees_pos_$(data_key)$(trial_id)"))
end

results_figure() do
    HybridZuptInsJl.plot_position_covariance(
        cov_runs; split_k=k0,
        title="Corrector position uncertainty, $run_label",
        save_path=stamped(SECTION, "pos_cov_trace_$(data_key)$(trial_id)"))
end
