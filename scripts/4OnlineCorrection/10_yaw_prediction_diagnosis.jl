### Why does V3 over-correct yaw: is the prediction too large, or is it applied wrongly?
###
### notes/014. With the heading error of the corrector e_ψ = ψ_gt − ψ_c and the
### yaw target y₄ = Δψ_gt − Δψ_ins, any filter that injects a yaw correction u
### per stride obeys, footfall to footfall,
###
###     Δe_ψ = y₄ − u        ⇒   u_implied = y₄ − Δe_ψ
###
### so the injection each filter *actually* applied can be read back from its
### trajectory. For V3 it must equal the GP's ŷ₄ (else the injection is wrong);
### for ZUPT-only it must be ~0 (else the target is not the heading-error
### increment it claims to be). Given the identity holds, the question is
### whether Σŷ₄ overshoots Σy₄ over the test half, and why:
###   - train mean of y₄ vs test mean (non-stationarity of the drift),
###   - lag-1 autocorrelation of y₄ (≈ −0.5 → differenced jitter, telescopes),
###   - corr(ŷ₄, y₄) on the test half (does the input-dependent part carry skill),
###   - V3 Static vs V3 HSGP (a constant bias vs the GP).
### It also measures, per footfall, how far the V3 stride/target is from the one
### built in the INS's own frame (notes/014 T3, the size of fix F1), and the
### |yaw(R') + ψ| error of `stride_heading` (T4).
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
include("_stride_variants.jl")
using Printf, DataFrames
import CSV, CairoMakie

const SECTION = "10_YawDiagnosis"

m = 200
train_ratio = 0.3
output_channels = [:pos_1, :pos_2, :yaw]

runs = OrderedDict{String,Tuple{Function,Type}}(
    "ZUPT only" => (HybridZuptInsJl.hybrid_zupt_aided_insv2, HybridZuptInsJl.BaseEstimator),
    "V2 HSGP" => (HybridZuptInsJl.hybrid_zupt_aided_insv2, HybridZuptInsJl.DecoupledHsgpEstimator),
    "V3 HSGP" => (HybridZuptInsJl.hybrid_zupt_aided_insv3, HybridZuptInsJl.DecoupledHsgpEstimator),
    "V3 Static" => (HybridZuptInsJl.hybrid_zupt_aided_insv3, HybridZuptInsJl.DecoupledStaticEstimator),
    "V3-F1 HSGP" => (HybridZuptInsJl.hybrid_zupt_aided_insv3_insframe, HybridZuptInsJl.DecoupledHsgpEstimator),
)
# Target and feature of this run depend on the INS and GT only, so it is the
# reference the others' strides are compared against (T3).
const REF_RUN = "V3-F1 HSGP"

"Per-footfall ŷ₄, zero where the footfall took no prediction (train half)."
function footfall_prediction(io)
    tgt, pred = io["target"], io["prediction"]
    at = Dict(t => j for (j, t) in enumerate(pred.t))
    return [haskey(at, t) ? pred.data[4, at[t]] : 0.0 for t in tgt.t]
end

lag1(x) = cor(x[1:end-1], x[2:end])
nanmedian(x) = (v = filter(!isnan, x); isempty(v) ? NaN : median(v))

rows = DataFrame()

for (data_key, hsgp_key) in DATASET_HSGP_KEYS
    params, FRAME, FEATURE_TYPE, _ = load_hsgp_params(hsgp_key; m=m)
    aligned = HybridZuptInsJl.collect_aligned_trajectories(
        OrderedDict(data_key => (data_dir(data_key), trial_ids(data_key))))[data_key]

    for (trial_id, res) in aligned
        gt = res.gt_traj_aligned
        out = OrderedDict(name => run_variant(f, T, output_channels, res, params;
            frame=FRAME, feature_type=FEATURE_TYPE, train_ratio=train_ratio)
                          for (name, (f, T)) in runs)

        ref = out[REF_RUN]
        y4_ref = ref.io["target"].data[4, :]
        feat_ref = ref.io["input"].data

        # T4: heading-rotation error of `stride_heading` at the GT footfalls.
        tilt_err = [abs(HybridZuptInsJl.wrap_pi(
            HybridZuptInsJl.matrix_to_euler(gt.R_nb[:, :, k]')[3] +
            HybridZuptInsJl.matrix_to_euler(gt.R_nb[:, :, k])[3])) for k in ref.diag.k]

        @printf("\n%s trial %d  (%d footfalls, T4 |yaw(R')+ψ|: max %.4f, rms %.4f rad)\n",
            data_key, trial_id, length(ref.diag), maximum(tilt_err), sqrt(mean(abs2, tilt_err)))
        @printf("%-11s %6s %8s %8s %8s %8s %9s %8s %8s %8s %7s %7s %9s %9s %9s %8s\n",
            "run", "n_test", "e(k0)", "e_end", "Σy4", "Σŷ4", "Σu_impl", "max|u−ŷ|",
            "ȳ4 tr", "ȳ4 te", "ρ1(y4)", "ρ(ŷ,y)", "yawRMSEte", "Δstride", "Δȳ4 tr", "tilt tr")

        sums = OrderedDict{String,Any}()
        for (name, r) in out
            n = length(r.diag)
            y4 = r.io["target"].data[4, :]
            @assert length(y4) == n "$name: $(length(y4)) targets for $n footfalls"
            ŷ4 = footfall_prediction(r.io)
            e = footfall_yaw_error(r.diag, gt)
            u = [NaN; y4[2:end] .- HybridZuptInsJl.wrap_pi.(diff(e))]

            te = findall(>(r.n_train_cutoff), r.diag.k)
            tr = findall(<=(r.n_train_cutoff), r.diag.k)
            j0 = first(te) - 1          # last train footfall

            # Mechanism for T3: the corrector only observes position and yaw, so
            # its roll/pitch can wander from the INS's (which ZUPTs keep near
            # GT). A tilted corrector frame shifts yaw(q̂⊗Δq) − yaw(q̂) off Δψ_ins.
            tilt = [norm(HybridZuptInsJl.matrix_to_euler(HybridZuptInsJl.quat_to_matrix(q))[1:2] .-
                         HybridZuptInsJl.matrix_to_euler(gt.R_nb[:, :, k])[1:2])
                    for (k, q) in zip(r.diag.k, r.diag.quat)]

            # The stride fed to the GP, against the one built in the INS frame.
            Δstride = maximum(norm.(eachcol(r.io["input"].data .- feat_ref)))

            row = (; dataset=data_key, trial_id, run=name,
                n_train=length(tr), n_test=length(te),
                e_k0=e[j0], e_end=e[end],
                sum_y4=sum(y4[te]), sum_yhat4=sum(ŷ4[te]), sum_u_implied=sum(u[te]),
                max_identity_resid=maximum(abs.(u[te] .- ŷ4[te])),
                mean_y4_train=mean(y4[tr[2:end]]), mean_y4_test=mean(y4[te]),
                mean_yhat4_test=mean(ŷ4[te]), std_yhat4_test=std(ŷ4[te]),
                rho1_y4=lag1(y4[2:end]),
                corr_yhat_y=std(ŷ4[te]) > 0 ? cor(ŷ4[te], y4[te]) : NaN,
                rms_y4_test=sqrt(mean(abs2, y4[te])),
                rms_resid_test=sqrt(mean(abs2, y4[te] .- ŷ4[te])),
                yaw_rmse_test=sqrt(mean(abs2, e[te])),
                max_target_dev=maximum(abs.(y4 .- y4_ref)),
                mean_target_dev_train=mean(y4[tr[2:end]] .- y4_ref[tr[2:end]]),
                max_tilt_dev_train=maximum(tilt[tr]),
                max_stride_dev=Δstride,
                t4_max=maximum(tilt_err))
            push!(rows, row; cols=:union)

            @printf("%-11s %6d %8.4f %8.4f %8.4f %8.4f %9.4f %8.1e %8.5f %8.5f %7.2f %7.2f %9.4f %9.1e %9.5f %8.4f\n",
                name, row.n_test, row.e_k0, row.e_end, row.sum_y4, row.sum_yhat4,
                row.sum_u_implied, row.max_identity_resid, row.mean_y4_train,
                row.mean_y4_test, row.rho1_y4, row.corr_yhat_y, row.yaw_rmse_test, Δstride,
                row.mean_target_dev_train, row.max_tilt_dev_train)

            sums[name] = (; k=r.diag.k[te], e=e[te],
                cy=cumsum(y4[te]), cŷ=cumsum(ŷ4[te]), cu=cumsum(u[te]))
        end

        results_figure() do
            fig = CairoMakie.Figure(size=(1000, 700))
            ax1 = CairoMakie.Axis(fig[1, 1]; ylabel="running sum [rad]",
                title="$data_key trial $trial_id, test half — V3 HSGP: true drift Σy₄ vs injected Σŷ₄")
            s = sums["V3 HSGP"]
            CairoMakie.lines!(ax1, s.k, s.cy; label="Σ y₄ (true increment error)")
            CairoMakie.lines!(ax1, s.k, s.cŷ; label="Σ ŷ₄ (GP, injected)")
            CairoMakie.lines!(ax1, s.k, s.cu; linestyle=:dash, label="Σ u implied by trajectory")
            CairoMakie.lines!(ax1, s.k, s.cy .- s.cŷ; label="Σ (y₄ − ŷ₄)")
            CairoMakie.axislegend(ax1; position=:lt)

            ax2 = CairoMakie.Axis(fig[2, 1]; xlabel="IMU sample k", ylabel="e_ψ = ψ_gt − ψ [rad]")
            for name in ("ZUPT only", "V2 HSGP", "V3 HSGP", "V3 Static", "V3-F1 HSGP")
                CairoMakie.lines!(ax2, sums[name].k, sums[name].e; label=name)
            end
            CairoMakie.axislegend(ax2; position=:lt)
            CairoMakie.save(stamped(SECTION, "yaw_sums_$(data_key)$(trial_id)"), fig)
        end
    end
end

csv_path = stamped(SECTION, "yaw_diagnosis"; ext="csv")
CSV.write(csv_path, rows)
@info "Saved $csv_path"

## Summary across trials, per dataset and run
@printf("\n%-5s %-11s %6s %9s %9s %9s %10s %9s %8s %8s %9s\n",
    "data", "run", "trials", "Σy4", "Σŷ4", "|Σŷ|/|Σy|", "max|u−ŷ|",
    "ρ1(y4)", "ρ(ŷ,y)", "ȳtr−ȳte", "yawRMSE")
println("(medians across trials)")
for g in groupby(rows, [:dataset, :run]; sort=false)
    @printf("%-5s %-11s %6d %9.4f %9.4f %9.2f %10.1e %9.2f %8.2f %8.5f %9.4f\n",
        g.dataset[1], g.run[1], nrow(g), median(g.sum_y4), median(g.sum_yhat4),
        median(abs.(g.sum_yhat4) ./ abs.(g.sum_y4)), median(g.max_identity_resid),
        median(g.rho1_y4), nanmedian(g.corr_yhat_y),
        median(g.mean_y4_train .- g.mean_y4_test), median(g.yaw_rmse_test))
end
