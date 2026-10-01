### Stride-level correction variants against the V2 baseline, across trials.
###
### Every variant runs on every trial of ANG2 and DCSC at each train_ratio, and
### is scored on the test half exactly as `run_online_correction_sweep` scores
### it (horizontal `rmse`, `rmse_rate`, `rmse_yaw` over the footfalls after the
### cutoff), plus the corrector's position NEES on the same footfalls. The sweep
### itself is hardwired to V2, which is why this loops by hand. notes/014.
###
### The paired table and figures are per-trial changes against "V2 HSGP" on the
### same (trial, train_ratio): negative = lower than V2. For mean NEES "lower" is
### only "better" while it stays above the χ²(3) mean of 3.
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("../5Results/_common.jl")
include("_stride_variants.jl")
using Printf, DataFrames
import CSV

const SECTION = "11_StrideCorrectionBenchmark"
const REFERENCE = "V2 HSGP"

# Set to the file name of a scores CSV under out/Results/11_StrideCorrectionBenchmark/
# to re-plot a finished run instead of recomputing it.
results_csv = nothing

m = 200
train_ratios = [0.3, 0.5]
all_ch = [:pos_1, :pos_2, :yaw]
pos_ch = [:pos_1, :pos_2]

V2 = HybridZuptInsJl.hybrid_zupt_aided_insv2
V3 = HybridZuptInsJl.hybrid_zupt_aided_insv3
Hsgp = HybridZuptInsJl.DecoupledHsgpEstimator
Static = HybridZuptInsJl.DecoupledStaticEstimator

# V3 is the INS-frame stride formulation (F1 in notes/014). The corrector-frame
# version it replaced is gone; its numbers are recorded in notes/014 §4.
variants = OrderedDict{String,Tuple{Function,Type,Vector{Symbol}}}(
    "ZUPT only" => (V2, HybridZuptInsJl.BaseEstimator, all_ch),
    "V2 HSGP" => (V2, Hsgp, all_ch),
    "V2 Static" => (V2, Static, all_ch),
    "V3 HSGP" => (V3, Hsgp, all_ch),
    "V3 Static" => (V3, Static, all_ch),
    "V3 HSGP no-yaw" => (V3, Hsgp, pos_ch),
)

if isnothing(results_csv)
    rows = DataFrame()
    for (dataset_order, (data_key, hsgp_key)) in enumerate(DATASET_HSGP_KEYS)
        params, FRAME, FEATURE_TYPE, _ = load_hsgp_params(hsgp_key; m=m)
        aligned = HybridZuptInsJl.collect_aligned_trajectories(
            OrderedDict(data_key => (data_dir(data_key), trial_ids(data_key))))[data_key]

        for (trial_id, res) in aligned, (tr_order, train_ratio) in enumerate(train_ratios)
            for (est_order, (name, (fn, T, channels))) in enumerate(variants)
                r = try
                    run_variant(fn, T, channels, res, params;
                        frame=FRAME, feature_type=FEATURE_TYPE, train_ratio=train_ratio)
                catch e
                    @warn "Skipping $data_key/$trial_id/$train_ratio/$name" exception = e
                    continue
                end

                # Same scoring as run_online_correction_sweep (DataProcessing.jl).
                gt_step = res.gt_traj_aligned[r.step_seg]
                cut = floor(Int, train_ratio * length(gt_step))
                _rmse = HybridZuptInsJl.rmse(r.traj[cut:end], gt_step[cut:end])[end]
                _rate = _rmse / HybridZuptInsJl.total_distance(gt_step[cut:end])
                _yaw = HybridZuptInsJl.rmse_yaw(r.traj[cut:end], gt_step[cut:end])[end]

                nees = HybridZuptInsJl.corrector_nees_series(r.diag, res.gt_traj_aligned)
                te = findall(>(r.n_train_cutoff), r.diag.k)
                trace = HybridZuptInsJl.pos_cov_trace(r.diag).trace

                push!(rows, (; dataset_name=data_key, dataset_order, trial_id,
                        train_ratio, train_ratio_order=tr_order,
                        estimator=name, estimator_order=est_order,
                        noise_spec_tag="NoiseSpec", noise_spec_order=1, seed=123,
                        rmse=_rmse, rmse_rate=_rate, rmse_yaw=_yaw,
                        mean_nees=mean(nees.pos[te]), med_nees=median(nees.pos[te]),
                        in95=HybridZuptInsJl.consistency_ratio(nees.pos[te], nees.lower, nees.upper),
                        final_tr_pp=trace[end]); cols=:union)
                @printf("%-5s %3d %.1f %-18s rmse %.4f yaw %.4f NEES %.2f\n",
                    data_key, trial_id, train_ratio, name, _rmse, _yaw, mean(nees.pos[te]))
            end
        end
    end
    csv_path = stamped(SECTION, "scores"; ext="csv")
    CSV.write(csv_path, rows)
    @info "Saved scores table: $csv_path"
else
    csv_path = results_path(SECTION, results_csv)
    rows = CSV.read(csv_path, DataFrame)
end

## Absolute medians per variant
println("\nMedian across trials (test half)")
@printf("%-5s %-4s %-18s %4s %9s %10s %9s %9s %8s %7s\n", "data", "tr", "variant", "n",
    "rmse[m]", "rate", "yaw[rad]", "meanNEES", "medNEES", "in95")
for g in groupby(rows, [:dataset_name, :train_ratio, :estimator]; sort=false)
    @printf("%-5s %-4.1f %-18s %4d %9.4f %10.5f %9.4f %9.2f %8.2f %6.1f%%\n",
        g.dataset_name[1], g.train_ratio[1], g.estimator[1], nrow(g),
        median(g.rmse), median(g.rmse_rate), median(g.rmse_yaw),
        median(g.mean_nees), median(g.med_nees), 100 * median(g.in95))
end

## Paired against V2 HSGP: median % change and #trials improved
println("\nPaired against \"$REFERENCE\": median % change [trials improved / total]")
@printf("%-5s %-4s %-18s %22s %22s %22s\n", "data", "tr", "variant", "rmse", "rmse_yaw", "mean_nees")
paired = OrderedDict(metric => HybridZuptInsJl.paired_estimator_contrast(rows;
    metric=metric, reference_estimator=REFERENCE) for metric in (:rmse, :rmse_yaw, :mean_nees))
for g in groupby(paired[:rmse], [:dataset_name, :train_ratio, :estimator]; sort=false)
    cells = map((:rmse, :rmse_yaw, :mean_nees)) do metric
        p = paired[metric]
        s = p[(p.dataset_name .== g.dataset_name[1]) .& (p.train_ratio .== g.train_ratio[1]) .&
              (p.estimator .== g.estimator[1]), :]
        @sprintf("%+8.1f%% [%2d/%2d]", median(s.rel_change_pct), count(<(0), s.delta), nrow(s))
    end
    @printf("%-5s %-4.1f %-18s %22s %22s %22s\n",
        g.dataset_name[1], g.train_ratio[1], g.estimator[1], cells...)
end

## Figures: per-trial change against V2 HSGP, grouped by train_ratio
for metric in (:rmse, :rmse_yaw), data_key in unique(rows.dataset_name)
    results_figure() do
        HybridZuptInsJl.plot_train_ratio_paired_relative_change(
            paired[metric], data_key; metric=metric, show_points=true,
            save_path=stamped(SECTION, "paired_$(metric)_$(data_key)_vs_V2"))
    end
end
