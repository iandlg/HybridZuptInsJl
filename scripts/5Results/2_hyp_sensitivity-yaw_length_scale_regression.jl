# Section 2 (sensitivity), companion figure: what the yaw length scale actually
# does to the regressed yaw correction.
#
# WHY THIS EXISTS. Two results from neighbouring sections sit awkwardly together:
#
#   * 2_hyp_sensitivity-param_sensitivity.jl finds the yaw length scale ℓ_s to be
#     the most consequential hyperparameter in the sweep, and finds the trained
#     value sitting on a local minimum -- perturbing it either way moves RMSE.
#   * 3_yaw_channel-cst_v_hsgp_yaw_correction.jl finds that, at the trained
#     value, correcting only the yaw channel with the HSGP gives essentially the
#     same final RMSE as a constant (static) correction.
#
# Taken together those read as a contradiction: how can the most sensitive
# parameter be sitting at a value where the model it parameterises is worth no
# more than a constant? This figure is the hint at the answer -- it shows the
# regressed yaw output itself at ℓ_s below, at, and above the trained value,
# with the static correction on the same axes for reference.
#
# SCOPE. One trial, one noise realisation, three length scales. This is an
# illustration for the narrative, NOT a measurement: nothing here is averaged
# over trials and no claim about magnitudes should rest on it. The quantitative
# statements belong to the two scripts named above.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using OrderedCollections, Printf
using CairoMakie: rich, subscript, RGBAf
import GLMakie   # for the activate!() at the end; results_figure leaves CairoMakie active

## 1. Hyperparameters
hsgp_p_key = 42
m = 200
hsgp_p, FRAME, FEATURE_TYPE, meta = load_hsgp_params(hsgp_p_key; m=m)

## 2. Trial
# Same dataset/trial/split as the sensitivity sweep, so this figure illustrates
# the same point on the same data rather than a nearby one.
data_key = "ANG2"
data_dir_path = data_dir(data_key)
trial_id = 15
train_ratio = 0.45

# Only the yaw channel is corrected, matching section 3 -- that is the setting in
# which HSGP and static came out level.
output_channels = [:yaw]

## 3. Length scales to compare
# Index 2 of a channel's hyperparameter vector is the length scale
# (1 = σ_n, 2 = ℓ_s, 3 = σ_f; see _HP_SYMBOL_PARTS in Plotting/OnlineHpSensitivity.jl).
const LENGTH_SCALE_IDX = 2

# Decades either side of the trained value. The same ±1 decade the sensitivity
# sweep used (`log_range = (-1.0, 1.0)`), so the three points here are the
# endpoints and centre of that sweep's yaw ℓ_s axis.
log10_offsets = [-1.0, 0.0, 1.0]

base_ls = hsgp_p.hp.yaw[LENGTH_SCALE_IDX]

function params_with_yaw_length_scale(base::HybridZuptInsJl.HsgpParameters, ls::Float64)
    new_hp = HybridZuptInsJl.modify_sehp(base.hp, :yaw, LENGTH_SCALE_IDX, ls)
    return HybridZuptInsJl.basecopy(base; new_hp=new_hp)
end

## 4. Run the filter once per method
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated =
    HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir_path, trial_id)

x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    HybridZuptInsJl.matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
)

N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]
window = round(Int, N / 60)

# Static first so it reads as the reference the HSGP variants are compared against.
# Keys stay plain ASCII -- they index the colour and label maps below; the rendered
# names live in `series_labels`.
estimators = OrderedDict{String,HybridZuptInsJl.AbstractEstimator}(
    "Static" => HybridZuptInsJl.DecoupledStaticEstimator(window; corrected_channels=output_channels),
)
series_labels = Dict{String,Any}("Static" => "Static")

# Every HSGP variant is the SAME estimator at a different setting, so they share one
# colour and are told apart by linestyle. Three shades of green could not do this job:
# at the trained ℓ_s and above, the corrections collapse onto nearly the same flat line,
# and no colour separates two curves drawn on top of each other. Linestyle does, and it
# still works when the figure is printed in greyscale.
#
# The trained value is solid and heaviest; the two perturbations are dashed and dotted,
# thinner, and slightly faded. That ranks them -- one is the setting the model actually
# uses, the others are excursions from it -- rather than presenting three equals.
hsgp_color = HybridZuptInsJl.method_color("HSGP")
faded(c, a) = RGBAf(c.r, c.g, c.b, a)

variant_style = Dict(-1.0 => :dot, 0.0 => :solid, 1.0 => :dash)

series_styles = Dict{String,Any}("Static" => :solid)
series_widths = Dict{String,Any}("Static" => 2.0)
series_colors = Dict{String,Any}("Static" => HybridZuptInsJl.method_color("Static"))

for offset in log10_offsets
    mult = 10.0^offset
    ls = base_ls * mult
    key = "HSGP x$(mult)"
    trained = offset == 0

    estimators[key] = HybridZuptInsJl.DecoupledHsgpEstimator(window;
        params=params_with_yaw_length_scale(hsgp_p, ls), corrected_channels=output_channels)

    # Spelled with hp_multiplier_label, the same helper that labels the multiplier axis
    # in the sensitivity sweep figure, so "×0.1" here and "×0.1" there are the same point
    # and a reader can carry one figure onto the other.
    series_labels[key] = rich("HSGP ℓ", subscript("s"), " ",
        HybridZuptInsJl.hp_multiplier_label(mult),
        trained ? "" : "")
    series_styles[key] = get(variant_style, offset, :solid)
    series_widths[key] = trained ? 2.0 : 1.2
    series_colors[key] = trained ? hsgp_color : faded(hsgp_color, 0.95)
end

predictions = OrderedDict{String,HybridZuptInsJl.CorrectionIO}()
target = nothing

for (label, estimator) in estimators
    _, _, _, io_data, _ = HybridZuptInsJl.hybrid_zupt_aided_insv2(
        inertial_updated, sim_config_updated, gt_traj_aligned, estimator;
        x_init=x_init, gt_available=gt_available,
        ref_frame=FRAME, feature_type=FEATURE_TYPE)

    predictions[label] = io_data["prediction"]
    # The target is the measured stride error and does not depend on the estimator,
    # so any run supplies it; taking the first keeps them from silently disagreeing.
    isnothing(target) && (global target = io_data["target"])
end

## 5. Plot
const SECTION = "2_HypSensitivity/YawLengthScaleRegression"

# The test segment is the one the argument is about: on the training half every
# variant reproduces the data it was fitted to, so nothing distinguishes them there.
# clip_quantile scales the y-axis to the target's full range plus a 10% margin: at ℓ_s = base/10
# the GP extrapolates to tens of radians, and left to autoscale that one series
# flattens the other three onto the zero line -- the exact comparison this figure
# exists to show. The target is what the corrections are trying to reproduce, so its
# range is the scale they should be judged at; the divergent series simply runs off
# the top of the panel, which is itself the point being made about it.
results_figure() do
    HybridZuptInsJl.plot_regression_comparison(predictions, target;
        channel=4, segment=:test, train_ratio=train_ratio,
        colors=series_colors, labels=series_labels,
        linestyles=series_styles, linewidths=series_widths, clip_quantile=1.1,
        dataset=data_key, trial_id=trial_id,
        save_path=stamped(SECTION, "yaw_length_scale_$(data_key)$(trial_id)"))
end

# Companion: unclipped, with the predictive bands and the numbers, so the figure
# above is auditable rather than something the reader has to take on trust.
results_figure() do
    HybridZuptInsJl.plot_regression_comparison(predictions, target;
        channel=4, segment=:test, train_ratio=train_ratio,
        colors=series_colors, labels=series_labels,
        linestyles=series_styles, linewidths=series_widths,
        show_std=true, show_rmse=true, show_mean_std=true,
        dataset=data_key, trial_id=trial_id,
        save_path=stamped(SECTION, "yaw_length_scale_$(data_key)$(trial_id)_unclipped"))
end

# Zoomed view: 40 s of the test segment, enough strides to see the shape of each
# correction without the whole segment compressed into a few hundred pixels.
results_figure() do
    HybridZuptInsJl.plot_regression_comparison(predictions, target;
        channel=4, segment=:test, train_ratio=train_ratio,
        colors=series_colors, labels=series_labels,
        linestyles=series_styles, linewidths=series_widths, clip_quantile=1.1,
        time_window=(400.0, 440.0),
        dataset=data_key, trial_id=trial_id, show_std=false,
        save_path=stamped(SECTION, "yaw_length_scale_$(data_key)$(trial_id)_zoom"))
end
GLMakie.activate!()

