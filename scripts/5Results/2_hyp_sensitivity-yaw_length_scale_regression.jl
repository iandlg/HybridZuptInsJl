# Section 2 (sensitivity), companion figure: the regressed yaw correction at the yaw
# length scale below, at and above the trained value, with the static correction for
# reference. Illustrates why ℓ_SE is the most sensitive hyperparameter
# (2_hyp_sensitivity-param_sensitivity.jl) although HSGP yaw ≈ static yaw in final RMSE
# (3_yaw_channel-cst_v_hsgp_yaw_correction.jl). One trial: an illustration, not a measurement.

using StrideGP
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
train_ratio = 0.3

# Only the yaw channel is corrected, matching section 3 -- that is the setting in
# which HSGP and static came out level.
output_channels = [:pos_1, :pos_2, :yaw]

## 3. Length scales to compare
# Index 2 of a channel's hyperparameter vector is the length scale
# (1 = σ_n, 2 = ℓ_SE, 3 = σ_SE; see _PARAM_KINDS in Plotting/OnlineHpSensitivity.jl).
const LENGTH_SCALE_IDX = 2

# Decades either side of the trained value. The same ±1 decade the sensitivity
# sweep used (`log_range = (-1.0, 1.0)`), so the three points here are the
# endpoints and centre of that sweep's yaw ℓ_SE axis.
log10_offsets = [-1.0, 0.0, 1.0]

base_ls = hsgp_p.hp.yaw[LENGTH_SCALE_IDX]

function params_with_yaw_length_scale(base::StrideGP.HsgpParameters, ls::Float64)
    new_hp = StrideGP.modify_sehp(base.hp, :yaw, LENGTH_SCALE_IDX, ls)
    return StrideGP.basecopy(base; new_hp=new_hp)
end

## 4. Run the filter once per method
ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial_updated, sim_config_updated =
    StrideGP.compute_aligned_ins_trajectory(data_dir_path, trial_id)

x_init = vcat(
    ins_traj_aligned.pos[:, 1],
    ins_traj_aligned.vel[:, 1],
    StrideGP.matrix_to_euler(ins_traj_aligned.R_nb[:, :, 1])
)

N = length(inertial_updated)
n_train_cutoff = floor(Int, train_ratio * N)
gt_available = [n <= n_train_cutoff for n in 1:N]
window = round(Int, N / 60)

# Correction filter (see CORRECTION_FILTERS in _common.jl). Its tag goes into
# every output file name, and picks the correctors below (CORRECTORS).
filter_tag = "V4"

# Static first so it reads as the reference the HSGP variants are compared against.
# Keys stay plain ASCII -- they index the colour and label maps below; the rendered
# names live in `series_labels`.
estimators = OrderedDict{String,StrideGP.AbstractEstimator}(
    "Static" => CORRECTORS[filter_tag].static(window; params=hsgp_p, corrected_channels=output_channels),
)
series_labels = Dict{String,Any}("Static" => "Static")

# Every HSGP variant is the SAME estimator at a different setting, so they share one
# colour and are told apart by linestyle. Three shades of green could not do this job:
# at the trained ℓ_SE and above, the corrections collapse onto nearly the same flat line,
# and no colour separates two curves drawn on top of each other. Linestyle does, and it
# still works when the figure is printed in greyscale.
#
# The trained value is solid and heaviest; the two perturbations are dashed and dotted,
# thinner, and slightly faded. That ranks them -- one is the setting the model actually
# uses, the others are excursions from it -- rather than presenting three equals.
hsgp_color = StrideGP.method_color("HSGP")
faded(c, a) = RGBAf(c.r, c.g, c.b, a)

variant_style = Dict(-1.0 => :dot, 0.0 => :solid, 1.0 => :dash)

series_styles = Dict{String,Any}("Static" => :solid)
series_widths = Dict{String,Any}("Static" => 2.0)
series_colors = Dict{String,Any}("Static" => StrideGP.method_color("Static"))

for offset in log10_offsets
    mult = 10.0^offset
    ls = base_ls * mult
    key = "HSGP x$(mult)"
    trained = offset == 0

    estimators[key] = CORRECTORS[filter_tag].hsgp(window;
        params=params_with_yaw_length_scale(hsgp_p, ls), corrected_channels=output_channels)

    # Spelled with hp_multiplier_label, the same helper that labels the multiplier axis
    # in the sensitivity sweep figure, so "×0.1" here and "×0.1" there are the same point
    # and a reader can carry one figure onto the other.
    series_labels[key] = rich("HSGP ℓ", subscript("s"), " ",
        StrideGP.hp_multiplier_label(mult),
        trained ? "" : "")
    series_styles[key] = get(variant_style, offset, :solid)
    series_widths[key] = trained ? 2.0 : 1.2
    series_colors[key] = trained ? hsgp_color : faded(hsgp_color, 0.95)
end

predictions = OrderedDict{String,StrideGP.CorrectionIO}()
target = nothing

for (label, estimator) in estimators
    _, _, _, io_data, _ = CORRECTION_FILTERS[filter_tag](
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
# clip_quantile scales the y-axis to the target's full range plus a 10% margin: at ℓ_SE = base/10
# the GP extrapolates to tens of radians, and left to autoscale that one series
# flattens the other three onto the zero line -- the exact comparison this figure
# exists to show. The target is what the corrections are trying to reproduce, so its
# range is the scale they should be judged at; the divergent series simply runs off
# the top of the panel, which is itself the point being made about it.
results_figure() do
    StrideGP.plot_regression_comparison(predictions, target;
        channel=4, segment=:full, train_ratio=train_ratio,
        colors=series_colors, labels=series_labels,
        linestyles=series_styles, linewidths=series_widths, clip_quantile=1.1,
        dataset=data_key, trial_id=trial_id,
        save_path=stamped(SECTION, "yaw_length_scale_$(filter_tag)_process_only_key$(hsgp_p_key)_$(data_key)$(trial_id)"))
end

# Zoomed view: 40 s of the test segment, enough strides to see the shape of each
# correction without the whole segment compressed into a few hundred pixels.
results_figure() do
    StrideGP.plot_regression_comparison(predictions, target;
        channel=4, segment=:full, train_ratio=train_ratio,
        colors=series_colors, labels=series_labels,
        linestyles=series_styles, linewidths=series_widths, clip_quantile=0.95,
        figsize=(900, 300),
        dataset=data_key, trial_id=trial_id, show_std=false,
        save_path=stamped(SECTION, "yaw_length_scale_$(filter_tag)_process_only_key$(hsgp_p_key)_$(data_key)$(trial_id)_zoom"))
end
GLMakie.activate!()

