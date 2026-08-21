# Shared preamble for the scripts/5Results/ figure generators.
#
# Every script here used to carry its own copy of the dataset dict, the
# hyperparameter-path dict, the trial-id lists and its own save/mkpath/theme
# incantation. That duplication is not cosmetic: it is how
# 3_yaw_channel-correlation_analysis.jl ended up running the DCSC trial list
# against the ANG2 dataset, how two scripts silently wrote no figure at all,
# and how the same experiment ended up under two different output roots.
#
# Include this after the module include, e.g.
#
#     include("../../src/HybridZuptInsJl.jl")
#     using .HybridZuptInsJl
#     include("_common.jl")
#
# Run from the repository root.

import CairoMakie, Dates

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

const DATA_DIRS = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2",
    "DCSC" => "data/dcsc_optitrack",
)

"""
Trial ids used for the cross-dataset comparisons. Defined once so the two
datasets cannot be mixed up: these lists are NOT interchangeable, and using the
DCSC list against ANG2 silently selects different walks rather than erroring.
"""
const TRIAL_IDS = Dict{String,Vector{Int}}(
    "ANG2" => [1, 2, 3, 4, 8, 9, 13, 15, 16],
    "DCSC" => [1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
)

data_dir(key::AbstractString) = DATA_DIRS[key]
trial_ids(key::AbstractString) = TRIAL_IDS[key]

# ---------------------------------------------------------------------------
# Trained hyperparameter sets
# ---------------------------------------------------------------------------

"""
Trained HSGP hyperparameter artifacts, keyed by the arbitrary integer used at
the top of each script. Keep the comment on each entry: the provenance of these
files is otherwise unrecoverable.
"""
const HSGP_PARAM_PATHS = Dict{Int,String}(
    41 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG23_HEADING_TWOD_STEP_YAW_2026-07-13T12:28:30.952.json",
    42 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-12T10:25:45.876.json",   # no output norm; trained on ANG2
    43 => "out/4OnlineCorrection/6_HypOpt/DCSC/HEADING-TWOD_STEP_YAW/DCSC_HEADING_TWOD_STEP_YAW_2026-08-12T10:41:50.718.json",   # no output norm; trained on DCSC
    44 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-19T12:56:03.443.json",   # as 42, higher lower noise bound
    45 => "out/4OnlineCorrection/6_HypOpt/ANG2/HEADING-TWOD_STEP_YAW/ANG2_HEADING_TWOD_STEP_YAW_2026-08-19T16:42:30.821.json",   # hand-tuned from 44
)

"""
    load_hsgp_params(key; m=200)

Load a trained hyperparameter set and its metadata, returning
`(params, frame, feature_type, meta)`. `m` overrides the basis count so every
script states the value it actually used rather than inheriting whatever was
stored.
"""
function load_hsgp_params(key::Int; m::Int=200)
    path = HSGP_PARAM_PATHS[key]
    isfile(path) || error("hyperparameter file for key $key not found: $path")
    params, meta, _ = HybridZuptInsJl.from_json(HybridZuptInsJl.HsgpParameters, path)
    params = HybridZuptInsJl.basecopy(params; new_m=m)
    frame = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.ReferenceFrame, meta["ref_frame"])
    feature = HybridZuptInsJl.string_to_enum(HybridZuptInsJl.FeatureType, meta["feature_type"])
    return params, frame, feature, meta
end

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

const RESULTS_ROOT = "out/Results"

"""
    results_path(section, filename) -> String

Absolute-ish path under `out/Results/<section>/`, creating the directory. All
five thesis sections write here, including the headline performance sweep which
previously wrote to out/4OnlineCorrection/4PerformanceResults and so was not
findable alongside the others.
"""
function results_path(section::AbstractString, filename::AbstractString)::String
    dir = joinpath(RESULTS_ROOT, section)
    mkpath(dir)
    return joinpath(dir, filename)
end

"""
    stamped(section, name; ext="svg") -> String

Timestamped output path. The timestamp goes *last* so `ls` sorts by experiment
first and chronology second, which is the order you actually browse in.
"""
function stamped(section::AbstractString, name::AbstractString; ext::AbstractString="svg")::String
    return results_path(section, "$(name)_$(Dates.now()).$(ext)")
end

# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------

"""
    results_figure(f)

Run `f` under the single theme used by every saved results figure. CairoMakie,
because these are vector figures destined for a thesis: GLMakie needs a live GL
context, so scripts that used it could not run headless and silently produced
raster-backed SVG when they did run.
"""
function results_figure(f::Function)
    # The package does `using GLMakie`, so GLMakie is the active backend by
    # default. Activate CairoMakie explicitly, otherwise a bare `save(path, fig)`
    # inside a plotting function is served by whichever backend happens to be
    # active -- which is how these scripts came to depend on the REPL having had
    # CairoMakie.activate!() typed into it at some earlier point.
    CairoMakie.activate!()
    return CairoMakie.with_theme(f, CairoMakie.theme_ggplot2())
end
