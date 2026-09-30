"""
Hyperparameter sensitivity curves.

NOTE: for a sweep carrying `probe`/`probe_kind` columns, use `OnlineHpProbe.jl`
instead. The multiplier axis these functions recover as `tested_value /
base_value` is only meaningful for a scale parameter; a location parameter swept
additively needs the offset axis it was swept on. This file is kept for the
sweeps and saved CSVs that predate that distinction.

Both the full grid ([`plot_hp_sensitivity`](@ref)) and the single-parameter
close-up ([`plot_hp_param_sensitivity`](@ref)) draw the same curve through
`_draw_hp_panel!`, so a panel of the grid and the close-up of the same parameter
cannot disagree about axes or units.

Two conventions, shared by both:

* **x is the multiplier applied to the tested hyperparameter**, on a log-scaled
  axis with a tick at each value actually swept. `vary_hsgp_parameters` samples
  geometrically around the base value (`log_around`), so the spacing is the
  sweep's own spacing; only the labels change, from exponents to `×0.32`, `×1`,
  `×3.16`. The axis previously read `log₁₀(relative change)`, which was doubly
  misleading: the quantity plotted was `log10(tested/base)`, a multiplier, and
  `relative_change` is the name of a different column entirely.
* **y is the RMSE change against the baseline, in percent** -- the
  `relative_change` column scaled by 100, so 0% is the untouched hyperparameter
  and −60% means the perturbation cut RMSE by well over half. The grid used to
  plot `rmse_ratio` (1.0 = baseline), which is the same information on a scale
  that has to be translated before it can be quoted.
"""

"""
Math symbols for the three hyperparameters, keyed by their position in a
channel's parameter vector (the same index `_HYPERPARAM_TYPES` is keyed by, and
the one that appears in a parameter name like `yaw[2]`).

Written as `(base, subscript)` pairs rather than plain strings because Unicode
has no subscript `f`, so `σ_f` cannot be spelled with combining characters the
way `σₙ` can. Makie's `rich`/`subscript` renders all three consistently.
"""
const _HP_SYMBOL_PARTS = Dict{Int,Tuple{String,String}}(
    1 => ("σ", "n"),   # noise
    2 => ("ℓ", "s"),   # length scale
    3 => ("σ", "f"),   # signal variance
)

"""
Hyperparameter type name -> its index in a channel's parameter vector. The
inverse of `_HYPERPARAM_TYPES`, so the mapping is stated once.
"""
const _HP_TYPE_INDEX = Dict{String,Int}(v => k for (k, v) in _HYPERPARAM_TYPES)

"""
One colour per hyperparameter type, fixed by type rather than by plotting order,
so ℓ_s is the same colour in every sensitivity figure regardless of which
parameters a given sweep happened to contain. Keyed by the index in a channel's
parameter vector (1 = σ_n, 2 = ℓ_s, 3 = σ_f).

These are Wong 4-6. Wong 1-3 are spoken for: they are what the estimator figures
in `scripts/5Results/` give to the three correction types (ZUPT only / Static /
HSGP), and reusing them here would put the same colour on a correction method in
one figure and a hyperparameter in the next. Wong 7 is the yellow, which is too
light to read as a thin line on the light background these figures use.
"""
const _HP_COLOR_INDICES = Dict{Int,Int}(
    1 => 4,   # σ_n  -> Wong 4, #CC79A7
    2 => 5,   # ℓ_s  -> Wong 5, #56B4E9
    3 => 6,   # σ_f  -> Wong 6, #D55E00
)

const _HP_FALLBACK_COLOR = Makie.RGBAf(0.45, 0.45, 0.45, 1.0)

"""
Display data for the normalisation statistics `make_stats_param_grid` sweeps
alongside the GP hyperparameters: the input mean and std that standardise a
feature before it reaches the kernel, the centering offset (`mid_norm`) that
places the HSGP domain around it, and the output mean/std that scale the
prediction back.

These are swept in the same units and against the same baseline as the
hyperparameters, so a sweep can contain both families and the signed-range
figure can rank them against each other. Without an entry here they fall back to
their raw name in a neutral grey -- readable, but three grey rows called
`input_mean[1]`, `input_std[1]`, `input_center[1]` cannot be told apart at a
glance, which is the whole point of that figure.

Keyed by the *name prefix* the sweep emits (`input_mean[2]` -> `"input_mean"`),
which is also the `type` column for every family except the centering term; see
`_STAT_TYPE_ALIASES`.

Colours are Tol muted, not more of the Wong palette: Wong 1-3 belong to the
correction methods in `scripts/5Results/`, 4-6 to the three hyperparameters
above, and 7 (yellow) is too light to read. A figure holding both families
therefore carries eight series with no two sharing a colour.
"""
const _STAT_PARAM_INFO = Dict{String,@NamedTuple{sym::String,sub::String,words::String,color::String}}(
    "input_mean" => (sym="μ", sub="x", words="input mean", color="#332288"),
    "input_std" => (sym="σ", sub="x", words="input std", color="#44AA99"),
    "input_center" => (sym="c", sub="x", words="input centering", color="#999933"),
    "output_mean" => (sym="μ", sub="y", words="output mean", color="#882255"),
    "output_std" => (sym="σ", sub="y", words="output std", color="#AA4499"),
)

"""
`type` strings that name the same family as a different `_STAT_PARAM_INFO` key.

The centering sweep is *named* `input_center[d]` but *typed* `mid_norm`, after
the field it writes (`HsgpParameters.mid_norm`). Both spellings appear in saved
CSVs, so both resolve here rather than one of them dropping to grey.
"""
const _STAT_TYPE_ALIASES = Dict{String,String}("mid_norm" => "input_center")

"""
    _stat_info(key) -> NamedTuple or nothing

Look up a statistics family by its name prefix or `type` string, resolving
[`_STAT_TYPE_ALIASES`](@ref). `nothing` for anything that is not one.
"""
function _stat_info(key::AbstractString)
    k = String(key)
    k = get(_STAT_TYPE_ALIASES, k, k)
    return get(_STAT_PARAM_INFO, k, nothing)
end

"""
    hp_type_color(idx_or_type)

Colour for a swept parameter type, given either a hyperparameter's index (1/2/3)
or a type name -- `"noise"` / `"length_scale"` / `"signal_variance"` for the GP
hyperparameters, or one of the [`_STAT_PARAM_INFO`](@ref) families for the
normalisation statistics. Anything else gets a neutral grey rather than silently
borrowing another type's colour.
"""
hp_type_color(idx::Int) =
    haskey(_HP_COLOR_INDICES, idx) ? Makie.wong_colors()[_HP_COLOR_INDICES[idx]] : _HP_FALLBACK_COLOR
function hp_type_color(type::AbstractString)
    info = _stat_info(type)
    isnothing(info) || return Makie.RGBAf(Makie.to_color(info.color))
    return hp_type_color(get(_HP_TYPE_INDEX, String(type), 0))
end

"""
    hp_param_color(name)

Colour for a sweep parameter such as `"yaw[2]"` or `"input_std[3]"`, resolved
through its type so that every figure agrees. Same channel guard as
[`hp_param_label`](@ref).
"""
function hp_param_color(name::AbstractString)
    parsed = _parse_hp_param(name)
    isnothing(parsed) && return _HP_FALLBACK_COLOR
    channel, idx = parsed
    isnothing(_stat_info(channel)) || return hp_type_color(channel)
    channel in _OUTPUT_NAMES || return _HP_FALLBACK_COLOR
    return hp_type_color(idx)
end

"""
    _parse_hp_param(name) -> (channel, idx) or nothing

Split a parameter name such as `"yaw[2]"` or `"input_mean[3]"` into its prefix
and bracketed index. What the index *means* depends on the prefix -- a
hyperparameter kind for an output channel, a feature dimension for a statistics
family -- so callers must resolve the prefix before using it. Returns `nothing`
for anything not of that shape (`"baseline"`), so callers can fall back to
showing the raw name.
"""
function _parse_hp_param(name::AbstractString)
    mt = match(r"^(.*)\[(\d+)\]$", name)
    isnothing(mt) && return nothing
    return (String(mt.captures[1]), parse(Int, mt.captures[2]))
end

"""
    hp_param_label(name; with_channel=true) -> rich text

Display label for a sweep parameter: the parameter's math symbol with what it
belongs to -- `yaw[2]` renders as `ℓ_s (yaw)` (hyperparameter, output channel),
`input_std[3]` as `σ_x [3]` (normalisation statistic, feature dimension). Falls
back to the raw name for anything outside both schemes.

`make_stats_param_grid` emits names of the same *shape* -- `input_mean[1]`,
`output_std[2]` -- where the bracketed number is a feature dimension, not a
hyperparameter kind. They are resolved through [`_STAT_PARAM_INFO`](@ref) first
for exactly that reason: matched against `_HP_SYMBOL_PARTS` instead, `input_mean[1]`
would be relabelled σ_n, which is wrong rather than merely ugly. A channel that
is neither a statistics family nor one of `_OUTPUT_NAMES` keeps its raw name.
"""
function hp_param_label(name::AbstractString; with_channel::Bool=true)
    parsed = _parse_hp_param(name)
    isnothing(parsed) && return rich(String(name))
    channel, idx = parsed
    # Statistics first: their names have the same `prefix[idx]` shape, but the
    # bracketed number is a feature dimension, so it stays in the label instead
    # of selecting a hyperparameter symbol.
    info = _stat_info(channel)
    isnothing(info) || return with_channel ?
                              rich(info.sym, subscript(info.sub), " [$idx]") :
                              rich(info.sym, subscript(info.sub))
    (channel in _OUTPUT_NAMES && haskey(_HP_SYMBOL_PARTS, idx)) || return rich(String(name))
    base, sub = _HP_SYMBOL_PARTS[idx]
    return with_channel ?
           rich(base, subscript(sub), " ($channel)") :
           rich(base, subscript(sub))
end

"""
    _hp_type_label(type) -> rich text

Legend label for a swept `type` (`"length_scale"`, `"input_std"`, ...): the math
symbol followed by the name in words, e.g. `ℓ_s  length scale` or
`σ_x  input std`. Unknown types pass through unchanged.
"""
function _hp_type_label(type::AbstractString)
    info = _stat_info(type)
    isnothing(info) || return rich(info.sym, subscript(info.sub), "  ", info.words)
    key = get(_HP_TYPE_INDEX, String(type), 0)
    haskey(_HP_SYMBOL_PARTS, key) || return rich(String(type))
    base, sub = _HP_SYMBOL_PARTS[key]
    return rich(base, subscript(sub), "  ", replace(String(type), "_" => " "))
end

"""
Display order for the normalisation statistics: inputs before outputs, and within
each the mean, the spread, then the domain offset. `_STAT_PARAM_INFO` is a `Dict`
and so carries no order of its own.
"""
const _STAT_TYPE_ORDER = String[
    "input_mean", "input_std", "input_center", "output_mean", "output_std",
]

"""
    _hp_type_rank(type) -> (family, position)

Sort key putting the two swept families in a fixed order rather than in whatever
order the bars happened to land in: the GP hyperparameters first (σ_n, ℓ_s, σ_f,
by their index in a channel's parameter vector), then the normalisation
statistics in [`_STAT_TYPE_ORDER`](@ref), then anything unrecognised.

A sweep holds both families and `plot_signed_relative_change` sorts its rows by
span, so the `type`s appear interleaved down the axis. Reading the legend in that
order means reading a hyperparameter, a statistic, another hyperparameter -- the
legend is the one place the two families are named, so it is the place to keep
them apart.
"""
function _hp_type_rank(type::AbstractString)
    key = String(type)
    key = get(_STAT_TYPE_ALIASES, key, key)
    hp_idx = get(_HP_TYPE_INDEX, key, nothing)
    isnothing(hp_idx) || return (1, hp_idx)
    stat_pos = findfirst(==(key), _STAT_TYPE_ORDER)
    isnothing(stat_pos) || return (2, stat_pos)
    return (3, 0)
end

"""
    _hp_ordered_types(types) -> Vector

`types` in [`_hp_type_rank`](@ref) order. Ties keep their relative input order --
the position is carried in the sort key rather than left to the algorithm's
stability -- so a sweep carrying a type this file has never heard of still gets
one legend entry per type rather than losing or reordering it silently.
"""
function _hp_ordered_types(types)
    ts = collect(types)
    return ts[sortperm(eachindex(ts); by=i -> (_hp_type_rank(ts[i]), i))]
end

"""
    _hp_tick_positions(mults; max_ticks=7) -> Vector{Float64}

Which of the swept multipliers get a labelled tick. A tick per swept value is
right for the 3-5 step sweeps, but a 15- or 21-step sweep overruns the axis and
the labels collide, so beyond `max_ticks` this thins them out.

Thinning is anchored on the ×1 tick and steps outwards by a constant stride, so
the baseline is always labelled and the kept ticks stay evenly spaced (the sweep
is geometric, so constant stride is constant distance on the log axis). The two
endpoints are added back when they are not already kept, since they are the
extremes of the range the figure is claiming -- but only when doing so leaves a
gap, otherwise re-adding them recreates the collision being avoided.
"""
function _hp_tick_positions(mults::AbstractVector{<:Real}; max_ticks::Int=7)
    pos = sort(unique(float.(mults)))
    n = length(pos)
    (n <= max_ticks || max_ticks < 2) && return pos

    anchor = argmin(abs.(log.(pos)))          # the ×1 tick, or the nearest to it
    stride = cld(n - 1, max_ticks - 1)
    idx = sort(unique(vcat(collect(anchor:(-stride):1), collect(anchor:stride:n))))

    for e in (1, n)
        if !(e in idx) && minimum(abs.(idx .- e)) > 1
            push!(idx, e)
        end
    end
    return pos[sort!(idx)]
end

"""
    _hp_multiplier_ticks(mults; max_ticks=7) -> (positions, labels)

Labelled ticks for the multiplier axis, `×<value>`. The exact baseline tick is
labelled `×1` rather than `×1.0`, since it is the reference the reader looks for
first. See [`_hp_tick_positions`](@ref) for which values get a label.
"""
function _hp_multiplier_ticks(mults::AbstractVector{<:Real}; max_ticks::Int=7)
    pos = _hp_tick_positions(mults; max_ticks=max_ticks)
    return (pos, map(hp_multiplier_label, pos))
end

"""
    hp_multiplier_label(m) -> String

A single multiplier rendered the way the sensitivity axis renders it: `×0.1`, `×1`,
`×10`. The exact baseline is `×1`, not `×1.0`, since it is the reference the reader
looks for first.

Public, and separate from [`_hp_multiplier_ticks`](@ref), so a figure that labels
individual series by their multiplier spells them exactly as the axis of the sweep
figure does -- that identity is what lets a reader carry a point from one to the other.
"""
function hp_multiplier_label(m::Real)::String
    isapprox(m, 1.0; rtol=1e-6) && return "×1"
    r = round(m; sigdigits=3)
    # Whole multipliers lose the trailing `.0`: `×10`, not `×10.0`. Same reasoning as
    # the `×1` case -- these are the round numbers a reader anchors on.
    return isinteger(r) && abs(r) < 1e15 ? "×$(Int(r))" : "×$r"
end

const _HP_PCT_TICKFORMAT = vs -> [string(round(v; digits=1), "%") for v in vs]

"""
    hp_param_name(channel, kind) -> String

The `parameter` string `vary_hsgp_parameters` uses for one hyperparameter, e.g.
`hp_param_name(:yaw, :length_scale) == "yaw[2]"`. Spelling the name out beats
hard-coding the index at the call site, where `"yaw[2]"` gives the reader no way
to tell a length scale from a signal variance.

`kind` is one of `:noise`, `:length_scale`, `:signal_variance` (the values of
`_HYPERPARAM_TYPES`).
"""
function hp_param_name(channel::Union{Symbol,AbstractString}, kind::Union{Symbol,AbstractString})::String
    want = String(kind)
    # The index in the parameter name is the hyperparameter's position in the
    # channel vector, which is what _HYPERPARAM_TYPES is keyed by.
    key = get(_HP_TYPE_INDEX, want, nothing)
    isnothing(key) && throw(ArgumentError(
        "unknown hyperparameter kind \"$want\"; have $(sort(collect(keys(_HP_TYPE_INDEX))))"))
    return "$(String(channel))[$key]"
end

"""
    stat_param_name(family, dim) -> String

The `parameter` string `vary_hsgp_parameters` uses for one normalisation
statistic, e.g. `stat_param_name(:input_std, 2) == "input_std[2]"`. The
counterpart to [`hp_param_name`](@ref) for the other family
`make_stats_param_grid` sweeps, so a script naming a focus parameter never has
to hard-code a bracketed index whose meaning differs between the two families.

`family` is one of the [`_STAT_PARAM_INFO`](@ref) keys -- `:input_mean`,
`:input_std`, `:input_center`, `:output_mean`, `:output_std` -- or an alias of
one (`:mid_norm` resolves to `:input_center`, matching the `type` string older
CSVs carry). `dim` is a *feature dimension* for the input families and an output
channel index for the output ones; unlike `hp_param_name`'s index it selects
which quantity is perturbed, not which kind of parameter it is.
"""
function stat_param_name(family::Union{Symbol,AbstractString}, dim::Integer)::String
    key = String(family)
    key = get(_STAT_TYPE_ALIASES, key, key)
    haskey(_STAT_PARAM_INFO, key) || throw(ArgumentError(
        "unknown statistics family \"$(family)\"; have $(sort(collect(keys(_STAT_PARAM_INFO))))"))
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    return "$key[$dim]"
end

"""
    param_slug(name) -> String

Filename-safe form of a sweep parameter name: `"yaw[2]"` -> `"yaw_2"`,
`"input_std[3]"` -> `"input_std_3"`. Brackets are legal in a POSIX filename but
are glob metacharacters, so a saved figure called `..._yaw[2]_sensitivity.svg`
is awkward to reach from a shell and from LaTeX's `\\includegraphics`.

Lives here rather than in a script because both parameter families produce names
of this shape and every caller wants the same transformation.
"""
param_slug(name::AbstractString)::String = replace(String(name), "[" => "_", "]" => "")
