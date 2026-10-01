"""
Label and colour vocabulary for the sensitivity figures (`OnlineHpProbe.jl`), plus the
shared multiplier-axis ticks. Parameters are labelled `[symbol]_k`, `k` being the
channel's position among the swept channels (hyperparameters) or the feature dimension
(statistics): `yaw[2]` → `[ℓ_SE]_3`, `input_std[3]` → `[σ_x]_3`.
"""

"""
Every kind a sweep can contain, in legend order: `type => (symbol, subscript, colour)`,
keyed by the sweep's `type` column. Hyperparameters take Wong 4-6 (1-3 belong to the
correction types), the statistics Tol muted.
"""
const _PARAM_KINDS = [
    "noise" => ("σ", "n", Makie.wong_colors()[4]),
    "length_scale" => ("ℓ", "SE", Makie.wong_colors()[5]),
    "signal_variance" => ("σ", "SE", Makie.wong_colors()[6]),
    "input_mean" => ("μ", "z", Makie.RGBAf(Makie.to_color("#332288"))),
    "input_std" => ("σ", "z", Makie.RGBAf(Makie.to_color("#44AA99"))),
    "input_center" => ("c", "z", Makie.RGBAf(Makie.to_color("#999933"))),
]
const _PARAM_KIND_INFO = Dict(_PARAM_KINDS)

"""
Hyperparameter type name -> its index in a channel's parameter vector. The
inverse of `_HYPERPARAM_TYPES`, so the mapping is stated once.
"""
const _HP_TYPE_INDEX = Dict{String,Int}(v => k for (k, v) in _HYPERPARAM_TYPES)

const _HP_FALLBACK_COLOR = Makie.RGBAf(0.45, 0.45, 0.45, 1.0)

"""
    _parse_hp_param(name) -> (prefix, idx) or nothing

Split a parameter name such as `"yaw[2]"` or `"input_mean[3]"` into its prefix
and bracketed index. `nothing` for anything not of that shape (`"baseline"`).
"""
function _parse_hp_param(name::AbstractString)
    mt = match(r"^(.*)\[(\d+)\]$", name)
    isnothing(mt) && return nothing
    return (String(mt.captures[1]), parse(Int, mt.captures[2]))
end

"""
    _param_kind(name) -> (type, key) or nothing

What a parameter name refers to. A statistics name gives `(family, dim)`; a
hyperparameter name gives `(kind, channel)`, since its bracketed index is the
kind (`yaw[2]` is the yaw length scale). `nothing` for anything else.
"""
function _param_kind(name::AbstractString)
    parsed = _parse_hp_param(name)
    isnothing(parsed) && return nothing
    prefix, idx = parsed
    haskey(_PARAM_KIND_INFO, prefix) && !haskey(_HP_TYPE_INDEX, prefix) && return (prefix, idx)
    (prefix in _OUTPUT_NAMES && haskey(_HYPERPARAM_TYPES, idx)) || return nothing
    return (_HYPERPARAM_TYPES[idx], prefix)
end

"""
    swept_channels(df) -> Vector{String}

The output channels whose hyperparameters `df` sweeps, in `_OUTPUT_NAMES` order.
That is the order they were swept in, since `make_rmse_evaluator` requires
ascending `output_channel_idxs`, so a channel's position here is the `k` in its
`[symbol]_k` label.
"""
function swept_channels(df::AbstractDataFrame)::Vector{String}
    chans = Set{String}()
    for name in unique(df.parameter)
        kind = _param_kind(name)
        isnothing(kind) || !(kind[2] isa String) || push!(chans, kind[2])
    end
    return filter(in(chans), _OUTPUT_NAMES)
end

"""
    param_symbol(type) -> rich text

The symbol of a swept kind, e.g. `ℓ_SE` or `σ_x`. Unknown types pass through as
their raw name.
"""
function param_symbol(type::AbstractString)
    info = get(_PARAM_KIND_INFO, String(type), nothing)
    isnothing(info) && return rich(String(type))
    return rich(info[1], subscript(info[2]))
end

"""
    param_label(name, channels) -> rich text

`[symbol]_k` for a sweep parameter: `yaw[2]` -> `[ℓ_SE]_3` when `channels` is
`["pos_1", "pos_2", "yaw"]`, and `input_std[3]` -> `[σ_x]_3`. `channels` is
[`swept_channels`](@ref) of the frame being plotted. Anything else, e.g.
`"baseline"`, keeps its raw name.
"""
function param_label(name::AbstractString, channels::AbstractVector{<:AbstractString})
    kind = _param_kind(name)
    isnothing(kind) && return rich(String(name))
    type, key = kind
    k = key isa Int ? key : findfirst(==(key), channels)
    isnothing(k) && throw(ArgumentError("param_label: channel \"$key\" is not in $channels"))
    sym, sub, _ = _PARAM_KIND_INFO[type]
    return rich("[", sym, subscript(sub), "]", subscript(string(k)))
end

"""
    type_color(type)

Colour of a swept kind, fixed by kind so every figure agrees. Grey for anything
unknown rather than silently borrowing another kind's colour.
"""
type_color(type::AbstractString) =
    haskey(_PARAM_KIND_INFO, type) ? _PARAM_KIND_INFO[type][3] : _HP_FALLBACK_COLOR

"""
    param_color(name)

Colour of a sweep parameter such as `"yaw[2]"` or `"input_std[3]"`, through its kind.
"""
function param_color(name::AbstractString)
    kind = _param_kind(name)
    return isnothing(kind) ? _HP_FALLBACK_COLOR : type_color(kind[1])
end

"""
    ordered_types(types) -> Vector{String}

`types` in legend order (the order of [`_PARAM_KINDS`](@ref)), followed by any
unknown type in its input order, so nothing is dropped from a legend.
"""
function ordered_types(types)
    ts = String.(collect(types))
    known = filter(in(ts), first.(_PARAM_KINDS))
    return vcat(known, filter(!in(known), unique(ts)))
end

"""
    _hp_tick_positions(mults; max_ticks=7) -> Vector{Float64}

Swept multipliers that get a labelled tick: all of them up to `max_ticks`, otherwise
an even stride anchored on ×1, plus the endpoints where they don't collide.
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

A multiplier as the sensitivity axis renders it: `×0.1`, `×1`, `×10`.
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

The sweep's `parameter` string for one hyperparameter, e.g.
`hp_param_name(:yaw, :length_scale) == "yaw[2]"`. `kind` is `:noise`, `:length_scale`
or `:signal_variance`.
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

The sweep's `parameter` string for one normalisation statistic, e.g.
`stat_param_name(:input_std, 2) == "input_std[2]"`. `family` is `:input_mean`,
`:input_std` or `:input_center`.
"""
function stat_param_name(family::Union{Symbol,AbstractString}, dim::Integer)::String
    key = String(family)
    stats = filter(!in(keys(_HP_TYPE_INDEX)), first.(_PARAM_KINDS))
    key in stats || throw(ArgumentError(
        "unknown statistics family \"$(family)\"; have $stats"))
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    return "$key[$dim]"
end

"""
    param_slug(name) -> String

Filename-safe parameter name: `"yaw[2]"` → `"yaw_2"` (brackets are glob characters).
"""
param_slug(name::AbstractString)::String = replace(String(name), "[" => "_", "]" => "")
