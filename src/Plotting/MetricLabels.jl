"""
Shared metric vocabulary for the results figures.

Metrics are stored as *parts* -- a base name and a subscript -- rather than as
finished label strings, because the same metric has to be rendered two ways: as
Makie rich text with a real subscript for figures, and as plain text for
terminal output. Unicode has no subscript ψ (the subscript block covers a dozen
Latin letters and five Greek ones, none of them ψ), so the terminal form cannot
simply be the figure form written out, and the two would drift if stored
separately.

**To rename a metric, edit `_METRIC_SYMBOL_PARTS` and nothing else.**
"""

"""
`(base, subscript)` per metric. Same shape as `_HP_SYMBOL_PARTS`
([`OnlineHpSensitivity.jl`](@ref)), which stores the hyperparameter symbols for
the same reason.

The subscripts name what is being measured, which the bare word "RMSE" did not:
`p` is horizontal position (`rmse`, `src/1Data/Trajectory.jl:238`, differences
`pos[1:2,:]`) and `ψ` is yaw (`rmse_yaw`, `:261`) -- different quantities in
different units that appear in the same chapter.
"""
const _METRIC_SYMBOL_PARTS = Dict{Symbol,Tuple{String,String}}(
    :rmse => ("RMSE", "p"),
    :rmse_rate => ("RMSE", "p"),
    :rmse_yaw => ("RMSE", "ψ"),
)

"""
Words appended after the symbol. `rmse_rate` is the position RMSE per distance
travelled, so it carries the same symbol as `rmse` and is distinguished by the
suffix and its unit rather than by a symbol of its own.
"""
const _METRIC_SUFFIX = Dict{Symbol,String}(
    :rmse => "",
    :rmse_rate => " rate",
    :rmse_yaw => "",
)

const _METRIC_UNITS = Dict{Symbol,String}(
    :rmse => "m",
    :rmse_rate => "m/m",
    :rmse_yaw => "rad",
)

"""
    check_metric(metric::Symbol) -> Symbol

Throw a helpful `ArgumentError` unless `metric` is one this package knows how to
plot. Returns the metric so it can be used inline.
"""
function check_metric(metric::Symbol)::Symbol
    haskey(_METRIC_SYMBOL_PARTS, metric) && return metric
    throw(ArgumentError(
        "metric must be one of $(join(sort(collect(keys(_METRIC_SYMBOL_PARTS))), ", ")), got :$metric"))
end

"""
    metric_symbol(metric::Symbol) -> rich text

The metric's name with its subscript and no unit, e.g. `RMSE_p`, `RMSE_p rate`.

For composing into a longer label, and for axes that plot a *derived* quantity
whose unit is not the metric's own -- a relative change, a ratio -- where
[`metric_label`](@ref) would advertise the wrong unit.
"""
function metric_symbol(metric::Symbol)
    base, sub = _METRIC_SYMBOL_PARTS[check_metric(metric)]
    return rich(base, subscript(sub), _METRIC_SUFFIX[metric])
end

"""
    metric_label(metric::Symbol) -> rich text

Axis label for `metric`, including units: `RMSE_p [m]`.

Returns rich text, not a `String`, so the subscript renders. Every consumer
passes it straight to an `Axis` `ylabel` or `title`, both of which accept rich
text; anything that needs to *interpolate* the name into a longer string should
compose with `rich(...)` and [`metric_symbol`](@ref) instead.
"""
metric_label(metric::Symbol) =
    rich(metric_symbol(metric), " [", _METRIC_UNITS[check_metric(metric)], "]")
