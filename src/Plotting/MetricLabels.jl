"""
Shared metric vocabulary for the results figures.

Before this existed, six plotting functions each carried their own copy of
`metric in (:rmse, :rmse_rate) || throw(...)` and their own
`metric == :rmse ? "RMSE" : "RMSE rate"` ternary. Adding a metric meant editing
six files, and the axis labels carried no units, so an RMSE in metres and a
dimensionless RMSE-rate were labelled in the same style and could be misread
for one another across figures.
"""

const _METRIC_LABELS = Dict{Symbol,String}(
    :rmse => "RMSE [m]",
    :rmse_rate => "RMSE rate [m/m]",
    :rmse_yaw => "Yaw RMSE [rad]",
)

const _METRIC_TITLES = Dict{Symbol,String}(
    :rmse => "Horizontal position RMSE",
    :rmse_rate => "Horizontal position RMSE per distance travelled",
    :rmse_yaw => "Yaw RMSE",
)

"""
    check_metric(metric::Symbol) -> Symbol

Throw a helpful `ArgumentError` unless `metric` is one this package knows how to
plot. Returns the metric so it can be used inline.
"""
function check_metric(metric::Symbol)::Symbol
    haskey(_METRIC_LABELS, metric) && return metric
    throw(ArgumentError(
        "metric must be one of $(join(sort(collect(keys(_METRIC_LABELS))), ", ")), got :$metric"))
end

"""
    metric_label(metric::Symbol) -> String

Axis label for `metric`, including units.
"""
metric_label(metric::Symbol)::String = _METRIC_LABELS[check_metric(metric)]

"""
    metric_title(metric::Symbol) -> String

Human-readable description of `metric`, for figure titles.
"""
metric_title(metric::Symbol)::String = _METRIC_TITLES[check_metric(metric)]
