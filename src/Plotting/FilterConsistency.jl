"""
    plot_noise_state_correlation(results; component_labels, title, save_path)

Bar plot of the per-component Pearson correlation `rho` returned by
`noise_state_correlation(diagnostics)`, one group of bars per entry of
`results` (e.g. `"cov_update=true"` vs `"cov_update=false"`), with a
horizontal `±band` significance envelope drawn via `hlines!`. This is the
most direct visual test of the Kalman independence assumption `E[v·dx']=0`:
bars extending past the band are significant correlation between the GP
pseudo-measurement error and the true state error it is supposed to be
independent of.

`results` values are the NamedTuple `(rho, band, n, significant)` returned by
`noise_state_correlation`.
"""
function plot_noise_state_correlation(
    results::AbstractDict{String,<:NamedTuple};
    component_labels::Vector{String}=["x", "y", "z", "yaw"],
    title::String="Independence assumption: corr(GP error, true state error)",
    save_path::Union{String,Nothing}=nothing
)
    fig = Figure(size=(700, 450))
    ax = Axis(fig[1, 1]; xlabel="State component", ylabel="Correlation ρ",
        title=title, xticks=(1:length(component_labels), component_labels),
        xgridvisible=false)

    n_groups = length(results)
    width = 0.8 / n_groups
    max_band = 0.0
    for (gi, (key, r)) in enumerate(results)
        xs = (1:length(r.rho)) .+ (gi - (n_groups + 1) / 2) * width
        barplot!(ax, xs, r.rho; width=width, label=key)
        max_band = max(max_band, r.band)
    end
    hlines!(ax, [max_band, -max_band]; color=:black, linestyle=:dash,
        label="95% band")
    axislegend(ax; position=:rt)

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_nees_comparison(runs; block, title, save_path)

Overlays NEES-over-time for multiple filter configurations against the
chi-square 95% envelope, using the NamedTuples returned by `nees_series`
(fields `k`, `pos`, `vel`, `att`, `lower`, `upper`, `dof`). One line per entry
of `runs` (e.g. `cov_update=true` / `cov_update=false` / baseline), `block`
selects which field (`:pos`, `:vel`, or `:att`) to plot. The lower/upper
bound is drawn once from the first run (identical dof=3 chi-square bounds for
every run, per `nees_series`'s docstring) — consistent filters keep their
line inside the band ~95% of the time.
"""
function plot_nees_comparison(
    runs::AbstractDict{String,<:NamedTuple};
    block::Symbol=:pos,
    title::String="NEES consistency ($block)",
    save_path::Union{String,Nothing}=nothing
)
    fig = Figure(size=(900, 450))
    ax = Axis(fig[1, 1]; xlabel="Sample index k", ylabel="NEES",
        title=title, xgridvisible=false)

    first_run = first(values(runs))
    ks = first_run.k
    band!(ax, ks, fill(first_run.lower, length(ks)), fill(first_run.upper, length(ks));
        color=(:gray, 0.2), label="95% envelope")

    for (key, r) in runs
        vals = getfield(r, block)
        isnothing(vals) && error("Run \"$key\" has no `$block` NEES (was include_vel set?).")
        lines!(ax, r.k, vals; label="$key, consistency=$(round(Int,100*consistency_ratio(r.pos, r.lower, r.upper)))%")
    end
    axislegend(ax; position=:rt)

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_innovation_whiteness(results; title, save_path)

Stem-style plot of the ACF returned by `whiteness_test` (fields `lags`,
`acf`, `band`, `ljung_box`, `pvalue`, `n`), with a `±band` significance
envelope. Persistent autocorrelation outside the band at lags >= 1 is the
signature of a measurement carrying state error, per `whiteness_test`'s
docstring — one series per `results` entry lets configurations be compared
directly. Each legend label is annotated with the Ljung-Box p-value.
"""
function plot_innovation_whiteness(
    results::AbstractDict{String,<:NamedTuple};
    title::String="Innovation/NIS autocorrelation",
    save_path::Union{String,Nothing}=nothing
)
    fig = Figure(size=(800, 450))
    ax = Axis(fig[1, 1]; xlabel="Lag", ylabel="ACF",
        title=title, xgridvisible=false)

    max_band = 0.0
    n_series = length(results)
    offset_step = 0.15 / max(n_series, 1)
    for (si, (key, r)) in enumerate(results)
        offset = (si - (n_series + 1) / 2) * offset_step
        xs = collect(r.lags) .+ offset
        for (x, y) in zip(xs, r.acf)
            lines!(ax, [x, x], [0.0, y]; color=Cycled(si))
        end
        scatter!(ax, xs, r.acf; color=Cycled(si),
            label="$key (p=$(round(r.pvalue, digits=3)))")
        max_band = max(max_band, r.band)
    end
    hlines!(ax, [max_band, -max_band]; color=:black, linestyle=:dash,
        label="95% band")
    axislegend(ax; position=:rt)

    isnothing(save_path) || save(save_path, fig)
    return fig
end
