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
        lines!(ax, r.k, vals; label="$key, consistency=$(round(Int,100*consistency_ratio(vals, r.lower, r.upper)))%")
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

"""
    plot_zupt_starvation(runs; poserr, smooth, title, save_path)

Shows *why* a GP covariance update costs position accuracy: it starves the ZUPT.

Position is never directly observed in a ZUPT-aided INS. The only channel that
walks back the error accumulated during the swing phase is the
position<->velocity cross-covariance carried in `P`, through the position rows
of the ZUPT gain `K[1:3,:] = P[1:3,4:6] * S^-1`. A GP measurement update applied
to the absolute `P` collapses `P[1:3,1:3]`, and that gain collapses with it.

`runs` maps a configuration name to the NamedTuple from `zupt_gain_series`:

  (a) `tr(P[1:3,1:3])` at ZUPT epochs -- the collapse (the cause).
  (b) `‖K[1:3,:]‖` -- the throttled channel (the mechanism).

`poserr` maps `name => (k, err)` for the bottom panel, which carries the causal
claim: include a counterfactual run with the collapsed `P` left in place
*everywhere except* the ZUPT gain (`p_split=:downstream_only,
zupt_gain_source=:P_alt`). If that curve lands on the `cov_update=false` curve,
the ZUPT gain is the whole mechanism and (b) is causal rather than correlated.
Names containing "counterfactual" are drawn dashed and heavier.

Both top panels plot a centred moving average of width `smooth` on a log scale.
The raw per-epoch values span decades within a single stance phase (`P` is cut
at every ZUPT and regrows through the swing), so the unsmoothed trace is a solid
band that hides the between-configuration difference and bloats vector output;
the exact per-run means are in the legend instead.

Note this deliberately does *not* plot cumulative delivered correction
`sum‖Δp‖`: that sums magnitudes irrespective of direction, and does not separate
the configurations.
"""
function plot_zupt_starvation(
    runs::AbstractDict{String,<:NamedTuple};
    poserr::Union{Nothing,AbstractDict}=nothing,
    smooth::Int=151,
    title::String="GP covariance update starves the ZUPT position correction",
    save_path::Union{String,Nothing}=nothing
)
    movmean(v, w) = begin
        w = min(w, length(v))
        w < 2 && return collect(v)
        h = w ÷ 2
        [mean(@view v[max(1, i-h):min(length(v), i+h)]) for i in eachindex(v)]
    end
    pos(v) = max.(v, eps())

    fig = Figure(size=(1150, 820))
    Label(fig[0, 1:2], title; fontsize=17, font=:bold)

    axa = Axis(fig[1, 1]; xlabel="Sample index k", ylabel="tr(P[1:3,1:3])  [m²]",
        title="Position uncertainty at ZUPT time steps",
        yscale=log10, xgridvisible=false)
    axb = Axis(fig[1, 2]; xlabel="Sample index k", ylabel="‖K[1:3,:]‖",
        title="Position gain magnitude at ZUPT time steps",
        yscale=log10, xgridvisible=false)

    smoothed_a = Vector{Float64}[]
    smoothed_b = Vector{Float64}[]
    for (si, (key, r)) in enumerate(runs)
        col = Makie.wong_colors()[mod1(si, 7)]
        sa, sb = pos(movmean(r.P_pos, smooth)), pos(movmean(r.K_pos, smooth))
        push!(smoothed_a, sa)
        push!(smoothed_b, sb)
        lines!(axa, r.k, sa; color=col, linewidth=2,
            label="$key  (mean $(round(r.mean_P_pos, sigdigits=3)))")
        lines!(axb, r.k, sb; color=col, linewidth=2,
            label="$key  (mean $(round(r.mean_K_pos, sigdigits=3)))")
    end

    # Bottom-left legends would otherwise sit on the lower trace: open up a
    # decade of headroom below the data on these log axes.
    for (ax, series) in ((axa, smoothed_a), (axb, smoothed_b))
        isempty(series) && continue
        lo = minimum(minimum, series)
        hi = maximum(maximum, series)
        ylims!(ax, lo / 10, hi * 2)
    end
    axislegend(axa; position=:lb, framevisible=false)
    axislegend(axb; position=:lb, framevisible=false)

    if !isnothing(poserr)
        axc = Axis(fig[2, 1:2]; xlabel="Sample index k",
            ylabel="‖position error‖ [m]",
            title="Restoring only the ZUPT gain recovers the performance",
            xgridvisible=false)
        for (si, (key, pe)) in enumerate(poserr)
            cf = occursin("counterfactual", lowercase(key))
            lines!(axc, pe[1], pe[2]; color=Cycled(si), label=key,
                linestyle=cf ? :dash : :solid, linewidth=cf ? 3.0 : 1.4)
        end
        axislegend(axc; position=:lt, framevisible=false)
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end
