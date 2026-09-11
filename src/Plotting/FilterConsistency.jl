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
Colours for the filter configurations compared in `scripts/5Results/6_single_filter.jl`.

Tol vibrant, deliberately *not* `Makie.wong_colors()`: Wong is what the rest of the
suite uses -- correction methods in `scripts/5Results/`, hyperparameter families in
`OnlineHpSensitivity.jl` -- and a filter configuration is neither of those, so
sharing the palette would imply a correspondence that does not exist. (Note that
`ColorSchemes.okabe_ito` is not an alternative here: it is the same palette as
Wong's, permuted.) Entry 1 is black for the uncorrected baseline, which is drawn
first and sits at the back.

Entries 2-4 are Tol vibrant's canonical three -- magenta, blue, orange -- in that
order, so that the most separated pair, blue and orange, falls on the third and
fourth run drawn. In §6 those are `cov_update=false` and the fixed-ZUPT-gain
counterfactual, which coincide *by design*: they are the pair a reader has to be
able to separate, and the earlier green-against-blue assignment was exactly the
pair that fails at it, on a light panel and worse under deuteranopia. No green in
the palette at all for that reason.

Used positionally only as a fallback; pass `colors` (name => colour) to keep one
configuration the same colour across every panel and every figure, which
positional assignment cannot do when the panels hold different subsets of runs.
"""
const FILTER_CONFIG_COLORS = ["#000000", "#EE3377", "#0077BB", "#EE7733",
    "#009988", "#33BBEE", "#CC3311"]

_run_color(colors, key, i) = begin
    fallback = FILTER_CONFIG_COLORS[mod1(i, length(FILTER_CONFIG_COLORS))]
    isnothing(colors) ? fallback : get(colors, key, fallback)
end

"""
    plot_nees_comparison(runs; block, yscale, split_k, colors, dashed, title, save_path)

Overlays NEES-over-time for multiple filter configurations against the
chi-square 95% envelope, using the NamedTuples returned by `nees_series`
(fields `k`, `pos`, `vel`, `att`, `lower`, `upper`, `dof`). One line per entry
of `runs` (e.g. `cov_update=true` / `cov_update=false` / baseline), `block`
selects which field (`:pos`, `:vel`, or `:att`) to plot. The lower/upper
bound is drawn once from the first run (identical dof=3 chi-square bounds for
every run, per `nees_series`'s docstring) — consistent filters keep their
line inside the band ~95% of the time.

`split_k` draws the train/test divider, and `yscale=log10` is what makes a
full-run series readable: the train half sits at NEES ~1 while an inconsistent
test half reaches ~1e3, so on a linear axis the earned shrink is flattened onto
the x-axis by the unearned one. NEES is a Mahalanobis quadratic form, hence
strictly positive, so the log axis is safe.

`colors` maps a run name to its colour ([`FILTER_CONFIG_COLORS`](@ref) positionally
otherwise) and `dashed` lists the runs drawn dashed. Legend entries are the run
names alone: the per-run statistics belong in the table the caller prints, and a
single consistency percentage over a run whose two halves differ by 70 points is
not a number worth putting on a figure.
"""
function plot_nees_comparison(
    runs::AbstractDict{String,<:NamedTuple};
    block::Symbol=:pos,
    yscale=identity,
    split_k::Union{Nothing,Int}=nothing,
    colors::Union{Nothing,AbstractDict}=nothing,
    dashed::AbstractVector{<:AbstractString}=String[],
    title::String="NEES consistency ($block)",
    save_path::Union{String,Nothing}=nothing
)
    fig = Figure(size=(900, 450))
    ax = Axis(fig[1, 1]; xlabel="Sample index k", ylabel="NEES",
        title=title, xgridvisible=false, yscale=yscale)

    first_run = first(values(runs))
    ks = first_run.k
    band!(ax, ks, fill(first_run.lower, length(ks)), fill(first_run.upper, length(ks));
        color=(:gray, 0.2), label="95% envelope")

    for (si, (key, r)) in enumerate(runs)
        vals = getfield(r, block)
        isnothing(vals) && error("Run \"$key\" has no `$block` NEES (was include_vel set?).")
        vals = yscale === log10 ? max.(vals, eps()) : vals
        dash = key in dashed
        lines!(ax, r.k, vals; color=_run_color(colors, key, si), label=key,
            linestyle=dash ? :dash : :solid, linewidth=dash ? 1.4 : 1.4)
    end
    isnothing(split_k) || vlines!(ax, [split_k]; color=:black, linestyle=:dash,
        linewidth=1.5, label="train | test")
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
    plot_zupt_starvation(runs; poserr, smooth, split_k, colors, dashed, title, save_path)

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
Name it in `dashed` so it is drawn dashed and heavier, and visibly on top of the
curve it is supposed to land on.

`colors` maps a run name to its colour, [`FILTER_CONFIG_COLORS`](@ref)
positionally otherwise. Pass it whenever the panels hold different subsets of the
runs -- (a)/(b) typically compare two configurations while (c) shows four -- since
positional assignment would then paint the same configuration differently in
different panels of one figure.

Both top panels plot a centred moving average of width `smooth` on a log scale.
The raw per-epoch values span decades within a single stance phase (`P` is cut
at every ZUPT and regrows through the swing), so the unsmoothed trace is a solid
band that hides the between-configuration difference and bloats vector output.
Legend entries are the run names alone; the per-run means belong in the table the
caller prints.

Note this deliberately does *not* plot cumulative delivered correction
`sum‖Δp‖`: that sums magnitudes irrespective of direction, and does not separate
the configurations.

`split_k` marks the train/test boundary and extends the reading to the train
half, where the mocap update — not the GP — is what shrinks `P`. That comparison
answers the question the test half cannot: the mocap update is also an absolute
4-dof update on `P`, so if it starves the gain just as hard while costing
nothing, the damage is not the shrink itself but the absence of anything to
replace the channel it closes. Note the configurations are *identical* left of
`split_k` (`cov_update` gates only the GP branch), so the curves coincide there
by construction. Each phase is smoothed independently: a centred moving average
run across the boundary would smear the step at `split_k`, which is the payload.
"""
function plot_zupt_starvation(
    runs::AbstractDict{String,<:NamedTuple};
    poserr::Union{Nothing,AbstractDict}=nothing,
    smooth::Int=151,
    split_k::Union{Nothing,Int}=nothing,
    colors::Union{Nothing,AbstractDict}=nothing,
    dashed::AbstractVector{<:AbstractString}=String[],
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

    # Index ranges to smooth and draw separately, so no moving average and no
    # line segment crosses the train/test boundary.
    phases(ks) = isnothing(split_k) ? [collect(eachindex(ks))] :
                 filter(!isempty, [findall(<(split_k), ks), findall(>=(split_k), ks)])
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
        col = _run_color(colors, key, si)
        dash = key in dashed
        for (ph, s) in enumerate(phases(r.k))
            sa = pos(movmean(view(r.P_pos, s), smooth))
            sb = pos(movmean(view(r.K_pos, s), smooth))
            push!(smoothed_a, sa)
            push!(smoothed_b, sb)
            # Label once per run, on its first phase only: a second labelled
            # segment would duplicate the run in the legend.
            lab = ph == 1 ? (; label=key) : (;)
            style = (; color=col, linestyle=dash ? :dash : :solid,
                linewidth=dash ? 2.0 : 2.0)
            lines!(axa, view(r.k, s), sa; style..., lab...)
            lines!(axb, view(r.k, s), sb; style..., lab...)
        end
    end

    # Bottom-left legends would otherwise sit on the lower trace: open up a
    # decade of headroom below the data on these log axes.
    for (ax, series) in ((axa, smoothed_a), (axb, smoothed_b))
        isempty(series) && continue
        lo = minimum(minimum, series)
        hi = maximum(maximum, series)
        ylims!(ax, lo / 10, hi * 2)
    end
    for ax in (axa, axb)
        isnothing(split_k) || vlines!(ax, [split_k]; color=:black,
            linestyle=:dash, linewidth=1.5, label="train | test")
    end
    axislegend(axa; position=:lb, framevisible=false)
    axislegend(axb; position=:lb, framevisible=false)

    if !isnothing(poserr)
        # Over a full run the mocap-anchored train half sits ~100x below the
        # test half, so a linear axis shows only the test half.
        axc = Axis(fig[2, 1:2]; xlabel="Sample index k",
            ylabel="‖position error‖ [m]",
            title=isnothing(split_k) ?
                  "Restoring only the ZUPT gain recovers the performance" :
                  "Position error: mocap-anchored train half, then GP-corrected test half",
            yscale=identity, # isnothing(split_k) ? identity : 
            xgridvisible=false)
        for (si, (key, pe)) in enumerate(poserr)
            dash = key in dashed
            lines!(axc, pe[1], isnothing(split_k) ? pe[2] : pos(pe[2]);
                color=_run_color(colors, key, si), label=key,
                linestyle=dash ? :dash : :solid, linewidth=dash ? 1.4 : 1.4)
        end
        isnothing(split_k) || vlines!(axc, [split_k]; color=:black,
            linestyle=:dash, linewidth=1.5, label="train | test")
        axislegend(axc; position=:lt, framevisible=false)
    end

    isnothing(save_path) || save(save_path, fig)
    return fig
end
