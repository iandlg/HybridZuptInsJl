"""
Colours for the filter configurations in `scripts/5Results/6_single_filter.jl`: Tol
vibrant (not Wong, which the suite uses for correction methods), black first for the
baseline, and no green so the blue/orange pair that must be told apart stays separable
under deuteranopia. Positional fallback only; pass `colors` (name => colour) to keep a
configuration's colour fixed across panels.
"""
const FILTER_CONFIG_COLORS = ["#000000", "#EE3377", "#0077BB", "#EE7733",
    "#009988", "#33BBEE", "#CC3311"]

_run_color(colors, key, i) = begin
    fallback = FILTER_CONFIG_COLORS[mod1(i, length(FILTER_CONFIG_COLORS))]
    isnothing(colors) ? fallback : get(colors, key, fallback)
end

"""
    plot_nees_comparison(runs; block, yscale, split_k, colors, dashed, title, save_path)

NEES over time of several runs (`name => nees_series` result) against the 95% χ²
envelope, for `block` `:pos`, `:vel` or `:att`. `split_k` draws the train/test divider;
`yscale=log10` keeps train (NEES ~1) and an inconsistent test half (~1e3) both readable.
`colors` maps names to colours ([`FILTER_CONFIG_COLORS`](@ref) otherwise), `dashed` lists
runs drawn dashed.
"""
function plot_nees_comparison(
    runs::AbstractDict{String,<:NamedTuple};
    block::Symbol=:pos,
    yscale=identity,
    split_k::Union{Nothing,Int}=nothing,
    colors::Union{Nothing,AbstractDict}=nothing,
    dashed::AbstractVector{<:AbstractString}=String[],
    legend_position=:rt,
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
    axislegend(ax; position=legend_position)

    isnothing(save_path) || save(save_path, fig)
    return fig
end

"""
    plot_zupt_starvation(runs; poserr, smooth, split_k, colors, dashed, title, save_path)

Why a GP covariance update costs position accuracy: collapsing `P[1:3,1:3]` throttles
the ZUPT position gain `K[1:3,:] = P[1:3,4:6] S⁻¹`, the only path that corrects position.

`runs` maps names to [`zupt_gain_series`](@ref) results: (a) `tr(P[1:3,1:3])` and (b)
`‖K[1:3,:]‖` at ZUPT epochs, both log-scale and smoothed with a centred moving average of
width `smooth` (per phase, so the step at `split_k` survives). `poserr` maps
`name => (k, err)` for the bottom position-error panel; mark a counterfactual run in
`dashed`. `colors` keeps names consistent across panels.
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

"""
    plot_nees_train_ratio(summary, dataset_name; phase="test", show_points, save_path)

Consistency over the train-ratio sweep, from `nees_summary`: one box per corrector per
train ratio, one point per trial. Columns are position (χ²(3)) and yaw (χ²(1)); the top
row is the ANEES on a log axis against its dof (dashed) and the per-sample 95% envelope
(grey), the bottom row the fraction of footfalls inside that envelope against 0.95.
"""
function plot_nees_train_ratio(
    summary::DataFrame,
    dataset_name::AbstractString;
    phase::AbstractString="test",
    show_points::Bool=true,
    save_path::Union{String,Nothing}=nothing,
)
    sub = summary[(summary.dataset_name .== dataset_name) .& (summary.phase .== phase), :]
    isempty(sub) && error("No $phase rows for dataset_name = $dataset_name")

    fig = Figure(size=(1200, 800))
    Label(fig[0, 1:2], "NEES consistency, $dataset_name, $phase phase"; fontsize=17, font=:bold)
    legend_ax = nothing
    for (col, (block, dof, name)) in enumerate(((:pos, 3, "Position"), (:yaw, 1, "Yaw")))
        lo, hi = quantile(Chisq(dof), 0.025), quantile(Chisq(dof), 0.975)

        ax_n = Axis(fig[1, col]; title="$name (χ²($dof))", ylabel="ANEES", yscale=log10,
            xlabel="Ground truth available online")
        hspan!(ax_n, lo, hi; color=(:gray, 0.2))
        hlines!(ax_n, [dof]; color=:black, linestyle=:dash, linewidth=1)
        col == 1 && (legend_ax = ax_n)
        _grouped_boxplot!(ax_n, sub, Symbol("anees_", block);
            group_col=:train_ratio, group_order_col=:train_ratio_order, show_points=show_points)

        ax_r = Axis(fig[2, col]; ylabel="inside 95% envelope", limits=(nothing, (0, 1.05)),
            xlabel="Ground truth available online")
        hlines!(ax_r, [0.95]; color=:black, linestyle=:dash, linewidth=1)
        _grouped_boxplot!(ax_r, sub, Symbol("inside_", block);
            group_col=:train_ratio, group_order_col=:train_ratio_order, show_points=show_points)

        ratio_order = Dict(r.train_ratio => r.train_ratio_order for r in eachrow(sub))
        ratios = sort(unique(sub.train_ratio), by=r -> ratio_order[r])
        for ax in (ax_n, ax_r)
            ax.xticks = (1:length(ratios), ["$(round(Int, 100r))%" for r in ratios])
        end
    end
    Legend(fig[3, 1:2], legend_ax; orientation=:horizontal, tellwidth=false)

    isnothing(save_path) || save(save_path, fig)
    return fig
end
