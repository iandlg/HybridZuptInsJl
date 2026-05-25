function plot_beta_evolution(beta_history::Matrix{Float64})
    m, N = size(beta_history)

    log_abs_beta = log10.(abs.(beta_history) .+ eps())
    vmax = maximum(log_abs_beta)
    vmin = minimum(log_abs_beta)

    fig = Figure(size=(1200, 600))
    ax = Axis(fig[1, 1];
        title="Beta vector evolution (log₁₀ |β|)",
        xlabel="Time step",
        ylabel="Beta index")

    hm = heatmap!(ax, 1:N, 1:m, log_abs_beta';
        colormap=:viridis,
        colorrange=(vmin, vmax))

    Colorbar(fig[1, 2], hm;
        label="log₁₀ |β|",
        width=15,
        ticksize=5,
        ticks=(range(vmin, vmax; length=6),
            ["10^$(round(v; digits=1))" for v in range(vmin, vmax; length=6)]))

    return fig
end