function plot_hp_sensitivity(
    df::DataFrame, grid::ParamGrid, log_range::Tuple{Float64,Float64};
    save_path::Union{String,Nothing}=nothing)

    n_rows, n_cols = size(grid.specs)
    fig = Figure(size=(400 * n_cols, 400 * n_rows))

    # Get baseline RMSE
    baseline_row = df[df.parameter.=="baseline", :]
    baseline_rmse = isempty(baseline_row) ? 1.0 : first(baseline_row.rmse)

    col_colors = Makie.wong_colors()
    for row in 1:n_rows
        for col in 1:n_cols
            spec = grid.specs[row, col]
            if isnothing(spec)
                # Skip empty cell – leave blank
                continue
            end
            # Find data for this parameter
            sub = df[df.parameter.==spec.name, :]
            if isempty(sub)
                @warn "Missing data for $(spec.name)"
                continue
            end
            ax = Axis(fig[row, col];
                xlabel="log₁₀(relative change)",
                ylabel=row == n_rows ? "RMSE ratio" : "",
                title=spec.name,
                xgridstyle=:dash,
                ygridstyle=:dash)
            base_val = first(sub.base_value)
            rel_change = log10.(sub.tested_value ./ base_val)
            lines!(ax, rel_change, sub.rmse_ratio;
                color=col_colors[col], linewidth=2)
            scatter!(ax, rel_change, sub.rmse_ratio;
                color=col_colors[col], markersize=6)
            hlines!(ax, 1.0, color=:gray, linestyle=:dash, linewidth=1)
            xlims!(ax, log_range...)
        end
    end

    # Add global y-label
    Label(fig[1:n_rows, 0], "RMSE ratio", rotation=π / 2, fontsize=14)

    if !isnothing(save_path)
        save(save_path, fig)
        @info "Saved figure: $save_path"
    end
    display(fig)
    return nothing
end