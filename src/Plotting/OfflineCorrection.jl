"""
    plot_regression_results(true_data::Dict{String,Any}, pred_data::Dict{String,Dict{String,Any}})

Plot regression results for yaw and position components (X, Y, Z) with RMSE annotations.

# Arguments
- `true_data`: Dictionary with keys:
    - `"yaw"` → true yaw (Vector{Float64}, length N)
    - `"pos"` → true positions (Matrix{Float64}, size 3×N)
- `pred_data`: Dictionary mapping method names (e.g., `"static"`, `"gp"`) to sub‑dicts:
    - `"yaw"` → predicted yaw (Vector{Float64}, length N)
    - `"pos"` → predicted positions (Matrix{Float64}, size 3×N)

# Returns
- `Figure` object.
"""
function plot_regression_results(
    pred_data::Union{Nothing,AbstractDict{String,Dict{String,VecOrMat{Float64}}}},
    true_data::AbstractDict{String,VecOrMat{Float64}})
    # Unpack true data
    yaw_true = true_data["yaw"]
    pos_true = true_data["pos"]
    N = length(yaw_true)
    @assert size(pos_true, 2) == N "pos_true must have N columns"

    # Create figure: 2×2 grid
    fig = Figure(size=(1200, 800))
    titles = ["Yaw", "X position", "Y position", "Z position"]
    axes = Vector{Any}(undef, 4)
    for idx in 1:4
        ax = Axis(fig[(idx-1)÷2+1, (idx-1)%2+1]; title=titles[idx], ylabel="Value")
        axes[idx] = ax
    end
    # Set xlabel only for bottom row
    for idx in 3:4
        axes[idx].xlabel = "Sample index"
    end

    # ---- Plot ground truth (black solid line) ----
    # Yaw
    lines!(axes[1], 1:N, yaw_true; color=:black, linewidth=1.5, label="True")
    # Positions
    for dim in 1:3
        lines!(axes[dim+1], 1:N, pos_true[dim, :]; color=:black, linewidth=1.5, label="True")
    end

    if !isnothing(pred_data)
        # ---- Plot each prediction method ----
        colors = [:red, :blue, :green, :orange, :purple]
        method_names = collect(keys(pred_data))
        for (method_idx, method_name) in enumerate(method_names)
            color = colors[(method_idx-1)%length(colors)+1]
            pred_dict = pred_data[method_name]

            # Yaw predictions
            yaw_pred = pred_dict["yaw"]
            @assert length(yaw_pred) == N
            rmse_yaw = sqrt(mean((yaw_pred .- yaw_true) .^ 2))
            label_yaw = "$method_name (RMSE = $(round(rmse_yaw, digits=4)))"
            lines!(axes[1], 1:N, yaw_pred; color=color, linestyle=:dash, linewidth=1.2, label=label_yaw)

            # Position predictions
            pos_pred = pred_dict["pos"]
            @assert size(pos_pred) == (3, N)
            for dim in 1:3
                rmse_pos = sqrt(mean((pos_pred[dim, :] .- pos_true[dim, :]) .^ 2))
                label_pos = "$method_name (RMSE = $(round(rmse_pos, digits=4)))"
                lines!(axes[dim+1], 1:N, pos_pred[dim, :]; color=color, linestyle=:dash,
                    linewidth=1.2, label=label_pos)
            end
        end
    end

    # Add legends and grids
    for ax in axes
        axislegend(ax; position=:rt, framevisible=true)
    end

    return fig
end

"""
    plot_input_features(
        features::Matrix{Float64}, t::Union{Vector{Float64},Nothing}=nothing;
        labels::Union{Vector{String},Nothing}=nothing,
        title::String="Input Features"
    )

Plot three‑dimensional input features over time.

# Arguments
- `features`: Feature matrix of size `(3, N)`.
- `t`: Optional time vector of length `N`. If `nothing`, uses sample indices.
- `labels`: Optional legend labels for the three dimensions (default: `["x", "y", "z"]`).
- `title`: Plot title.

# Returns
- A `Figure` object.
"""
function plot_input_features(
    features::Matrix{Float64},
    t::Union{Vector{Float64},Nothing}=nothing;
    labels::Union{Vector{String},Nothing}=nothing,
    title::String="Input Features"
)
    @assert size(features, 1) == 3 "features must have 3 rows"
    N = size(features, 2)
    if t === nothing
        t = 1:N
    end
    if labels === nothing
        labels = ["x", "y", "z"]
    end
    @assert length(labels) == 3 "Must provide three labels"

    colors = [:red, :green, :blue]
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1]; xlabel="Time (s)", ylabel="Value", title=title,
        xgridvisible=true, ygridvisible=true)

    for i in 1:3
        lines!(ax, t, features[i, :]; color=colors[i], linewidth=1.2, label=labels[i])
    end
    axislegend(ax; position=:rt)
    return fig
end