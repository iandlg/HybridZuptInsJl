"""
    plot_regression_results(
        pred_data::Union{Nothing,AbstractDict{String,Dict{String,VecOrMat{Float64}}}},
        true_data::AbstractDict{String,VecOrMat{Float64}})

Plot regression results for yaw and position components (X, Y, Z) with RMSE annotations.
The true_data dictionary may contain a key `"t"` (Vector{Float64}) for the time axis;
otherwise the x-axis is sample indices.

# Arguments
- `pred_data`: Dictionary mapping method names (e.g., `"static"`, `"gp"`) to sub‑dicts:
    - `"yaw"` → predicted yaw (Vector{Float64}, length N)
    - `"pos"` → predicted positions (Matrix{Float64}, size 3×N)
- `true_data`: Dictionary with keys:
    - `"yaw"` → true yaw (Vector{Float64}, length N)
    - `"pos"` → true positions (Matrix{Float64}, size 3×N)
    - `"t"`   → (optional) time vector (Vector{Float64}, length N)

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
    @assert size(pos_true, 2) == N

    # Time axis: use true_data["t"] if present, otherwise indices
    t = haskey(true_data, "t") ? true_data["t"] : collect(1:N)

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
        axes[idx].xlabel = "Time (s)"   # if we have time, otherwise "Sample index"
    end
    if !haskey(true_data, "t")
        # if no time, indicate that x-axis is sample index
        for idx in 3:4
            axes[idx].xlabel = "Sample index"
        end
    end

    # ---- Plot ground truth (black solid line) ----
    lines!(axes[1], t, yaw_true; color=:black, linewidth=1.5, label="True")
    for dim in 1:3
        lines!(axes[dim+1], t, pos_true[dim, :]; color=:black, linewidth=1.5, label="True")
    end

    if !isnothing(pred_data)
        colors = [:red, :blue, :green, :orange, :purple]
        method_names = collect(keys(pred_data))
        for (method_idx, method_name) in enumerate(method_names)
            color = colors[(method_idx-1)%length(colors)+1]
            pred_dict = pred_data[method_name]

            # Yaw predictions
            yaw_pred = pred_dict["yaw"]
            rmse_yaw = sqrt(mean((yaw_pred .- yaw_true) .^ 2))
            label_yaw = "$method_name (RMSE = $(round(rmse_yaw, digits=4)))"
            lines!(axes[1], t, yaw_pred; color=color, linestyle=:dash, linewidth=1.2, label=label_yaw)

            # Position predictions
            pos_pred = pred_dict["pos"]
            for dim in 1:3
                rmse_pos = sqrt(mean((pos_pred[dim, :] .- pos_true[dim, :]) .^ 2))
                label_pos = "$method_name (RMSE = $(round(rmse_pos, digits=4)))"
                lines!(axes[dim+1], t, pos_pred[dim, :]; color=color, linestyle=:dash,
                    linewidth=1.2, label=label_pos)
            end
        end
    end

    # Add legends and grids
    for ax in axes
        axislegend(ax; position=:rt, framevisible=true)
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    return fig
end

"""
    plot_regression_results(
        pred_data::Union{Nothing,AbstractDict{String,Dict{String,VecOrMat{Float64}}}},
        true_data::AbstractDict{String,VecOrMat{Float64}})

Plot regression results for yaw and position components (X, Y, Z) with RMSE annotations.
The true_data dictionary may contain a key `"t"` (Vector{Float64}) for the time axis;
otherwise the x-axis is sample indices.

# Arguments
- `pred_data`: Dictionary mapping method names (e.g., `"static"`, `"gp"`) to sub‑dicts:
    - `"yaw"` → predicted yaw (Vector{Float64}, length N)
    - `"pos"` → predicted positions (Matrix{Float64}, size 3×N)
- `true_data`: Dictionary with keys:
    - `"yaw"` → true yaw (Vector{Float64}, length N)
    - `"pos"` → true positions (Matrix{Float64}, size 3×N)
    - `"t"`   → (optional) time vector (Vector{Float64}, length N)

# Returns
- `Figure` object.
"""
function plot_regression_results_no_rmse(
    pred_data::Union{Nothing,AbstractDict{String,Dict{String,VecOrMat{Float64}}}},
    true_data::AbstractDict{String,VecOrMat{Float64}})

    # Unpack true data
    yaw_true = true_data["yaw"]
    pos_true = true_data["pos"]
    N = length(yaw_true)
    @assert size(pos_true, 2) == N

    # Time axis: use true_data["t"] if present, otherwise indices
    t = haskey(true_data, "t") ? true_data["t"] : collect(1:N)

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
        axes[idx].xlabel = "Time (s)"   # if we have time, otherwise "Sample index"
    end
    if !haskey(true_data, "t")
        # if no time, indicate that x-axis is sample index
        for idx in 3:4
            axes[idx].xlabel = "Sample index"
        end
    end

    # ---- Plot ground truth (black solid line) ----
    lines!(axes[1], t, yaw_true; color=:black, linewidth=1.5, label="True")
    for dim in 1:3
        lines!(axes[dim+1], t, pos_true[dim, :]; color=:black, linewidth=1.5, label="True")
    end

    # ---- Plot ground truth uncertainty bands if available ----
    if haskey(true_data, "yaw_std")
        yaw_std_true = true_data["yaw_std"]
        band!(axes[1], t, yaw_true .- 1 .* yaw_std_true, yaw_true .+ 1 .* yaw_std_true;
            color=(:black, 0.1), label="True σ uncertainty")
    end
    if haskey(true_data, "pos_std")
        pos_std_true = true_data["pos_std"]
        for dim in 1:3
            band!(axes[dim+1], t, pos_true[dim, :] .- 1 .* pos_std_true[dim, :], pos_true[dim, :] .+ 1 .* pos_std_true[dim, :];
                color=(:black, 0.1), label="True σ uncertainty")
        end
    end

    if !isnothing(pred_data)
        colors = [:red, :blue, :green, :orange, :purple]
        method_names = collect(keys(pred_data))
        for (method_idx, method_name) in enumerate(method_names)
            color = colors[(method_idx-1)%length(colors)+1]
            pred_dict = pred_data[method_name]

            # Yaw predictions
            yaw_pred = pred_dict["yaw"]
            t = collect(1:length(yaw_pred))
            if haskey(pred_dict, "t")
                t = pred_dict["t"]
            end
            label_yaw = "$method_name"
            lines!(axes[1], t, yaw_pred; color=color, linestyle=:dash, linewidth=1.2, label=label_yaw)

            # Add yaw uncertainty band if available
            if haskey(pred_dict, "yaw_std")
                yaw_std = pred_dict["yaw_std"]
                band!(axes[1], t, yaw_pred .- 1 .* yaw_std, yaw_pred .+ 1 .* yaw_std;
                    color=(color, 0.2), label="σ uncertainty")
            end

            # Position predictions
            pos_pred = pred_dict["pos"]
            for dim in 1:3
                label_pos = "$method_name"
                lines!(axes[dim+1], t, pos_pred[dim, :]; color=color, linestyle=:dash,
                    linewidth=1.2, label=label_pos)

                # Add position uncertainty band if available
                if haskey(pred_dict, "pos_std")
                    pos_std = pred_dict["pos_std"]
                    band!(axes[dim+1], t, pos_pred[dim, :] .- 1 .* pos_std[dim, :], pos_pred[dim, :] .+ 1 .* pos_std[dim, :];
                        color=(color, 0.2), label="σ uncertainty")
                end
            end
        end
    end

    # Add legends and grids
    for ax in axes
        axislegend(ax; position=:rt, framevisible=true)
        ax.xgridvisible = true
        ax.ygridvisible = true
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