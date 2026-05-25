"""
    plot_regression_results(
        pred_data::Union{Nothing,AbstractDict{String,Union{CorrectionOutput, Dict}}},
        true_data::Union{CorrectionOutput, Matrix{Float64}})

Plot regression results for yaw and position components (X, Y, Z) with RMSE annotations.
Supports both CorrectionOutput objects and traditional dictionary/matrix formats.

# Arguments
- `pred_data`: Dictionary mapping method names to CorrectionOutput or dict with:
    - `"yaw"` → predicted yaw (Vector{Float64}, length N)
    - `"pos"` → predicted positions (Matrix{Float64}, size 3×N)
- `true_data`: CorrectionOutput or dict/matrix with:
    - `"yaw"` → true yaw (Vector{Float64}, length N)
    - `"pos"` → true positions (Matrix{Float64}, size 3×N)
    - `"t"`   → (optional) time vector (Vector{Float64}, length N)

# Returns
- `Figure` object.
"""
function plot_regression_results(
    pred_data::Union{Nothing,AbstractDict{String,CorrectionOutput}},
    true_data::Union{Nothing,CorrectionOutput}=nothing
)
    # ── Time axis: prefer true_data, else first pred ──────────────────
    ref = isnothing(true_data) ? first(values(pred_data)) : true_data
    t = ref.t
    N = length(t)

    titles = ["X correction (m)", "Y correction (m)", "Z correction (m)", "Yaw correction (rad)"]
    row_idx = [1, 2, 3, 4]   # output row indices

    fig = Figure(size=(1200, 900))
    axes = Vector{Axis}(undef, 4)
    for idx in 1:4
        axes[idx] = Axis(fig[(idx-1)÷2+1, (idx-1)%2+1];
            title=titles[idx],
            ylabel=idx <= 3 ? "meters" : "radians",
            xlabel=idx >= 3 ? "Time (s)" : "")
    end

    # ── Ground truth ──────────────────────────────────────────────────
    if !isnothing(true_data)
        for (plot_idx, row) in enumerate(row_idx)
            lines!(axes[plot_idx], true_data.t, true_data.output[row, :];
                color=:black, linewidth=0.9, label="True")

            if !isnothing(true_data.output_std)
                μ = true_data.output[row, :]
                σ = true_data.output_std[row, :]
                label = "True ±σ"
                band!(axes[plot_idx], true_data.t, μ .- σ, μ .+ σ;
                    color=(:black, 0.15), label=label)
            end
        end
    end

    # ── Predictions ───────────────────────────────────────────────────
    if !isnothing(pred_data)
        colors = [:red, :blue, :green, :orange, :purple]

        for (method_idx, (method_name, pred)) in enumerate(pred_data)
            color = colors[(method_idx-1)%length(colors)+1]

            # ── RMSE on overlapping interval ──────────────────────────
            rmse = fill(NaN, 4)
            if !isnothing(true_data)
                gt_trim, pred_trim = truncate_to_overlap(true_data, pred)
                is_compatible(gt_trim, pred_trim) ||
                    throw(ArgumentError(
                        "Series '$method_name' is not compatible with ground truth " *
                        "(different lengths or timestamps differ > 1e-9). " *
                        "Truncate first or resample."))

                for (plot_idx, row) in enumerate(row_idx)
                    rmse[plot_idx] = sqrt(mean(
                        (pred_trim.output[row, :] .- gt_trim.output[row, :]) .^ 2))
                end
            end

            for (plot_idx, row) in enumerate(row_idx)
                rmse_str = isnan(rmse[plot_idx]) ? "" :
                           " (RMSE=$(round(rmse[plot_idx]; digits=4)))"
                label = "$method_name$rmse_str"

                lines!(axes[plot_idx], pred.t, pred.output[row, :];
                    color=color,
                    linewidth=1.4,
                    label=label)

                # ── Confidence band (±1 std) ──────────────────────────
                if !isnothing(pred.output_std)
                    μ = pred.output[row, :]
                    σ = pred.output_std[row, :]
                    label = "$method_name ±σ"
                    band!(axes[plot_idx], pred.t, μ .- σ, μ .+ σ;
                        color=(color, 0.15), label=label)
                end
            end
        end
    end

    # ── Legends / grid ────────────────────────────────────────────────
    for ax in axes
        axislegend(ax; position=:rt, framevisible=true)
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    return fig
end

function plot_regression_results(
    pred_data::CorrectionOutput,
    true_data::Union{Nothing,CorrectionOutput}=nothing
)

    pred_data = Dict(
        "Prediction" => pred_data
    )
    return plot_regression_results(pred_data, true_data)
end
"""
    plot_input_features(
        features::Matrix{Float64}, t::Union{Vector{Float64},Nothing}=nothing;
        labels::Union{Vector{String},Nothing}=nothing,
        title::String="Input Features"
    )

Plot three-dimensional input features over time.

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