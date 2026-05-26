"""
    plot_regression_results(
        pred_data::Union{Nothing,AbstractDict{String,Union{CorrectionIO, Dict}}},
        true_data::Union{CorrectionIO, Matrix{Float64}})

Plot regression results for yaw and position components (X, Y, Z) with RMSE annotations.
Supports both CorrectionIO objects and traditional dictionary/matrix formats.

# Arguments
- `pred_data`: Dictionary mapping method names to CorrectionIO
- `true_data`: CorrectionIO

# Returns
- `Figure` object.
"""
function plot_regression_results(
    pred_data::Union{Nothing,AbstractDict{String,CorrectionIO}},
    true_data::Union{Nothing,CorrectionIO}=nothing
)
    # ── Time axis: prefer true_data, else first pred ──────────────────
    ref = isnothing(true_data) ? first(values(pred_data)) : true_data
    t = ref.t
    N = length(t)

    titles = ["X correction (m)", "Y correction (m)", "Z correction (m)", "Yaw correction (rad)"]
    row_idx = [1, 2, 3, 4]

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
            lines!(axes[plot_idx], true_data.t, true_data.data[row, :];
                color=:black, linewidth=0.9, label="True")

            if !isnothing(true_data.data_std)
                μ = true_data.data[row, :]
                σ = true_data.data_std[row, :]
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
                        (pred_trim.data[row, :] .- gt_trim.data[row, :]) .^ 2))
                end
            end

            for (plot_idx, row) in enumerate(row_idx)
                rmse_str = isnan(rmse[plot_idx]) ? "" :
                           " (RMSE=$(round(rmse[plot_idx]; digits=4)))"
                label = "$method_name$rmse_str"

                lines!(axes[plot_idx], pred.t, pred.data[row, :];
                    color=color,
                    linewidth=1.4,
                    label=label)

                # ── Confidence band (±1 std) ──────────────────────────
                if !isnothing(pred.data_std)
                    μ = pred.data[row, :]
                    σ = pred.data_std[row, :]
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
    pred_data::CorrectionIO,
    true_data::Union{Nothing,CorrectionIO}=nothing
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
    feature_std::Union{Nothing,Matrix{Float64}}=nothing,
    labels::Union{Vector{String},Nothing}=nothing,
    title::String="Input Features"
)
    N = size(features, 2)
    n_channel = size(features, 1)
    if t === nothing
        t = 1:N
    end
    if labels === nothing
        labels = ["Feature $i" for i in 1:n_channel]
    end

    colors = [:red, :blue, :green, :orange, :purple]
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1]; xlabel="Time (s)", ylabel="Value", title=title,
        xgridvisible=true, ygridvisible=true)

    for i in 1:3
        lines!(ax, t, features[i, :]; color=colors[i], linewidth=1.2, label=labels[i])

        if !isnothing(feature_std)
            label = "Feature $i ±σ"
            band!(ax, t, features[i, :] .- feature_std[i, :], features[i, :] .+ feature_std[i, :];
                color=(colors[i], 0.15), label=label)

        end
    end
    axislegend(ax; position=:rt)
    return fig
end

function plot_input_features(
    features::CorrectionIO
)
    plot_input_features(features.data, features.t; feature_std=features.data_std)
end