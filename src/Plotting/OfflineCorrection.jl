"""
    plot_line_with_std!(ax, t, data, data_std; color, label=nothing, linewidth=1.4, kwargs...)

Add a line plot to `ax` with two uncertainty bands (1σ and 2σ) drawn from `data_std`.
If `data_std` is `nothing`, only the line is plotted.

Only the line carries the `label` – the bands are silent in the legend.
"""
function plot_line_with_std!(ax, t, data, data_std; color, label=nothing, linewidth=1.4, kwargs...)
    if !isnothing(data_std)
        # 2σ band (lighter)
        # band!(ax, t, data .- 2data_std, data .+ 2data_std; color=(color, 0.1), label=label)
        # 1σ band
        band!(ax, t, data .- data_std, data .+ data_std; color=(color, 0.15), label=label)
    end
    lines!(ax, t, data; color=color, linewidth=linewidth, label=label, kwargs...)
end

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
    ref = isnothing(true_data) ? first(values(pred_data)) : true_data
    t = ref.t
    n_dim = size(ref.data, 1)
    @assert 1 <= n_dim <= 4

    n_cols = min(2, n_dim)
    n_rows = ceil(Int, n_dim / n_cols)
    fig = Figure(size=(1200, 900))
    axes = Vector{Axis}(undef, n_dim)

    for idx in 1:n_dim
        row = (idx - 1) ÷ n_cols + 1
        col = (idx - 1) % n_cols + 1
        ax = Axis(fig[row, col])

        if idx <= 3
            ax.ylabel = "meters"
        elseif idx == 4
            ax.ylabel = "radians"
        end
        if row == n_rows
            ax.xlabel = "Time (s)"
        end
        axes[idx] = ax
    end

    # Ground truth
    if !isnothing(true_data)
        for (plot_idx, row) in enumerate(1:n_dim)
            std_vals = isnothing(true_data.data_std) ? nothing : true_data.data_std[row, :]
            plot_line_with_std!(axes[plot_idx], true_data.t, true_data.data[row, :],
                std_vals;
                color=:black, linewidth=0.9, label="Target")
        end
    end

    # Predictions
    if !isnothing(pred_data)
        colors = [:red, :blue, :green, :orange, :purple]

        for (method_idx, (method_name, pred)) in enumerate(pred_data)
            color = colors[(method_idx-1)%length(colors)+1]

            # RMSE on overlapping interval
            rmse = fill(NaN, 4)
            if !isnothing(true_data)
                res = truncate_to_overlap(true_data, pred)
                if !isnothing(res)
                    gt_trim, pred_trim = res
                    is_compatible(gt_trim, pred_trim) ||
                        throw(ArgumentError(
                            "Series '$method_name' is not compatible with ground truth " *
                            "(different lengths or timestamps differ > 1e-9). " *
                            "Truncate first or resample."))
                    for (plot_idx, row) in enumerate(1:n_dim)
                        rmse[plot_idx] = sqrt(mean(
                            (pred_trim.data[row, :] .- gt_trim.data[row, :]) .^ 2))
                    end
                end
            end

            for (plot_idx, row) in enumerate(1:n_dim)
                rmse_str = isnan(rmse[plot_idx]) ? "" :
                           " (RMSE=$(round(rmse[plot_idx]; digits=4)))"
                std_vals = isnothing(pred.data_std) ? nothing : pred.data_std[row, :]
                std_str = isnothing(std_vals) ? "" : " (Mean Std=$(round(mean(std_vals);digits=3)))"
                label = "$method_name$rmse_str$std_str"
                plot_line_with_std!(axes[plot_idx], pred.t, pred.data[row, :],
                    std_vals;
                    color=color, label=label)
            end
        end
    end

    for ax in axes
        axislegend(ax; position=:rt, merge=true)
        ax.xgridvisible = true
        ax.ygridvisible = true
    end
    # Legend(fig[n_rows+1, :], [ax for ax in axes]; orientation=:horizontal, merge=true)

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

    palette = [:red, :blue, :green, :orange, :purple]
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1]; xlabel="Time (s)", ylabel="Value", title=title,
        xgridvisible=true, ygridvisible=true)

    # Plot every channel present. Previously hard-coded to 1:3, which silently
    # dropped channels 4+ for TWOD_STEP_YAW / THREED_STEP_DT_YAW features.
    for i in 1:n_channel
        color = palette[mod1(i, length(palette))]
        lines!(ax, t, features[i, :]; color=color, linewidth=1.2, label=labels[i])

        if !isnothing(feature_std)
            label = "Feature $i ±σ"
            band!(ax, t, features[i, :] .- feature_std[i, :], features[i, :] .+ feature_std[i, :];
                color=(color, 0.15), label=label)

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