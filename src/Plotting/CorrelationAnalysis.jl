"""
    compute_correlation_matrix(X::Matrix{Float64}, Y::Matrix{Float64}) -> Matrix{Float64}

Compute Pearson correlation matrix between columns of X (input features) and rows of Y (outputs).
- `X`: input features, size (n_features, n_samples)
- `Y`: output corrections, size (n_outputs, n_samples)

Returns matrix of size (n_features, n_outputs) where element (i,j) is cor(X[i,:], Y[j,:]).
"""
function compute_correlation_matrix(X::Matrix{Float64}, Y::Matrix{Float64})::Matrix{Float64}
    n_in, n_samples = size(X)
    n_out = size(Y, 1)
    @assert size(Y, 2) == n_samples "X and Y must have the same number of samples"

    corr_mat = zeros(n_in, n_out)
    for i in 1:n_in
        for j in 1:n_out
            corr_mat[i, j] = cor(X[i, :], Y[j, :])
        end
    end
    return corr_mat
end


"""
    perform_cca(X::Matrix{Float64}, Y::Matrix{Float64}) -> Tuple{Vector{Float64}, Matrix{Float64}, Matrix{Float64}}

Perform Canonical Correlation Analysis between input features X and output corrections Y.
- `X`: input features, size (n_features, n_samples)
- `Y`: output corrections, size (n_outputs, n_samples)

Returns:
- `canonical_corrs`: Vector of canonical correlations (length = min(n_features, n_outputs))
- `x_proj`: Projection matrix for X (size n_features × n_components)
- `y_proj`: Projection matrix for Y (size n_outputs × n_components)
"""
function perform_cca(X::Matrix{Float64}, Y::Matrix{Float64})
    # Ensure column-major format: each column is a sample
    @assert size(X, 2) == size(Y, 2) "X and Y must have same number of samples"

    # Fit CCA model
    model = MultivariateStats.fit(MultivariateStats.CCA, X, Y)

    # Extract canonical correlations
    canonical_corrs = MultivariateStats.cor(model)

    # Get projection matrices
    x_proj = MultivariateStats.projection(model, :x)  # projects X to canonical space
    y_proj = MultivariateStats.projection(model, :y)  # projects Y to canonical space

    return canonical_corrs, x_proj, y_proj
end



"""
    run_correlation_analysis(input_io::CorrectionIO, output_io::CorrectionIO;
                             feature_type::Union{FeatureType,Nothing}=nothing)

Compute and display the correlation heatmap between each input feature and each output correction,
using data already packaged in `CorrectionIO` structures.

# Arguments
- `input_io`: correction input features (`data` matrix of size n_features × n_steps)
- `output_io`: correction outputs (`data` matrix of size 4 × n_steps)
- `feature_type`: optional `FeatureType` to label input axes; if `nothing` generic names are used.
"""
function run_correlation_analysis(
    input_io::CorrectionIO,
    output_io::CorrectionIO;
    feature_type::Union{FeatureType,Nothing}=nothing,
    output_labels::Vector{String}=["Δx", "Δy", "Δz", "Δψ"]
)
    # Extract data matrices
    input_feature = input_io.data
    output_matrix = output_io.data

    # Determine input labels
    n_features = size(input_feature, 1)
    feature_labels = Dict{Any,Vector{String}}(
        HybridZuptInsJl.THREED_STEP => ["step_x", "step_y", "step_z"],
        HybridZuptInsJl.TWOD_STEP_DT => ["step_x", "step_y", "dt"],
        HybridZuptInsJl.THREED_STEP_DT => ["step_x", "step_y", "step_z", "dt"],
        HybridZuptInsJl.TWOD_STEP_YAW => ["step_x", "step_y", "yaw"],
        HybridZuptInsJl.TWOD_STEP_DT_YAW => ["step_x", "step_y", "dt", "yaw"],
        HybridZuptInsJl.THREED_STEP_DT_YAW => ["step_x", "step_y", "step_z", "dt", "yaw"],
    )

    n_features = size(input_feature, 1)
    input_labels = if feature_type === nothing
        ["Feature$i" for i in 1:n_features]
    else
        get(feature_labels, feature_type, ["Feature$i" for i in 1:n_features])
    end

    # Correlation matrix
    corr_mat = compute_correlation_matrix(input_feature, output_matrix)

    fig = plot_correlation_heatmap(corr_mat, input_labels, output_labels;
        title="Correlations ($(feature_type === nothing ? "generic" : feature_type))")

    println("\nCorrelation matrix ($(feature_type === nothing ? "generic" : feature_type)):")
    println("Inputs: ", join(input_labels, ", "))
    println("Outputs: ", join(output_labels, ", "))
    display(round.(corr_mat, digits=3))

    # CCA
    n_in, n_out = size(input_feature, 1), size(output_matrix, 1)
    n_components = min(n_in, n_out)

    canonical_corrs, x_proj, y_proj = perform_cca(input_feature, output_matrix)

    println("\nCCA Results:")
    println("Canonical correlations: ", round.(canonical_corrs[1:n_components], digits=4))
    println("Total canonical correlation: ",
        round(sqrt(mean(canonical_corrs[1:n_components] .^ 2)), digits=4))

    fig_corr = plot_canonical_correlations(canonical_corrs, n_components)

    return fig, canonical_corrs, fig_corr
end

"""
    plot_correlation_heatmap(corr_mat::Matrix{Float64},
                             input_labels::Vector{String},
                             output_labels::Vector{String};
                             title::String = "Input-Output Correlation Heatmap")

Plot a heatmap of the correlation matrix with colorbar and annotated values.
"""
function plot_correlation_heatmap(corr_mat::Matrix{Float64},
    input_labels::Vector{String},
    output_labels::Vector{String};
    title::String="Input-Output Correlation Heatmap")

    n_in, n_out = size(corr_mat)   # rows: input features, cols: outputs
    @assert length(input_labels) == n_in
    @assert length(output_labels) == n_out

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xaxisposition=:top,            # input labels at the top
        yaxisposition=:left,           # output labels at the left
        xticks=(1:n_in, input_labels),
        yticks=(1:n_out, output_labels),
        xticklabelrotation=0,          # no rotation
        title=title,
        xlabel="Input features",
        ylabel="Output corrections",
        xgridvisible=false,
        ygridvisible=false)

    # z matrix: rows = outputs (y-axis), columns = inputs (x-axis)
    hm = heatmap!(ax, 1:n_in, 1:n_out, (corr_mat) .^ 2)

    # Add text annotations
    for i in 1:n_in, j in 1:n_out
        val = corr_mat[i, j]^2
        if val >= 0.1
            text!(ax, i, j, text="$(round(val, digits=2))";
                color=:black,
                align=(:center, :center))
        end
    end
    # Horizontal colorbar below the heatmap
    Colorbar(fig[2, 1], hm, label="R²", vertical=false)

    return fig
end

"""
    plot_canonical_correlations(canonical_corrs::Vector{Float64}, n_components::Int; title::String = "Canonical Correlations")

Plot the canonical correlations as a bar chart.
"""
function plot_canonical_correlations(canonical_corrs::Vector{Float64}, n_components::Int; title::String="Canonical Correlations")
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="Canonical Variate", ylabel="Canonical Correlation", title=title)
    bar_positions = 1:n_components
    bars = barplot!(ax, bar_positions, canonical_corrs[1:n_components], color=:steelblue)
    # Add value labels on top of bars
    for (i, val) in enumerate(canonical_corrs[1:n_components])
        text!(ax, i, val + 0.02, text="$(round(val, digits=2))", align=(:center, :bottom))
    end
    ylims!(ax, 0, 1)
    return fig
end