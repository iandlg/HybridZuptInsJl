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
    run_correlation_analysis(trial_id::Int, data_key::String,
                             ref_frame::ReferenceFrame, feature_type::FeatureType)

Load the specified trial, compute training input-output pairs, and display the
correlation heatmap between each input feature and each output correction.
"""
function run_correlation_analysis(
    trial_id::Int, data_dir::String,
    ref_frame::ReferenceFrame, feature_type::FeatureType
)

    ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ =
        HybridZuptInsJl.compute_aligned_ins_trajectory(data_dir, trial_id)

    # Compute training IO (outputs and input features)
    true_outputs, input_feature = HybridZuptInsJl.compute_training_io(
        ins_traj_aligned, gt_traj_aligned, segs;
        ref_frame=ref_frame,
        feature_type=feature_type
    )

    # Get output correction matrix (4 × n_steps)
    output_matrix = true_outputs.data

    # Input feature dimensions depend on feature_type
    n_features = size(input_feature, 1)
    input_labels = if feature_type == HybridZuptInsJl.THREED_STEP
        ["step_x", "step_y", "step_z"]
    elseif feature_type == HybridZuptInsJl.TWOD_STEP_DT
        ["step_x", "step_y", "dt"]
    else # THREED_STEP_DT
        ["step_x", "step_y", "step_z", "dt"]
    end

    output_labels = ["Δx", "Δy", "Δz", "Δψ"]

    # Compute correlation matrix
    corr_mat = compute_correlation_matrix(input_feature, output_matrix)

    # Display heatmap
    fig = plot_correlation_heatmap(corr_mat, input_labels, output_labels;
        title="Correlations ($ref_frame, $feature_type)")

    # Optional: print matrix to console
    println("\nCorrelation matrix ($ref_frame, $feature_type):")
    println("Inputs: ", join(input_labels, ", "))
    println("Outputs: ", join(output_labels, ", "))
    display(round.(corr_mat, digits=3))

    # Compute CCA
    n_in, n_out = size(input_feature, 1), size(output_matrix, 1)
    n_components = min(n_in, n_out)

    canonical_corrs, x_proj, y_proj = perform_cca(input_feature, output_matrix)

    # Print CCA results
    println("\nCCA Results:")
    println("Canonical correlations: ", round.(canonical_corrs[1:n_components], digits=4))
    println("Total canonical correlation: ", round(sqrt(mean(canonical_corrs[1:n_components] .^ 2)), digits=4))
    println("x_proj : ")
    display(x_proj)
    println("y_proj : ")
    display(y_proj)

    # println("="^60)
    # println("  CCA INTERPRETATION FOR: $title")
    # println("="^60)

    # # Iterate through each canonical dimension
    # num_dims = length(corrs)
    # for i in 1:num_dims
    #     @printf("\nStandard Dimension %d (Correlation: %.4f)\n", i, corrs[i])
    #     println("-"^45)

    #     # Display Input (X) Weights
    #     println("  Input Weights (X):")
    #     for j in 1:length(x_labels)
    #         @printf("    %-10s : %6.3f\n", x_labels[j], x_proj[j, i])
    #     end

    #     # Display Output (Y) Weights
    #     println("  Output Weights (Y):")
    #     for j in 1:length(y_labels)
    #         @printf("    %-10s : %6.3f\n", y_labels[j], y_proj[j, i])
    #     end
    # end
    # println("="^60 * "\n")

    # Visualizations
    fig_corr = plot_canonical_correlations(canonical_corrs, n_components)
    display(fig_corr)

    return fig, corr_mat, fig_corr
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

    n_in, n_out = size(corr_mat)
    @assert length(input_labels) == n_in
    @assert length(output_labels) == n_out

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xticks=(1:n_out, output_labels),
        yticks=(1:n_in, input_labels),
        xticklabelrotation=π / 4,
        title=title,
        xlabel="Output corrections",
        ylabel="Input features",
        xgridvisible=false,
        ygridvisible=false)

    # Heatmap
    hm = heatmap!(ax, 1:n_out, 1:n_in, corr_mat)

    # Add text annotations
    for i in 1:n_in, j in 1:n_out
        val = corr_mat[i, j]
        text!(ax, j, i, text="$(round(val, digits=2))";
            color=:white,
            align=(:center, :center))
    end

    Colorbar(fig[1, 2], hm, label="Pearson correlation")
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