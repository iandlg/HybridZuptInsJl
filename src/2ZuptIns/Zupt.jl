function detect_zupt(u::AbstractMatrix{T}, simdata::InsConfig) where T<:Real
    N = size(u, 2)
    W = simdata.window_size
    gamma = simdata.gamma
    g = simdata.g
    sigma_a = simdata.sigma_a
    sigma_g = simdata.sigma_g

    test_stats = glrt(u, W, g, sigma_a, sigma_g)   # Vector{Float64}

    # Create a boolean array (initialised false)
    zupt = zeros(Bool, N)        # Vector{Bool}
    display(zupt)

    for k in eachindex(test_stats)
        if test_stats[k] < gamma
            zupt[k:k+W-1] .= true   # assign true to the whole window
        end
    end

    # Padding for log-likelihood (unchanged)
    pad = floor(Int, W / 2)
    pad_left = max(1, pad)
    pad_right = max(1, pad)
    max_val = maximum(test_stats)
    T_padded = vcat(fill(max_val, pad_left), test_stats, fill(max_val, pad_right))

    logL = -(W / 2) .* T_padded

    if length(logL) > N
        logL = logL[1:N]
    elseif length(logL) < N
        logL = vcat(logL, fill(logL[end], N - length(logL)))
    end
    zupt = Bool.(zupt)   #
    return zupt, logL
end

function glrt(u::AbstractMatrix{T}, W::Int, g::Real, sigma_a::Real, sigma_g::Real) where T<:Real
    N = size(u, 2)
    sigma2_a = sigma_a^2
    sigma2_g = sigma_g^2
    test_stats = Vector{Float64}(undef, N - W + 1)

    for k in 1:(N-W+1)
        window_acc = u[1:3, k:k+W-1]
        window_gyr = u[4:6, k:k+W-1]

        ya_m = mean(window_acc, dims=2)[:]          # (3,)
        g_ref = g * ya_m / norm(ya_m)

        acc_residuals = window_acc .- g_ref
        T_val = (sum(window_gyr .^ 2) / sigma2_g + sum(acc_residuals .^ 2) / sigma2_a) / W
        test_stats[k] = T_val
    end

    return test_stats
end