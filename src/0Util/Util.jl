function rolling_mean(arr::Vector{T}, n::Int) where T<:Real
    so_far = sum(arr[1:n])
    out = zero(arr[n:end])
    out[1] = so_far
    for (i, (start, stop)) in enumerate(zip(arr, arr[n+1:end]))
        so_far += stop - start
        out[i+1] = so_far / n
    end
    return out
end

function log_around(base::Float64, exp_range::Tuple{Float64,Float64}, n_steps::Int)
    lo, hi = exp_range
    exps = range(lo, hi, length=n_steps)
    return base .* 10.0 .^ exps
end

const Optional{T} = Union{Nothing,T}