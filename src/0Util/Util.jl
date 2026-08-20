function rolling_mean(arr::Vector{T}, n::Int) where T<:Real
    length(arr) >= n || throw(ArgumentError("n must be ≤ length(arr), got n=$n, length=$(length(arr))"))
    # Causal moving sum: conv with ones(n) produces length = length(arr) + n - 1
    sums = conv(arr, ones(n))[1:length(arr)]
    # For each sample i (1-based), the number of averaged points = min(i, n)
    divisors = min.(1:length(arr), n)
    return sums ./ divisors
end

function log_around(base::Float64, exp_range::Tuple{Float64,Float64}, n_steps::Int)
    lo, hi = exp_range
    exps = range(lo, hi, length=n_steps)
    return base .* 10.0 .^ exps
end

const Optional{T} = Union{Nothing,T}

"""
    gcc_phat(x1::AbstractVector, x2::AbstractVector, fs::Real)

Compute the Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
between two signals `x1` and `x2`, and return the estimated time delay.

# Arguments
- `x1::AbstractVector`: First input signal.
- `x2::AbstractVector`: Second input signal.
- `fs::Real`: Sampling frequency of both signals (Hz).

# Returns
- `tau::Float64`: Estimated time delay (seconds) between `x1` and `x2`.
  A positive value means `x2` is delayed relative to `x1`.
- `cc::Vector{Float64}`: Full cross-correlation sequence after PHAT weighting.
- `lags::UnitRange{Int64}`: Lag indices (in samples) corresponding to `cc`.
"""
function gcc_phat(x1::AbstractVector, x2::AbstractVector, fs::Real)
    n = nextpow(2, length(x1) + length(x2))
    X1 = FFTW.fft(vcat(x1, zeros(n - length(x1))))
    X2 = FFTW.fft(vcat(x2, zeros(n - length(x2))))
    R = X1 .* conj(X2)
    R ./= (abs.(R) .+ eps())   # PHAT weighting
    cc = real(FFTW.ifft(R))
    cc = FFTW.fftshift(cc)
    lags = (-n÷2):(n÷2-1)
    _, idx = findmax(cc)
    return lags[idx] / fs, cc, lags
end

"""
    resample_to_grid(t, x, t_grid)

Resample a 1D signal `x` (sampled at times `t`) onto a new time grid `t_grid`
using linear interpolation with linear extrapolation outside the original range.

Returns a vector of interpolated values at each point in `t_grid`.
"""
function resample_to_grid(t::AbstractVector, x::AbstractVector, t_grid::AbstractVector)
    itp = Interpolations.LinearInterpolation(t, x, extrapolation_bc=Interpolations.Line())
    return itp.(t_grid)
end

"""Mahalanobis square distance, robust to near-singular / non-symmetric S."""
function mahalanobis(nu::AbstractVector, S::AbstractMatrix)
    Ss = Symmetric((Matrix(S) + Matrix(S)') / 2)
    try
        return dot(nu, Ss \ nu)
    catch
        return dot(nu, pinv(Matrix(Ss)) * nu)
    end
end

"""Wrap an angle to (-pi, pi]."""
wrap_pi(a) = atan(sin(a), cos(a))