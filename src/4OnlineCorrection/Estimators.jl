function stride_heading(;
    R_wb::AbstractMatrix{T}, ΔpΔθ3::AbstractVector{T}, Σ_ΔpΔθ3::Union{Nothing,AbstractMatrix{Float64}}=nothing
)::Tuple{AbstractVector{T},Union{Nothing,AbstractMatrix{Float64}},AbstractMatrix{Float64}} where T<:Real
    temp = zeros(Float64, 4, 4)
    temp[4, 4] = matrix_to_euler(R_wb')[3]

    temp[1:3, 1:3] = euler_to_matrix(temp[2:4, 4])
    temp[4, 4] = 1.0
    return temp * ΔpΔθ3, isnothing(Σ_ΔpΔθ3) ? nothing : temp * Σ_ΔpΔθ3 * temp', Matrix(temp')
end

function stride_body(;
    R_wb::AbstractMatrix{T}, ΔpΔθ3::AbstractVector{T}, Σ_ΔpΔθ3::Union{Nothing,AbstractMatrix{Float64}}=nothing
)::Tuple{AbstractVector{T},Union{Nothing,AbstractMatrix{Float64}},AbstractMatrix{Float64}} where T<:Real
    temp = zeros(Float64, 4, 4)
    temp[1:3, 1:3] = R_wb'
    temp[4, 4] = 1.0
    return temp * ΔpΔθ3, isnothing(Σ_ΔpΔθ3) ? nothing : temp * Σ_ΔpΔθ3 * temp', Matrix(temp')
end

function stride_local(ref_frame::ReferenceFrame;
    R_wb::AbstractMatrix{T}, ΔpΔθ3::AbstractVector{T}, Σ_ΔpΔθ3::Union{Nothing,AbstractMatrix{Float64}}=nothing
)::Tuple{AbstractVector{T},Optional{AbstractMatrix{Float64}},AbstractMatrix{Float64}} where T<:Real
    return Dict{ReferenceFrame,Function}(
        HEADING => () -> stride_heading(; R_wb=R_wb, ΔpΔθ3=ΔpΔθ3, Σ_ΔpΔθ3=Σ_ΔpΔθ3),
        BODY => () -> stride_body(; R_wb=R_wb, ΔpΔθ3=ΔpΔθ3, Σ_ΔpΔθ3=Σ_ΔpΔθ3),
    )[ref_frame]()
end


function compute_feature(feature_type::FeatureType;
    ins_stride::AbstractVector{T}, Σ_ins_stride::Union{Nothing,AbstractMatrix{T}}=nothing,
    ΔT::T, ΔT_var::T=1e-8,
)::Tuple{AbstractVector{T},Union{Nothing,AbstractMatrix{T}}} where T<:Real
    feature_fun = Dict{FeatureType,Function}(
        THREED_STEP => () -> ins_stride[1:3],
        TWOD_STEP_DT => () -> let
            @info "got $feature_type" maxlog = 5
            [ins_stride[1:2]; ΔT]
        end,
        THREED_STEP_DT => () -> [ins_stride[1:3]; ΔT],
        AUG_STEP => () -> ins_stride,
        TWOD_STEP_YAW => () -> [ins_stride[1:2]; ins_stride[4]],
        TWOD_STEP_DT_YAW => () -> [ins_stride[1:2]; ΔT; ins_stride[4]],
        THREED_STEP_DT_YAW => () -> [ins_stride[1:3]; ΔT; ins_stride[4]],
    )[feature_type]
    @info "$feature_type" maxlog = 5

    feature_cov_fun = Dict{FeatureType,Function}(
        THREED_STEP => () -> Σ_ins_stride[1:3, 1:3],
        TWOD_STEP_DT => () -> [
            Σ_ins_stride[1:2, 1:2] zeros(Float64, (2, 1));
            zeros(Float64, (1, 2)) ΔT_var
        ],
        THREED_STEP_DT => () -> [
            Σ_ins_stride[1:3, 1:3] zeros(Float64, (3, 1));
            zeros(Float64, (1, 3)) ΔT_var
        ],
        AUG_STEP => () -> Σ_ins_stride,
        TWOD_STEP_YAW => () -> Σ_ins_stride[[1:2; 4], [1:2; 4]],
        TWOD_STEP_DT_YAW => () -> [
            Σ_ins_stride[1:2, 1:2] zeros(Float64, (2, 1)) Σ_ins_stride[1:2, 4:4];
            zeros(Float64, (1, 2)) ΔT_var 0.0;
            Σ_ins_stride[4:4, 1:2] 0.0 Σ_ins_stride[4, 4]
        ],
        THREED_STEP_DT_YAW => () -> [
            Σ_ins_stride[1:3, 1:3] zeros(Float64, (3, 1)) Σ_ins_stride[1:3, 4:4];
            zeros(Float64, (1, 3)) ΔT_var 0.0;
            Σ_ins_stride[4:4, 1:3] 0.0 Σ_ins_stride[4, 4]
        ],
    )[feature_type]
    return feature_fun(), isnothing(Σ_ins_stride) ? nothing : feature_cov_fun()
end

function ∂feature_norm∂δpδθ(feature_type::FeatureType; σ_input::AbstractVector{T}, R_aug_wl::AbstractMatrix{T}, q_curr::AbstractVector{T}) where T<:Real
    feature_deriv_fun = Dict{FeatureType,Function}(
        THREED_STEP => () -> ∂feature_norm𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:3]),
        TWOD_STEP_DT => () -> [
            ∂feature_norm𝑥𝑦_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:2]);
            ∂feature_normΔT_∂δpδθ(T)'
        ],
        THREED_STEP_DT => () -> [
            ∂feature_norm𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:3]);
            ∂feature_normΔT_∂δpδθ(T)'
        ],
        AUG_STEP => () -> [
            ∂feature_norm𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:3]);
            ∂feature_normΔθ3_∂δpδθ(q_curr, σ_input[4])'
        ],
        TWOD_STEP_YAW => () -> [
            ∂feature_norm𝑥𝑦_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:2]);
            ∂feature_normΔθ3_∂δpδθ(q_curr, σ_input[3])'
        ],
        TWOD_STEP_DT_YAW => () -> [
            ∂feature_norm𝑥𝑦_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:2]);
            ∂feature_normΔT_∂δpδθ(T)';
            ∂feature_normΔθ3_∂δpδθ(q_curr, σ_input[4])'
        ],
        THREED_STEP_DT_YAW => () -> [
            ∂feature_norm𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_input[1:3]);
            ∂feature_normΔT_∂δpδθ(T)';
            ∂feature_normΔθ3_∂δpδθ(q_curr, σ_input[4])'
        ]
    )[feature_type]
    return feature_deriv_fun()
end

function normalize_feature!(
    feature_type::FeatureType;
    feature::AbstractVector{Float64},
    Σ_feature::Union{Nothing,AbstractMatrix{Float64}}=nothing,
    input_stats::Vector{Vector{Float64}},
    mid_norm::Vector{Float64}
)::Tuple{AbstractVector{Float64},Union{Nothing,AbstractMatrix{Float64}}}

    μ = input_stats[1]
    σ = input_stats[2]

    # In-place normalization
    feature .-= μ

    # Indices corresponding to angular variables
    angle_idx = Dict(
        AUG_STEP => [4],
        TWOD_STEP_YAW => [3],
        TWOD_STEP_DT_YAW => [4],
        THREED_STEP => Int[],
        TWOD_STEP_DT => Int[],
        THREED_STEP_DT => Int[],
        THREED_STEP_DT_YAW => [5]
    )[feature_type]

    # Wrap angles after mean subtraction
    for i in angle_idx
        feature[i] = atan(sin(feature[i]), cos(feature[i]))
    end

    # Scale features in-place
    feature ./= σ

    # Center feature
    feature .-= mid_norm

    # Covariance normalization
    Σ_feature_norm = if isnothing(Σ_feature)
        nothing
    else
        D = Diagonal(1.0 ./ σ)
        D * Σ_feature * D
    end

    return feature, Σ_feature_norm
end

const _OUTPUT_NAMES = ["pos_1", "pos_2", "pos_3", "yaw"]
_full_range(orig_idx::Int, m::Int) = ((orig_idx-1)*m+1):(orig_idx*m)

abstract type AbstractEstimator end

function initialize_corrector!(
    c::AbstractEstimator;
    t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwarg...)
    error("initialize_corrector! not implemented for $(typeof(c))")
end

function dynamic_update!(c::AbstractEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwarg...)
    error("dynamic_update! not implemented for $(typeof(c))")
end

function stride_measurement_update!(c::AbstractEstimator; feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    ins_stride::AbstractVector{Float64}, Σ_ins_stride::AbstractMatrix{Float64},
    kwargs...
)
    error("stride_measurement_update! not implemented for $(typeof(c))")
end

function posyaw_measurement_update!(c::AbstractEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    error("posyaw_measurement_update! not implemented for $(typeof(c))")
end

function learned_measurement_update!(c::AbstractEstimator;
    kwargs...
)::NTuple{4,Optional{AbstractVector{Float64}}}
    error("learned_measurement_update! not implemented for $(typeof(c))")
end

function relinearize!(c::AbstractEstimator; kwarg...)
    error("relinearize! not implemented for $(typeof(c))")
end

function get_β_Σβ(c::AbstractEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    error("get_β_Σβ not implemented for $(typeof(c))")
end

# Accessors
function get_time(c::AbstractEstimator)::AbstractVector{Float64}
    return c.t[1:c.i]
end

function get_pos(c::AbstractEstimator)::AbstractMatrix{Float64}
    return c.pos[:, 1:c.i]
end

function get_quat(c::AbstractEstimator)::AbstractMatrix{Float64}
    return c.quat[:, 1:c.i]
end

function get_trajectory(c::AbstractEstimator)::Trajectory
    Trajectory(
        get_time(c),
        get_pos(c),
        quat_to_matrix(get_quat(c))
    )
end


function stride_error(ref_frame::ReferenceFrame;
    R_wb::NTuple{2,AbstractMatrix{Float64}},
    Δp::AbstractVector{Float64},
    Σ_ΔpΔθ3::AbstractMatrix{Float64}=nothing,
    R_wb_gt::NTuple{2,AbstractMatrix{Float64}},
    Δp_gt::AbstractVector{Float64},
    Σ_ΔpΔθ3_gt::AbstractMatrix{Float64}=nothing
)::Tuple{AbstractVector{Float64},AbstractMatrix{Float64},AbstractVector{Float64},AbstractMatrix{Float64},AbstractMatrix{Float64}}
    # Compute ground truth stride in local frame
    Δθ3_gt = matrix_to_euler(R_wb_gt[2])[3] -
             matrix_to_euler(R_wb_gt[1])[3]
    Δθ3_gt = atan(sin(Δθ3_gt), cos(Δθ3_gt))

    gt_stride, Σ_gt_stride, _ = stride_local(ref_frame;
        R_wb=R_wb_gt[1],
        ΔpΔθ3=[Δp_gt; Δθ3_gt],
        Σ_ΔpΔθ3=Σ_ΔpΔθ3_gt
    )

    # Compute estimated stride from INS in local frame
    Δθ3 = matrix_to_euler(R_wb[2])[3] -
          matrix_to_euler(R_wb[1])[3]
    Δθ3 = atan(sin(Δθ3), cos(Δθ3))

    ins_stride, Σ_ins_stride, R_aug_wl = stride_local(ref_frame;
        R_wb=R_wb[1],
        ΔpΔθ3=[Δp; Δθ3],
        Σ_ΔpΔθ3=Σ_ΔpΔθ3
    )

    # Compute stride error
    stride_err = gt_stride - ins_stride
    stride_err[4] = atan(sin(stride_err[4]), cos(stride_err[4]))  # Wrap angle to [-π, π]

    # Combine covariances from ground truth and INS estimates
    Σ_err = Σ_gt_stride + Σ_ins_stride

    return stride_err, Σ_err, ins_stride, Σ_ins_stride, R_aug_wl
end

# ── Concrete correctors ───────────────────────────────────────────────────
mutable struct BaseEstimator <: AbstractEstimator
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractMatrix{Float64}
    Σ::AbstractMatrix{Float64}
    G::AbstractMatrix{Float64}
    H::AbstractArray{Float64,3}
    i::Int
    F::AbstractMatrix{Float64}
end

function BaseEstimator(N::Int)::BaseEstimator
    @assert N > 1 "Invalid number of "
    return BaseEstimator(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 6, N),
        zeros(Float64, 6, 6), zeros(Float64, 6, 6), zeros(Float64, 4, 6, N), 1, zeros(Float64, 6, 6))
end

function initialize_corrector!(c::BaseEstimator; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ .= Σpq_init
    c.G .= Matrix{Float64}(I, 6, 6)
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::BaseEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    # c.β = c.β
    c.G .= 0.0
    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F .= 0.0
    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)

    # Covariance update
    c.δx[:, c.i] .= 0.0
    c.Σ .= c.F * c.Σ * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::BaseEstimator;
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...
)
    # c.H[1:3, 1:3, c.i] = R_aug_wl[1:3, 1:3]' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # # c.H[4, 4:6, c.i] = ∂θ3_∂δθ_left(c.quat[:, c.i])
    # c.H[4, 6, c.i] = 1.0

    # c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
    #     c.δx[:, c.i], c.Σ[:, :, c.i],
    #     stride_err,
    #     c.H[:, :, c.i],
    #     Σ_err
    # )
    return nothing, nothing
end

function posyaw_measurement_update!(c::BaseEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0] #  ∂θ3_∂δθ_left(c.quat[:, c.i])
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::BaseEstimator; kwargs...)::NTuple{4,Optional{AbstractVector{Float64}}}
    # Do nothing
    return nothing, nothing, nothing, nothing
end

function relinearize!(c::BaseEstimator)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.δx[:, c.i] .= 0.0
end

function get_β_Σβ(c::BaseEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    return nothing
end

mutable struct JointStaticEstimator <: AbstractEstimator
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractMatrix{Float64}
    Σ::AbstractMatrix{Float64}
    G::AbstractMatrix{Float64}
    H::AbstractMatrix{Float64}
    i::Int
    F::AbstractMatrix{Float64}

    # Corrector specific Arguments
    stride_bias::AbstractMatrix{Float64}

    # Kalman worspace
    ws::KalmanWorkspace{Float64}

    # Channel-selection
    correction_mask::Vector{Int}   # indices into [:pos_1, :pos_2, :pos_3, :yaw] that are corrected
    p::Int                          # number of corrected channels

end

function JointStaticEstimator(N::Int;
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw]
)::JointStaticEstimator
    @assert N > 1 "Invalid number of allocations, got $N"

    correction_mask = Int[]
    for (idx, sym) in enumerate([:pos_1, :pos_2, :pos_3, :yaw])
        if sym in corrected_channels
            push!(correction_mask, idx)
        end
    end
    p = length(correction_mask)
    @assert p >= 1 "Must correct at least one channel"

    return JointStaticEstimator(
        zeros(Float64, N),          # t
        zeros(Float64, 3, N),       # pos
        zeros(Float64, 4, N),       # quat
        zeros(Float64, 6+p, N),      # δx
        zeros(Float64, 6 + p, 6+p),     # Σ
        zeros(Float64, 6+p, 6),      # G
        zeros(Float64, 4, 6+p),      # H
        1,                          # i
        zeros(Float64, 6+p, 6+p),     # F
        zeros(Float64, p, N),        # stride_bias
        KalmanWorkspace{Float64}(6+p, 4),
        correction_mask,
        p
    )
end

function initialize_corrector!(c::JointStaticEstimator; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.stride_bias[:, 1] .= 0.0
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6] .= Σpq_init
    c.Σ[7:end, 7:end] = Matrix{Float64}(I, c.p, c.p) .* 1e2
    # c.Σ[10, 10] *= 1e2
    c.G .= Matrix{Float64}(I, 6+c.p, 6)
    c.H .= 0.0
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::JointStaticEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    c.stride_bias[:, c.i] = c.stride_bias[:, c.i-1]

    c.G .= 0.0
    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F .= 0.0
    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)
    c.F[7:end, 7:end] = Matrix{Float64}(I, c.p, c.p)

    # Covariance update
    c.δx[:, c.i] .= 0.0
    c.Σ .= c.F * c.Σ * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::JointStaticEstimator;
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)
    # c.H[1:3, 1:3] = R_aug_wl[1:3, 1:3]' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # c.H[4, 4:6] = [0.0, 0.0, 1.0]
    # c.H[:, 7:end] = Matrix{Float64}(I, 4, 4)

    # stride_err -= c.stride_bias[:, c.i]
    # # stride_err[4] = atan(sin(stride_err[4]), cos(stride_err[4]))

    Hfull = zeros(Float64, 4, 6)
    Hfull[1:3, 1:3] = R_aug_wl[1:3, 1:3]'
    Hfull[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[1:c.p, 1:6] .= Hfull[c.correction_mask, :]
    c.H[1:c.p, 7:(6+c.p)] .= Matrix{Float64}(I, c.p, c.p)


    measurement_update!(
        view(c.δx, 1:(6+c.p), c.i), c.Σ,
        stride_err[c.correction_mask] - c.stride_bias[:, c.i],
        c.H[1:c.p, :],
        Σ_err[c.correction_mask, c.correction_mask],
        c.ws
    )
    return nothing, nothing
end

function posyaw_measurement_update!(c::JointStaticEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[:, 7:end] .= 0.0
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ

    measurement_update!(
        view(c.δx, 1:(6+c.p), c.i), c.Σ,
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H,
        Σy,
        c.ws
    )
end

function learned_measurement_update!(c::JointStaticEstimator;
    R_aug_wl, kwargs...)::NTuple{4,Optional{AbstractVector{Float64}}}

    Hfull = zeros(Float64, 4, 6+c.p)
    Hfull[1:3, 1:3] = R_aug_wl[1:3, 1:3]'
    Hfull[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[1:c.p, :] .= Hfull[c.correction_mask, :]


    measurement_update!(
        view(c.δx, 1:(6+c.p), c.i), c.Σ,
        c.stride_bias[:, c.i],
        c.H[1:c.p, :],
        c.Σ[7:end, 7:end],
        c.ws
    )
    # get full pred
    pred_full, Σ_pred_full = zeros(4), zeros(4, 4)
    pred_full[c.correction_mask] = c.stride_bias[:, c.i]
    Σ_pred_full[c.correction_mask, c.correction_mask] = c.Σ[7:end, 7:end]
    return pred_full, diag(Σ_pred_full), nothing, nothing
end

function relinearize!(c::JointStaticEstimator)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.stride_bias[:, c.i] += c.δx[7:end, c.i]
    c.δx[:, c.i] .= 0.0

    # c.Σ[1:6, 7:end] .= 0.0
    # c.Σ[7:end, 1:6] .= 0.0 # zero cross covs

end

function get_β_Σβ(c::JointStaticEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    return nothing
end

mutable struct DecoupledStaticEstimator <: AbstractEstimator
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractVector{Float64}
    Σ::AbstractMatrix{Float64}
    G::AbstractMatrix{Float64}
    H::AbstractMatrix{Float64}
    i::Int
    F::AbstractMatrix{Float64}

    # Corrector specific Arguments
    stride_bias::AbstractVector{Float64}
    Σ_bias::AbstractMatrix{Float64}


    ws_state::KalmanWorkspace{Float64}
    ws_bias::KalmanWorkspace{Float64}

    # Channel-selection
    correction_mask::Vector{Int}   # indices into [:pos_1, :pos_2, :pos_3, :yaw] that are corrected
    p::Int                          # number of corrected channels

end

function DecoupledStaticEstimator(N::Int;
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw]
)::DecoupledStaticEstimator
    @assert N > 1 "Invalid number of allocations, got $N"

    correction_mask = Int[]
    for (idx, sym) in enumerate([:pos_1, :pos_2, :pos_3, :yaw])
        if sym in corrected_channels
            push!(correction_mask, idx)
        end
    end
    p = length(correction_mask)
    @assert p >= 1 "Must correct at least one channel"

    return DecoupledStaticEstimator(
        zeros(Float64, N),          # t
        zeros(Float64, 3, N),       # pos
        zeros(Float64, 4, N),       # quat
        zeros(Float64, 6),      # δx
        zeros(Float64, 6, 6),     # Σ
        zeros(Float64, 6, 6),      # G
        zeros(Float64, 4, 6),      # H
        1,                          # i
        zeros(Float64, 6, 6),     # F
        zeros(Float64, p),        # stride_bias
        zeros(Float64, p, p),
        KalmanWorkspace{Float64}(6, 4),
        KalmanWorkspace{Float64}(p, p),
        correction_mask,
        p
    )
end

function initialize_corrector!(c::DecoupledStaticEstimator; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ .= Σpq_init

    c.stride_bias .= 0.0
    c.Σ_bias = Matrix{Float64}(I, c.p, c.p) .* 1e2

    c.G .= Matrix{Float64}(I, 6, 6)
    c.H .= 0.0
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::DecoupledStaticEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)

    c.G .= 0.0
    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F .= 0.0
    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)

    # Covariance update
    c.δx .= 0.0
    c.Σ .= c.F * c.Σ * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::DecoupledStaticEstimator;
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64}, kwargs...)
    c.H .= 0.0
    for i in 1:c.p
        c.H[i, i] = 1.0
    end

    measurement_update!(
        c.stride_bias, c.Σ_bias,
        stride_err[c.correction_mask],
        c.H[1:c.p, 1:c.p],
        Σ_err[c.correction_mask, c.correction_mask],
        c.ws_bias
    )
    residual = zeros(4)
    residual[c.correction_mask] .= stride_err[c.correction_mask] - c.stride_bias
    return residual, nothing
end

function posyaw_measurement_update!(c::DecoupledStaticEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H .= 0.0
    c.H[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]

    measurement_update!(
        c.δx, c.Σ,
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H,
        Σy,
        c.ws_state
    )
end

function learned_measurement_update!(c::DecoupledStaticEstimator;
    R_aug_wl, kwargs...)::NTuple{4,Optional{AbstractVector{Float64}}}

    # Update measurement matrix H_update
    c.H .= 0.0
    Hfull = zeros(Float64, 4, 6)
    Hfull[1:3, 1:3] = R_aug_wl[1:3, 1:3]'
    Hfull[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[1:c.p, 1:6] .= Hfull[c.correction_mask, :]


    measurement_update!(
        c.δx, c.Σ,
        c.stride_bias,
        c.H[1:c.p, 1:6],
        c.Σ_bias,
        c.ws_state
    )
    pred_full, Σ_pred_full = zeros(4), zeros(4, 4)
    pred_full[c.correction_mask] = c.stride_bias
    Σ_pred_full[c.correction_mask, c.correction_mask] = c.Σ_bias
    return pred_full, diag(Σ_pred_full), nothing, nothing
end

function relinearize!(c::DecoupledStaticEstimator)
    c.pos[:, c.i] += c.δx[1:3]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6]), c.quat[:, c.i])
    c.δx .= 0.0
end

function get_β_Σβ(c::DecoupledStaticEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    return nothing
end

mutable struct DecoupledHsgpEstimator <: AbstractEstimator
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractMatrix{Float64}
    Σ::AbstractMatrix{Float64}
    G::AbstractMatrix{Float64}
    H::AbstractArray{Float64,3}
    i::Int
    F::AbstractMatrix{Float64}

    # Corrector specific Arguments
    params::HsgpParameters
    β::AbstractVector{Float64}
    Σβ::AbstractMatrix{Float64}
    ∂y∂z::AbstractMatrix{Float64}
    Φ::AbstractMatrix{Float64}
    per_dim_eigvals::AbstractMatrix{Float64}
    σ_n::AbstractVector{Float64}

    ws_state::KalmanWorkspace{Float64}
    ws_hsgp::KalmanWorkspace{Float64}

    # Channel-selection
    correction_mask::Vector{Int}   # indices into [:pos_1, :pos_2, :pos_3, :yaw] that are corrected
    p::Int                          # number of corrected channels
end

function DecoupledHsgpEstimator(N::Int, params::HsgpParameters;
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw])::DecoupledHsgpEstimator
    @assert N > 1 "Invalid number of allocations, got $N"

    correction_mask = Int[]
    for (idx, sym) in enumerate([:pos_1, :pos_2, :pos_3, :yaw])
        if sym in corrected_channels
            push!(correction_mask, idx)
        end
    end
    p = length(correction_mask)
    @assert p >= 1 "Must correct at least one channel"

    return DecoupledHsgpEstimator(
        zeros(Float64, N),                              # t
        zeros(Float64, 3, N),                           # pos
        zeros(Float64, 4, N),                           # quat
        zeros(Float64, 6, N),                           # δx
        zeros(Float64, 6, 6),                           # Σ
        zeros(Float64, 6, 6),                           # G
        zeros(Float64, 4, 6, N),                        # H
        1,                                              # i
        zeros(Float64, 6, 6),                           # F
        params,                                         # params
        zeros(Float64, params.m * p),                   # β
        zeros(Float64, params.m * p, params.m * p),     # Σβ
        zeros(Float64, p, params.d),                    # ∂y∂z
        zeros(Float64, p, params.m * p),                # Φ
        zeros(Float64, params.m, params.d),             # per_dim_eigvals
        zeros(Float64, p),                              # σ_n
        KalmanWorkspace{Float64}(6, 4),
        KalmanWorkspace{Float64}(params.m * p, p),
        correction_mask,
        p
    )
end


function initialize_corrector!(c::DecoupledHsgpEstimator;
    t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64},
    β_Σβ_0::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}=nothing, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6] = Σpq_init
    c.G[:, :, 1] = Matrix{Float64}(I, 6, 6)

    m = c.params.m
    p = c.p
    mask = c.correction_mask

    # Initialize HSGP
    c.per_dim_eigvals = calc_eigenvalues(c.params.LL, c.params.m, c.params.d)
    psd = zeros(Float64, p * m)
    for (j, orig_idx) in enumerate(mask)
        field = Symbol(_OUTPUT_NAMES[orig_idx])
        psd[((j-1)*m+1):(j*m)] = power_spectral_density(
            sqrt.(c.per_dim_eigvals),
            getfield(c.params.hp, field)[2],
            getfield(c.params.hp, field)[3]
        )
    end

    for (j, orig_idx) in enumerate(mask)
        field = Symbol(_OUTPUT_NAMES[orig_idx])
        c.σ_n[j] = getfield(c.params.hp, field)[1]
    end

    # β_Σβ_0 is always full 4m-sized (matching JointHsgpEstimator's convention);
    # extract only the masked sub-blocks into the reduced internal state.
    if isnothing(β_Σβ_0)
        c.β .= 0.0
    else
        β_full, _ = β_Σβ_0
        @assert length(β_full) == 4 * m "β_Σβ_0[1] must have length 4m = $(4*m), got $(length(β_full))"
        for j in 1:p
            orig_idx = mask[j]
            c.β[((j-1)*m+1):(j*m)] = β_full[_full_range(orig_idx, m)]
        end
    end

    c.Σβ .= 0.0
    if isnothing(β_Σβ_0)
        c.Σβ[diagind(c.Σβ)] .= psd
    else
        _, Σβ_full = β_Σβ_0
        @assert size(Σβ_full) == (4 * m, 4 * m) "β_Σβ_0[2] must be (4m,4m) = $((4*m,4*m)), got $(size(Σβ_full))"
        for j1 in 1:p, j2 in 1:p
            r1 = _full_range(mask[j1], m)
            r2 = _full_range(mask[j2], m)
            c.Σβ[((j1-1)*m+1):(j1*m), ((j2-1)*m+1):(j2*m)] = Σβ_full[r1, r2]
        end
    end

    c.∂y∂z .= 0.0
    c.Φ .= 0.0
    c.G .= Matrix{Float64}(I, 6, 6)
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::DecoupledHsgpEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    c.G .= 0.0
    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F .= 0.0
    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)

    # Covariance update
    c.δx[:, c.i] .= 0.0
    c.Σ .= c.F * c.Σ * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::DecoupledHsgpEstimator;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64},
    kwargs...)

    p = c.p
    mask = c.correction_mask
    m = c.params.m

    # Mask down to corrected channels only — β/Σβ update is fully reduced-dim
    stride_err_masked = (stride_err[mask] .- c.params.output_stats[1][mask]) ./ c.params.output_stats[2][mask]
    Σ_err_masked = Diagonal(1 ./ c.params.output_stats[2][mask]) * Σ_err[mask, mask] * Diagonal(1 ./ c.params.output_stats[2][mask])

    # Compute input feature and normalize
    normalize_feature!(feature_type;
        feature=feature, Σ_feature=Σ_feature, input_stats=c.params.input_stats, mid_norm=c.params.mid_norm)

    # ------------ Construct Feature Covariance (masked channels only) ---------
    for output_d in 1:p
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d] = dot(calc_eigenvectors_dx(
                    reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
                ), c.β[((output_d-1)*m+1):(output_d*m)])
        end
    end
    Σ_err_masked += c.∂y∂z * Σ_feature * c.∂y∂z'

    # ---------- Construct measurement matrix Φ (p × p*m) --------------
    kron!(c.Φ, I(p), calc_eigenvectors(
        reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals)
    )

    # α = tr(Diagonal(c.σ_n .^ 2)) / tr(Σ_err_masked)
    measurement_update!(c.β, c.Σβ, stride_err_masked, c.Φ, Σ_err_masked, c.ws_hsgp)

    return nothing, nothing
end

function posyaw_measurement_update!(c::DecoupledHsgpEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]

    measurement_update!(
        view(c.δx, 1:6, c.i), c.Σ,
        [curr_pos .- c.pos[:, c.i]; atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))],
        c.H[:, :, c.i],
        Σy,
        c.ws_state
    )
end

function learned_measurement_update!(c::DecoupledHsgpEstimator;
    feature_type::FeatureType,
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)::NTuple{4,Optional{AbstractVector{Float64}}}

    p = c.p
    mask = c.correction_mask
    m = c.params.m

    # Normalise estimated feature and feature covariance
    normalize_feature!(feature_type;
        feature=feature, Σ_feature=Σ_feature, input_stats=c.params.input_stats, mid_norm=c.params.mid_norm)

    for output_d in 1:p
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d:input_d] = calc_eigenvectors_dx(
                reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
            ) * c.β[((output_d-1)*m+1):(output_d*m)]
        end
    end

    kron!(c.Φ, I(p), calc_eigenvectors(reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals))

    # Compute prediction and prediction covariance (reduced p-dim)
    pred_masked = c.Φ * c.β
    Σ_pred_masked = c.Φ * c.Σβ * c.Φ' + c.∂y∂z * Σ_feature * c.∂y∂z'   # Predictive + input uncertainty

    # Denormalise (masked channels only)
    pred_masked = pred_masked .* c.params.output_stats[2][mask] .+ c.params.output_stats[1][mask]
    Σ_pred_masked = Diagonal(c.params.output_stats[2][mask]) * Σ_pred_masked * Diagonal(c.params.output_stats[2][mask])

    # Update measurement matrix H_update
    c.H[1:p, 1:6, c.i] .= 0.0
    Hfull = zeros(Float64, 4, 6)
    Hfull[1:3, 1:3] = R_aug_wl[1:3, 1:3]'
    Hfull[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[1:p, 1:6, c.i] .= Hfull[mask, :]


    measurement_update!(
        view(c.δx, 1:6, c.i), c.Σ,
        pred_masked,
        c.H[1:p, 1:6, c.i],
        Σ_pred_masked,
        c.ws_state
    )

    # Output 4 channel convention
    pred_full, Σ_pred_full = zeros(4), zeros(4, 4)
    pred_full[mask] = pred_masked
    Σ_pred_full[mask, mask] = Σ_pred_masked

    return pred_full, diag(Σ_pred_full), nothing, nothing
end

function relinearize!(c::DecoupledHsgpEstimator)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    # c.δx[:, c.i] .= 0.0
end

function get_β_Σβ(c::DecoupledHsgpEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    m = c.params.m
    p = c.p
    mask = c.correction_mask

    β_full = zeros(Float64, 4 * m)
    Σβ_full = zeros(Float64, 4 * m, 4 * m)

    for j in 1:p
        orig_idx = mask[j]
        β_full[_full_range(orig_idx, m)] = c.β[((j-1)*m+1):(j*m)]
    end

    for j1 in 1:p, j2 in 1:p
        r1 = _full_range(mask[j1], m)
        r2 = _full_range(mask[j2], m)
        Σβ_full[r1, r2] = c.Σβ[((j1-1)*m+1):(j1*m), ((j2-1)*m+1):(j2*m)]
    end

    return β_full, Σβ_full
end

mutable struct JointHsgpEstimator <: AbstractEstimator
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractVector{Float64}
    Σ::AbstractMatrix{Float64}
    G::AbstractMatrix{Float64}
    H::AbstractMatrix{Float64}
    i::Int
    F::AbstractMatrix{Float64}

    # Corrector specific Arguments
    params::HsgpParameters
    β::AbstractVector{Float64}
    ∂y∂z::AbstractMatrix{Float64}
    ϕ::AbstractMatrix{Float64}
    per_dim_eigvals::AbstractMatrix{Float64}
    ws::KalmanWorkspace{Float64}

    # Channel-selection
    correction_mask::Vector{Int}   # indices into [:pos_1, :pos_2, :pos_3, :yaw] that are corrected
    p::Int                          # number of corrected channels (length(correction_mask))
end

function JointHsgpEstimator(N::Int, params::HsgpParameters;
    corrected_channels::Vector{Symbol}=[:pos_1, :pos_2, :pos_3, :yaw]
)::JointHsgpEstimator
    @assert N > 1 "Invalid number of prealocations"

    correction_mask = Int[]
    for (idx, sym) in enumerate([:pos_1, :pos_2, :pos_3, :yaw])
        if sym in corrected_channels
            push!(correction_mask, idx)
        end
    end
    p = length(correction_mask)
    @assert p >= 1 "Must correct at least one channel"

    return JointHsgpEstimator(
        Vector{Float64}(undef, N),                                  # t
        Matrix{Float64}(undef, 3, N),                               # pos
        Matrix{Float64}(undef, 4, N),                               # quat
        Vector{Float64}(undef, 6 + p * params.m),                   # δx
        Matrix{Float64}(undef, 6 + p * params.m, 6 + p * params.m), # Σ
        Matrix{Float64}(undef, 6, 6),                               # G
        Matrix{Float64}(undef, 4, 6 + p * params.m),                # H (4 rows = max(measurement dims: p for stride/learned, 4 for posyaw))
        1,                                                          # i
        Matrix{Float64}(undef, 6, 6),                               # F
        params,                                                     # params
        Vector{Float64}(undef, params.m * p),                       # β
        Matrix{Float64}(undef, p, params.d),                        # ∂y∂z
        Matrix{Float64}(undef, 1, params.m),                        # ϕ
        Matrix{Float64}(undef, params.m, params.d),                 # per_dim_eigvals
        KalmanWorkspace{Float64}(6 + p * params.m, 4),
        correction_mask,
        p
    )
end

function initialize_corrector!(c::JointHsgpEstimator;
    t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64},
    β_Σβ_0::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}=nothing, kwargs...)
    c.i = 1
    c.t[1] = t
    m = c.params.m
    p = c.p
    mask = c.correction_mask

    # Initialize nominal states
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init

    if isnothing(β_Σβ_0)
        c.β .= 0.0
    else
        β_full, _ = β_Σβ_0
        @assert length(β_full) == 4 * m "β_Σβ_0[1] must have length 4m = $(4*m), got $(length(β_full))"
        for j in 1:p
            orig_idx = mask[j]
            c.β[((j-1)*m+1):(j*m)] = β_full[_full_range(orig_idx, m)]
        end
    end

    # Initialize error state
    c.δx .= 0.0
    # Initialize HSGP
    c.per_dim_eigvals = calc_eigenvalues(c.params.LL, c.params.m, c.params.d)

    psd = zeros(Float64, p * m)
    for (j, orig_idx) in enumerate(mask)
        field = Symbol(_OUTPUT_NAMES[orig_idx])
        psd[((j-1)*m+1):(j*m)] = power_spectral_density(
            sqrt.(c.per_dim_eigvals),
            getfield(c.params.hp, field)[2],
            getfield(c.params.hp, field)[3]
        )
    end

    # Initialize state covariance
    c.Σ .= 0.0
    c.Σ[1:6, 1:6] = Σpq_init
    if isnothing(β_Σβ_0)
        c.Σ[7:end, 7:end] = Diagonal(psd)
    else
        _, Σβ_full = β_Σβ_0
        @assert size(Σβ_full) == (4 * m, 4 * m) "β_Σβ_0[2] must be (4m,4m) = $((4*m,4*m)), got $(size(Σβ_full))"
        for j1 in 1:p, j2 in 1:p
            r1 = _full_range(mask[j1], m)
            r2 = _full_range(mask[j2], m)
            c.Σ[(6+(j1-1)*m+1):(6+j1*m), (6+(j2-1)*m+1):(6+j2*m)] = Σβ_full[r1, r2]
        end
    end

    c.∂y∂z .= 0.0
    c.ϕ .= 0.0
    c.G .= 0.0
    c.H .= 0.0
    c.F .= 0.0

end

function dynamic_update!(c::JointHsgpEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)

    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)

    # Covariance update
    c.δx .= 0.0
    c.Σ[1:6, 1:6] = c.F * c.Σ[1:6, 1:6] * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::JointHsgpEstimator;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)

    p = c.p
    mask = c.correction_mask
    m = c.params.m

    # ------ Construct measurement (masked to corrected channels) --------
    normalize_feature!(feature_type;
        feature=feature, Σ_feature=Σ_feature, input_stats=c.params.input_stats, mid_norm=c.params.mid_norm)

    c.ϕ = calc_eigenvectors(reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals)

    stride_err_masked = stride_err[mask]
    Σ_err_masked = Σ_err[mask, mask]

    # Compute denormalised prediction residual for corrected channels only
    for j in 1:p
        orig_idx = mask[j]
        stride_err_masked[j] -= c.params.output_stats[2][orig_idx] * dot(c.ϕ, c.β[((j-1)*m+1):(j*m)]) +
                                c.params.output_stats[1][orig_idx]
    end

    # ------- Construct measurement matrix 𝐇 = [Hp, Hθ, Hβ] ∈ R^{p × (6 + p m)}
    for output_d in 1:p
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d] = dot(calc_eigenvectors_dx(
                    reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
                ), c.β[((output_d-1)*m+1):(output_d*m)])
        end
    end
    c.∂y∂z = Diagonal(c.params.output_stats[2][mask]) * c.∂y∂z # denormalize prediction

    # Construct Hβ
    kron!(view(c.H, 1:p, 7:(6+p*m)), Diagonal(c.params.output_stats[2][mask]), c.ϕ)

    # Template Hp / Hθ over all 4 physical channels, then select the corrected rows
    Hfull = zeros(Float64, 4, 6)
    Hfull[1:3, 1:3] = R_aug_wl[1:3, 1:3]'
    Hfull[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[1:p, 1:6] .= Hfull[mask, :]

    # Add GP contribution to δp and δθ
    c.H[1:p, 1:6] .+= c.∂y∂z * ∂feature_norm∂δpδθ(
        feature_type; σ_input=c.params.input_stats[2], R_aug_wl=R_aug_wl, q_curr=c.quat[:, c.i])

    measurement_update!(c.δx, c.Σ, stride_err_masked, view(c.H, 1:p, :), Σ_err_masked, c.ws)
    stride_err_full = zeros(4)
    Σ_err_full = zeros(4, 4)
    stride_err_full[mask] = stride_err_masked
    Σ_err_full[mask, mask] = Σ_err_masked
    return stride_err_full, Σ_err_full
end

function posyaw_measurement_update!(c::JointHsgpEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[:, 7:end] .= 0.0

    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]

    measurement_update!(
        c.δx, c.Σ,
        [curr_pos .- c.pos[:, c.i]; atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))],
        view(c.H, 1:4, :), Σy,
        c.ws
    )
end

function learned_measurement_update!(c::JointHsgpEstimator;
    feature_type::FeatureType,
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)::NTuple{4,Optional{AbstractVector{Float64}}}

    p = c.p
    mask = c.correction_mask
    m = c.params.m

    # ------ Construct Pseudo Measurement (corrected channels only) ------
    normalize_feature!(feature_type;
        feature=feature, Σ_feature=Σ_feature, input_stats=c.params.input_stats, mid_norm=c.params.mid_norm)

    c.ϕ = calc_eigenvectors(reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals)

    pred = Vector{Float64}(undef, p)
    for j in 1:p
        pred[j] = dot(c.ϕ, c.β[((j-1)*m+1):(j*m)])
    end
    pred_norm = deepcopy(pred)
    pred = c.params.output_stats[2][mask] .* pred .+ c.params.output_stats[1][mask]

    for output_d in 1:p
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d:input_d] = calc_eigenvectors_dx(
                reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
            ) * c.β[((output_d-1)*m+1):(output_d*m)]
        end
    end

    # --- Compute covariance in normalized output space
    Σ_pred = Matrix{Float64}(undef, p, p)
    for row in 1:p
        for col in 1:p
            Σ_pred[row:row, col:col] = c.ϕ * c.Σ[((row-1)*m+7):(row*m+6), ((col-1)*m+7):(col*m+6)] * c.ϕ'
        end
    end

    Σ_pred += c.∂y∂z * Σ_feature * c.∂y∂z' # input uncertainty
    Σ_pred_norm = deepcopy(Σ_pred)

    # --- Denormalise prediction covariance ---
    Σ_pred = Diagonal(c.params.output_stats[2][mask]) * Σ_pred * Diagonal(c.params.output_stats[2][mask])

    # Update measurement matrix H_update
    c.H[1:p, :] .= 0.0
    Hfull = zeros(Float64, 4, 6)
    Hfull[1:3, 1:3] = R_aug_wl[1:3, 1:3]'
    Hfull[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[1:p, 1:6] .= Hfull[mask, :]

    measurement_update!(c.δx, c.Σ, pred, view(c.H, 1:p, :), Σ_pred, c.ws)
    pred_norm_full, pred_full = zeros(4), zeros(4)
    Σ_pred_norm_full, Σ_pred_full = zeros(4, 4), zeros(4, 4)
    pred_norm_full[mask] = pred_norm
    pred_full[mask] = pred
    Σ_pred_norm_full[mask, mask] = Σ_pred_norm
    Σ_pred_full[mask, mask] = Σ_pred
    return pred_full, diag(Σ_pred_full), pred_norm_full, diag(Σ_pred_norm_full)
end

function relinearize!(c::JointHsgpEstimator)
    c.pos[:, c.i] += c.δx[1:3]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6]), c.quat[:, c.i])
    c.β += c.δx[7:end]
    c.δx .= 0.0

    # c.Σ[1:6, 7:end] .= 0.0
    # c.Σ[7:end, 1:6] .= 0.0
end

function get_β_Σβ(c::JointHsgpEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    m = c.params.m
    p = c.p
    mask = c.correction_mask

    β_full = zeros(Float64, 4 * m)
    Σβ_full = zeros(Float64, 4 * m, 4 * m)

    for j in 1:p
        orig_idx = mask[j]
        β_full[_full_range(orig_idx, m)] = c.β[((j-1)*m+1):(j*m)]
    end

    Σβ_internal = c.Σ[7:end, 7:end]
    for j1 in 1:p, j2 in 1:p
        r1 = _full_range(mask[j1], m)
        r2 = _full_range(mask[j2], m)
        Σβ_full[r1, r2] = Σβ_internal[((j1-1)*m+1):(j1*m), ((j2-1)*m+1):(j2*m)]
    end

    return β_full, Σβ_full
end