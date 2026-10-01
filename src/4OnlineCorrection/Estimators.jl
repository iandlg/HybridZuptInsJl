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
        feature[i] = wrap_pi(feature[i])
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

function posyaw_measurement_update!(c::AbstractEstimator; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    error("posyaw_measurement_update! not implemented for $(typeof(c))")
end

function relinearize!(c::AbstractEstimator; kwarg...)
    error("relinearize! not implemented for $(typeof(c))")
end

function get_model(c::AbstractEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    error("get_model not implemented for $(typeof(c))")
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
    Δθ3_gt = wrap_pi(Δθ3_gt)

    gt_stride, Σ_gt_stride, _ = stride_local(ref_frame;
        R_wb=R_wb_gt[1],
        ΔpΔθ3=[Δp_gt; Δθ3_gt],
        Σ_ΔpΔθ3=Σ_ΔpΔθ3_gt
    )

    # Compute estimated stride from INS in local frame
    Δθ3 = matrix_to_euler(R_wb[2])[3] -
          matrix_to_euler(R_wb[1])[3]
    Δθ3 = wrap_pi(Δθ3)
    ins_stride, Σ_ins_stride, R_aug_wl = stride_local(ref_frame;
        R_wb=R_wb[1],
        ΔpΔθ3=[Δp; Δθ3],
        Σ_ΔpΔθ3=Σ_ΔpΔθ3
    )

    # Compute stride error
    stride_err = gt_stride - ins_stride
    stride_err[4] = wrap_pi(stride_err[4])

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

function BaseEstimator(N::Int; kwargs...)::BaseEstimator
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

"""`σ_stride` is the per-stride process noise (`InsConfig.sigma_stride`), added in the
stride's heading frame on top of the INS's own `Σpq`, which alone under-states the
per-stride heading error (notes/022)."""
function dynamic_update!(c::BaseEstimator; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64},
    σ_stride::AbstractVector{Float64}, kwargs...)
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

    R_ψ = stride_local(HEADING; R_wb=R_prev, ΔpΔθ3=zeros(4))[3][1:3, 1:3]
    B = [R_ψ zeros(3); zeros(3, 3) [0.0, 0.0, 1.0]]
    c.Σ .+= B * Diagonal(σ_stride .^ 2) * B'
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
        vcat(curr_pos .- c.pos[:, c.i], wrap_pi(curr_θ3 - θ3_estim)),
        c.H[:, :, c.i],
        Σy
    )
end

function relinearize!(c::BaseEstimator)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.δx[:, c.i] .= 0.0
end

function get_model(c::BaseEstimator)::Optional{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    return nothing
end
