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

function ∂feature_∂δpδθ(feature_type::FeatureType; σ_output::AbstractVector{T}, R_aug_wl::AbstractMatrix{T}, q_curr::AbstractVector{T}) where T<:Real
    feature_deriv_fun = Dict{FeatureType,Function}(
        THREED_STEP => () -> ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3]),
        TWOD_STEP_DT => () -> [
            ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3])[1:2, :];
            ∂featureΔT_∂δpδθ(T)'
        ],
        THREED_STEP_DT => () -> [
            ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3]);
            ∂featureΔT_∂δpδθ(T)'
        ],
        AUG_STEP => () -> [
            ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3]);
            ∂featureΔθ3_∂δpδθ(q_curr, σ_output[4])'
        ],
        TWOD_STEP_YAW => () -> [
            ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3])[1:2, :];
            ∂featureΔθ3_∂δpδθ(q_curr, σ_output[4])'
        ],
        TWOD_STEP_DT_YAW => () -> [
            ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3])[1:2, :];
            ∂featureΔT_∂δpδθ(T)';
            ∂featureΔθ3_∂δpδθ(q_curr, σ_output[4])'
        ],
        THREED_STEP_DT_YAW => () -> [
            ∂feature𝑥𝑦𝑧_∂δpδθ(R_aug_wl[1:3, 1:3], σ_output[1:3]);
            ∂featureΔT_∂δpδθ(T)';
            ∂featureΔθ3_∂δpδθ(q_curr, σ_output[4])'
        ]
    )[feature_type]
    return feature_deriv_fun()
end

function normalize_feature!(
    feature_type::FeatureType;
    feature::AbstractVector{Float64},
    Σ_feature::Union{Nothing,AbstractMatrix{Float64}}=nothing,
    input_stats::Vector{Vector{Float64}}
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

    # Covariance normalization
    Σ_feature_norm = if isnothing(Σ_feature)
        nothing
    else
        D = Diagonal(1.0 ./ σ)
        D * Σ_feature * D
    end

    return feature, Σ_feature_norm
end

abstract type AbstractCorrector end

function initialize_corrector!(
    c::AbstractCorrector;
    t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwarg...)
    error("initialize_corrector! not implemented for $(typeof(c))")
end

function dynamic_update!(c::AbstractCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwarg...)
    error("dynamic_update! not implemented for $(typeof(c))")
end

function stride_measurement_update!(c::AbstractCorrector; feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    ins_stride::AbstractVector{Float64}, Σ_ins_stride::AbstractMatrix{Float64},
    kwargs...
)
    error("stride_measurement_update! not implemented for $(typeof(c))")
end

function posyaw_measurement_update!(c::AbstractCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    error("posyaw_measurement_update! not implemented for $(typeof(c))")
end

function learned_measurement_update!(c::AbstractCorrector;
    kwargs...
)::NTuple{2,Optional{AbstractVector{Float64}}}
    error("learned_measurement_update! not implemented for $(typeof(c))")
end

function relinearize!(c::AbstractCorrector; kwarg...)
    error("relinearize! not implemented for $(typeof(c))")
end

# Accessors
function get_time(c::AbstractCorrector)::AbstractVector{Float64}
    return c.t[1:c.i]
end

function get_pos(c::AbstractCorrector)::AbstractMatrix{Float64}
    return c.pos[:, 1:c.i]
end

function get_quat(c::AbstractCorrector)::AbstractMatrix{Float64}
    return c.quat[:, 1:c.i]
end

function get_trajectory(c::AbstractCorrector)::Trajectory
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
mutable struct DefaultCorrector <: AbstractCorrector
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

function DefaultCorrector(N::Int)::DefaultCorrector
    @assert N > 1 "Invalid number of "
    return DefaultCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 6, N),
        zeros(Float64, 6, 6), zeros(Float64, 6, 6), zeros(Float64, 4, 6, N), 1, zeros(Float64, 6, 6))
end

function initialize_corrector!(c::DefaultCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ .= Σpq_init
    c.G .= Matrix{Float64}(I, 6, 6)
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::DefaultCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
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

function stride_measurement_update!(c::DefaultCorrector;
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
    return
end

function posyaw_measurement_update!(c::DefaultCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
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

function learned_measurement_update!(c::DefaultCorrector; kwargs...)::NTuple{2,Optional{AbstractVector{Float64}}}
    # Do nothing
    return nothing, nothing
end

function relinearize!(c::DefaultCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.δx[:, c.i] .= 0.0
end

mutable struct StaticCorrector <: AbstractCorrector
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
    stride_err_mean::AbstractMatrix{Float64}
end

function StaticCorrector(N::Int)::StaticCorrector
    @assert N > 1 "Invalid number of "
    return StaticCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 10, N),
        zeros(Float64, 10, 10), zeros(Float64, 10, 6), zeros(Float64, 4, 10, N), 1, zeros(Float64, 10, 10),
        zeros(Float64, 4, N)
    )
end

function initialize_corrector!(c::StaticCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.stride_err_mean[:, 1] .= 0.0
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6] .= Σpq_init
    c.Σ[7:10, 7:10] .= Matrix{Float64}(I, 4, 4) * 1e2
    c.Σ[10, 10] *= 1e2
    c.G .= Matrix{Float64}(I, 10, 6)
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::StaticCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    c.stride_err_mean[:, c.i] = c.stride_err_mean[:, c.i-1]

    c.G .= 0.0
    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F .= 0.0
    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)
    c.F[7:10, 7:10] = Matrix{Float64}(I, 4, 4)

    # Covariance update
    c.δx[:, c.i] .= 0.0
    c.Σ .= c.F * c.Σ * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::StaticCorrector;
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)
    c.H[1:3, 1:3, c.i] = R_aug_wl[1:3, 1:3]' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # c.H[4, 4:6, c.i] = ∂θ3_∂δθ_left(c.quat[:, c.i])
    c.H[4, 6, c.i] = 1.0
    c.H[1:4, 7:10, c.i] = Matrix{Float64}(I, 4, 4)

    stride_err -= c.stride_err_mean[:, c.i]
    stride_err[4] = atan(sin(stride_err[4]), cos(stride_err[4]))

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        stride_err,
        c.H[:, :, c.i],
        Σ_err
    )
    return
end

function posyaw_measurement_update!(c::StaticCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # c.H[4, 4:6, c.i] = ∂θ3_∂δθ_left(c.quat[:, c.i])
    c.H[1:4, 7:10, c.i] .= 0.0
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::StaticCorrector;
    R_aug_wl, kwargs...)::NTuple{2,Optional{AbstractVector{Float64}}}
    c.H[1:3, 1:3, c.i] = R_aug_wl[1:3, 1:3]'
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # c.H[4, 4:6, c.i] = ∂θ3_∂δθ_left(c.quat[:, c.i])
    c.H[1:4, 7:10, c.i] .= 0.0

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        c.stride_err_mean[:, c.i],
        c.H[:, :, c.i],
        Diagonal(fill(1e-5, 4) .^ 2)
    )
    return c.stride_err_mean[:, c.i], fill(1e-5, 4) .^ 2
end

function relinearize!(c::StaticCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.stride_err_mean[:, c.i] += c.δx[7:10, c.i]
    c.δx[:, c.i] .= 0.0

    c.Σ[1:6, 7:10] .= 0.0
    c.Σ[7:10, 1:6] .= 0.0
end

mutable struct SplitHybridCorrector <: AbstractCorrector
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
end

function SplitHybridCorrector(N::Int, params::HsgpParameters, feature_type::FeatureType)::SplitHybridCorrector
    @assert N > 1 "Invalid number of "
    return SplitHybridCorrector(
        zeros(Float64, N),
        zeros(Float64, 3, N),
        zeros(Float64, 4, N),
        zeros(Float64, 6, N),
        zeros(Float64, 6, 6),
        zeros(Float64, 6, 6),
        zeros(Float64, 4, 6, N),
        1,
        zeros(Float64, 6, 6),
        params,
        zeros(Float64, params.m * 4),
        zeros(Float64, params.m * 4, params.m * 4),
        zeros(Float64, 4, params.d),
        zeros(Float64, 4, params.m * 4),
        zeros(Float64, params.m, params.d),
        zeros(Float64, 4)
    )
end

function initialize_corrector!(c::SplitHybridCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6] = Σpq_init
    c.G[:, :, 1] = Matrix{Float64}(I, 6, 6)

    # Initialize HSGP
    c.per_dim_eigvals = calc_eigenvalues(c.params.LL, c.params.m, c.params.d)
    psd = zeros(Float64, 4 * c.params.m)

    for (idx, field) in enumerate(fieldnames(SeHyperparams))
        psd[((idx-1)*c.params.m+1):(idx*c.params.m)] = power_spectral_density(
            sqrt.(c.per_dim_eigvals),
            getfield(c.params.hp, field)[2],
            getfield(c.params.hp, field)[3]
        )
    end

    for (idx, field) in enumerate(fieldnames(SeHyperparams))
        c.σ_n[idx] = getfield(c.params.hp, field)[1]
    end

    output_names = ["pos_1", "pos_2", "pos_3", "yaw"]
    c.β .= 0.0
    c.Σβ = Diagonal(psd)
    c.∂y∂z .= 0.0
    c.Φ .= 0.0
    c.G .= Matrix{Float64}(I, 6, 6)
    c.F .= 0.0
    c.i = 1
end

function dynamic_update!(c::SplitHybridCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
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

function stride_measurement_update!(c::SplitHybridCorrector;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64},
    kwargs...)

    # Normalise target and target covariance
    stride_err -= c.params.output_stats[1] # remove output mean
    stride_err ./= c.params.output_stats[2] # normalize with output std deviation
    Σ_err = Diagonal(1 ./ c.params.output_stats[2]) * (Σ_err) * Diagonal(1 ./ c.params.output_stats[2])

    # Compute input feature and normalize
    normalize_feature!(feature_type; feature=feature, Σ_feature=Σ_feature, c.params.input_stats)

    # ------------ Construct Feature Covariance ---------
    for output_d in axes(c.∂y∂z, 1)
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d] = dot(calc_eigenvectors_dx(
                    reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
                ), c.β[((output_d-1)*c.params.m+1):(output_d*c.params.m)])
        end
    end
    Σ_err += c.∂y∂z * Σ_feature * c.∂y∂z'

    # ---------- Construct measurement matrix H_update --------------
    kron!(c.Φ, I(4), calc_eigenvectors(
        reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals)
    )

    α = tr(Diagonal(c.σ_n .^ 2)) / tr(Σ_err)
    c.β, c.Σβ = measurement_update(
        c.β, c.Σβ, stride_err, c.Φ, Diagonal(c.σ_n .^ 2) + α * Σ_err # Diagonal(noise_vect .^ 2)
    )

    return
end


function posyaw_measurement_update!(c::SplitHybridCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        [curr_pos .- c.pos[:, c.i]; atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))],
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::SplitHybridCorrector;
    feature_type::FeatureType,
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)::NTuple{2,Optional{AbstractVector{Float64}}}
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]

    # Normalise estimated feature and feature covariance
    normalize_feature!(feature_type; feature=feature, Σ_feature=Σ_feature, c.params.input_stats)

    for output_d in axes(c.∂y∂z, 1)
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d:input_d] = calc_eigenvectors_dx(
                reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
            ) * c.β[((output_d-1)*c.params.m+1):(output_d*c.params.m)]
        end
    end

    kron!(c.Φ, I(4), calc_eigenvectors(reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals))

    # Compute prediction and prediction covariance
    pred = c.Φ * c.β
    Σ_pred = c.Φ * c.Σβ * c.Φ' + c.∂y∂z * Σ_feature * c.∂y∂z' # Predictive + Input uncertainty

    # Denormalise prediction and prediction covariance
    pred = pred .* c.params.output_stats[2] .+ c.params.output_stats[1]
    pred[4] = atan(sin(pred[4]), cos(pred[4]))
    Σ_pred = Diagonal(c.params.output_stats[2]) * Σ_pred * Diagonal(c.params.output_stats[2])

    # Rotate to body frame 
    pred = R_aug_wl * pred
    Σ_pred = R_aug_wl * Σ_pred * R_aug_wl'

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        pred,
        c.H[:, :, c.i],
        Σ_pred
    )
    return pred, diag(Σ_pred)
end

function relinearize!(c::SplitHybridCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.δx[:, c.i] .= 0.0
end


mutable struct SlamCorrector <: AbstractCorrector
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
end

function SlamCorrector(N::Int, params::HsgpParameters)::SlamCorrector
    @assert N > 1 "Invalid number of prealocations"
    return SlamCorrector(
        Vector{Float64}(undef, N),                                  # t
        Matrix{Float64}(undef, 3, N),                               # pos
        Matrix{Float64}(undef, 4, N),                               # quat
        Vector{Float64}(undef, 6 + 4 * params.m),                   # δx
        Matrix{Float64}(undef, 6 + 4 * params.m, 6 + 4 * params.m), # Σ
        Matrix{Float64}(undef, 6, 6),                               # G
        Matrix{Float64}(undef, 4, 6 + 4 * params.m),                # H
        1,                                                          # i
        Matrix{Float64}(undef, 6, 6),                               # F
        params,                                                     # params
        Vector{Float64}(undef, params.m * 4),                       # β
        Matrix{Float64}(undef, 4, params.d),                        # ∂y∂z
        Matrix{Float64}(undef, 1, params.m),                           # ϕ
        Matrix{Float64}(undef, params.m, params.d),                 # per_dim_eigvals
    )
end

function initialize_corrector!(c::SlamCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.i = 1
    c.t[1] = t
    # Initialize nominal states
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.β .= 0.0
    # Initialize error state
    c.δx .= 0.0
    # Initialize HSGP
    c.per_dim_eigvals = calc_eigenvalues(c.params.LL, c.params.m, c.params.d)

    psd = zeros(Float64, 4 * c.params.m)
    for (idx, field) in enumerate(fieldnames(SeHyperparams))
        psd[((idx-1)*c.params.m+1):(idx*c.params.m)] = power_spectral_density(
            sqrt.(c.per_dim_eigvals),
            getfield(c.params.hp, field)[2],
            getfield(c.params.hp, field)[3]
        )
    end
    # Initialize state covariance
    c.Σ .= 0.0
    c.Σ[1:6, 1:6] = Σpq_init
    c.Σ[7:end, 7:end] = Diagonal(psd)

    c.∂y∂z .= 0.0
    c.ϕ .= 0.0
    c.G .= 0.0
    c.H .= 0.0
    c.F .= 0.0

end

function dynamic_update!(c::SlamCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    # c.β = c.β

    c.G[1:3, 1:3] = R_prev
    c.G[4:6, 4:6] = quat_to_matrix(c.quat[:, c.i])

    c.F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6] .= -skew(R_prev * Δp)

    # Covariance update
    c.δx .= 0.0
    c.Σ[1:6, 1:6] = c.F * c.Σ[1:6, 1:6] * c.F' + c.G * Σpq * c.G'
end

function stride_measurement_update!(c::SlamCorrector;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)

    # ------ Construct measurement --------
    # Normalise input feature
    normalize_feature!(feature_type; feature=feature, Σ_feature=Σ_feature, c.params.input_stats)

    # Compute eigenvector
    c.ϕ = calc_eigenvectors(reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals)

    # Compute denormalised prediction
    for i in 1:4
        stride_err[i] -= c.params.output_stats[2][i] * dot(c.ϕ, c.β[((i-1)*c.params.m+1):(i*c.params.m)]) + c.params.output_stats[1][i]
    end
    # Σ_err stays the same ie stride error measurement covariance
    # ---> nominal measurement is done

    # ------- Construct measurement matrix 𝐇 = [Hp, Hθ, Hβ] ∈ R^{4 × (6 + 4 m)}
    for output_d in axes(c.∂y∂z, 1)
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d] = dot(calc_eigenvectors_dx(
                    reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
                ), c.β[((output_d-1)*c.params.m+1):(output_d*c.params.m)])
        end
    end
    c.∂y∂z = Diagonal(c.params.output_stats[2]) * c.∂y∂z # denormalize prediction

    # Construct Hβ
    kron!(view(c.H, 1:4, 7:(6+4*c.params.m)), Matrix{Float64}(I, 4, 4), c.ϕ)

    # Construct Hp
    c.H[:, 1:6] .= 0.0
    c.H[1:3, 1:3] = R_aug_wl[1:3, 1:3]'

    # Construct Hθ
    c.H[4, 4:6] = ∂θ3_∂δθ_left(c.quat[:, c.i])

    # Add GP contribution to δp and δθ
    c.H[:, 1:6] += c.∂y∂z * ∂feature_∂δpδθ(feature_type; σ_output=c.params.output_stats[2], R_aug_wl=R_aug_wl, q_curr=c.quat[:, c.i])

    c.δx, c.Σ = measurement_update(
        c.δx, c.Σ,
        stride_err,
        c.H,
        Σ_err
    )

    return
end


function posyaw_measurement_update!(c::SlamCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6] = [0.0, 0.0, 1.0]
    c.H[:, 7:end] .= 0.0

    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx, c.Σ = measurement_update(
        c.δx, c.Σ,
        [curr_pos .- c.pos[:, c.i]; atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))],
        c.H,
        Σy
    )
end

function learned_measurement_update!(c::SlamCorrector;
    feature_type::FeatureType,
    feature::AbstractVector{Float64}, Σ_feature::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)::NTuple{2,Optional{AbstractVector{Float64}}}
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]

    # Normalise estimated feature and feature covariance
    normalize_feature!(feature_type; feature=feature, Σ_feature=Σ_feature, c.params.input_stats)

    for output_d in axes(c.∂y∂z, 1)
        for input_d in axes(c.∂y∂z, 2)
            c.∂y∂z[output_d, input_d:input_d] = calc_eigenvectors_dx(
                reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals, input_d
            ) * c.β[((output_d-1)*c.params.m+1):(output_d*c.params.m)]
        end
    end

    kron!(c.Φ, I(4), calc_eigenvectors(reshape(feature, 1, c.params.d), c.params.LL, c.per_dim_eigvals))

    # Compute prediction and prediction covariance
    pred = c.Φ * c.β
    Σ_pred = c.Φ * c.Σβ * c.Φ' + c.∂y∂z * Σ_feature * c.∂y∂z' # Predictive + Input uncertainty

    # Denormalise prediction and prediction covariance
    pred = pred .* c.params.output_stats[2] .+ c.params.output_stats[1]
    pred[4] = atan(sin(pred[4]), cos(pred[4]))
    Σ_pred = Diagonal(c.params.output_stats[2]) * Σ_pred * Diagonal(c.params.output_stats[2])

    # Rotate to body frame 
    pred = R_aug_wl * pred
    Σ_pred = R_aug_wl * Σ_pred * R_aug_wl'

    c.δx[:, c.i], c.Σ = measurement_update(
        c.δx[:, c.i], c.Σ,
        pred,
        c.H[:, :, c.i],
        Σ_pred
    )
    return pred, diag(Σ_pred)
end

function relinearize!(c::SlamCorrector)
    c.pos[:, c.i] += c.δx[1:3]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6]), c.quat[:, c.i])
    c.β += c.δx[7:end]
    c.δx .= 0.0

    c.Σ[1:6, 7:end] .= 0.0
    c.Σ[7:end, 1:6] .= 0.0
end