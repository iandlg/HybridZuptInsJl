function stride_heading(;
    R_wb::AbstractMatrix{T}, ΔpΔθ3::AbstractVector{T}, Σ_ΔpΔθ3::Union{Nothing,AbstractMatrix{Float64}}=nothing
)::Tuple{AbstractVector{T},Union{Nothing,AbstractMatrix{Float64}},AbstractMatrix{Float64}} where T<:Real
    temp = zeros(Float64, 4, 4)
    temp[4, 4] = matrix_to_euler(R_wb')[3]

    temp[1:3, 1:3] = euler_to_matrix(temp[2:4, 4])
    temp[4, 4] = 1.0
    return temp * ΔpΔθ3, isnothing(Σ_ΔpΔθ3) ? nothing : temp * Σ_ΔpΔθ3 * temp', temp'
end

function stride_body(;
    R_wb::AbstractMatrix{T}, ΔpΔθ3::AbstractVector{T}, Σ_ΔpΔθ3::Union{Nothing,AbstractMatrix{Float64}}=nothing
)::Tuple{AbstractVector{T},Union{Nothing,AbstractMatrix{Float64}},AbstractMatrix{Float64}} where T<:Real
    temp = zeros(Float64, 4, 4)
    temp[1:3, 1:3] = R_wb'
    temp[4, 4] = 1.0
    return temp * ΔpΔθ3, isnothing(Σ_ΔpΔθ3) ? nothing : temp * Σ_ΔpΔθ3 * temp', temp'
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
    stride_local_aug::AbstractVector{T}, Σ_stride_local_aug::Union{Nothing,AbstractMatrix{Float64}}=nothing,
    ΔT::T, ΔT_var::T=1e-8,
)::Tuple{AbstractVector{T},Union{Nothing,AbstractMatrix{Float64}}} where T<:Real
    feature_fun = Dict{FeatureType,Function}(
        THREED_STEP => () -> stride_local_aug[1:3],
        TWOD_STEP_DT => () -> [stride_local_aug[1:2]; ΔT],
        THREED_STEP_DT => () -> [stride_local_aug[1:3]; ΔT],
        AUG_STEP => () -> stride_local_aug,
        TWOD_STEP_YAW => () -> [stride_local_aug[1:2]; stride_local_aug[4]],
        TWOD_STEP_DT_YAW => () -> [stride_local_aug[1:3]; ΔT; stride_local_aug[4]]
    )[feature_type]

    feature_cov_fun = Dict{FeatureType,Function}(
        THREED_STEP => () -> Σ_stride_local_aug[1:3, 1:3],
        TWOD_STEP_DT => () -> [
            Σ_stride_local_aug[1:2, 1:2] zeros(Float64, (2, 1));
            zeros(Float64, (1, 2)) ΔT_var
        ],
        THREED_STEP_DT => () -> [
            Σ_stride_local_aug[1:3, 1:3] zeros(Float64, (3, 1));
            zeros(Float64, (1, 3)) ΔT_var
        ],
        AUG_STEP => () -> Σ_stride_local_aug,
        TWOD_STEP_YAW => () -> Σ_stride_local_aug[[1:2; 4], [1:2; 4]],
        TWOD_STEP_DT_YAW => () -> [
            Σ_stride_local_aug[1:2, 1:2] zeros(Float64, (2, 1)) Σ_stride_local_aug[1:2, 4:4];
            zeros(Float64, (1, 2)) ΔT_var 0.0;
            Σ_stride_local_aug[4:4, 1:2] 0.0 Σ_stride_local_aug[4, 4]
        ],
    )[feature_type]
    return feature_fun(), isnothing(Σ_stride_local_aug) ? nothing : feature_cov_fun()
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
)::NTuple{Optional{AbstractVector{Float64}}}
    error("stride_measurement_update! not implemented for $(typeof(c))")
end

function posyaw_measurement_update!(c::AbstractCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    error("posyaw_measurement_update! not implemented for $(typeof(c))")
end

function learned_measurement_update!(c::AbstractCorrector;
    kwargs...
)::NTuple{4,Optional{AbstractVector{Float64}}}
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
    Σ::AbstractArray{Float64,3}
    G::AbstractArray{Float64,3}
    H::AbstractArray{Float64,3}
    i::Int
end

function DefaultCorrector(N::Int)::DefaultCorrector
    @assert N > 1 "Invalid number of "
    return DefaultCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 6, N),
        zeros(Float64, 6, 6, N), repeat(Matrix{Float64}(I, 6, 6), 1, 1, N), zeros(Float64, 4, 6, N), 1)
end

function initialize_corrector!(c::DefaultCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[:, :, 1] = Σpq_init
    c.G[:, :, 1] = Matrix{Float64}(I, 6, 6)
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
    c.G[:, :, c.i] .= 0.0
    c.G[1:3, 1:3, c.i] = R_prev
    c.G[4:6, 4:6, c.i] = quat_to_matrix(c.quat[:, c.i])

    F = zeros(Float64, 6, 6)
    F[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
    F[4:6, 4:6] = Matrix{Float64}(I, 3, 3)
    F[1:3, 4:6] .= -skew(R_prev * Δp)

    # Covariance update
    c.δx[:, c.i] .= 0.0 # c.F[:, :, c.i] * c.δx[:, c.i-1]
    c.Σ[:, :, c.i] = F * c.Σ[:, :, c.i-1] * F' + c.G[:, :, c.i] * Σpq * c.G[:, :, c.i]'
end

function stride_measurement_update!(c::DefaultCorrector;
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...
)
    # c.H[1:3, 1:3, c.i] = R_aug_wl[1:3, 1:3]' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # # c.H[4, 4:6, c.i] = jacobian_∂θ3_∂δθ_left(c.quat[:, c.i])
    # c.H[4, 6, c.i] = 1.0

    # c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
    #     c.δx[:, c.i], c.Σ[:, :, c.i],
    #     stride_err,
    #     c.H[:, :, c.i],
    #     Σ_err
    # )
    return nothing, nothing
end

function posyaw_measurement_update!(c::DefaultCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::DefaultCorrector; kwargs...)
    # Do nothing
    return nothing, nothing, nothing, nothing
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
    Σ::AbstractArray{Float64,3}
    G::AbstractArray{Float64,3}
    H::AbstractArray{Float64,3}
    i::Int

    # Corrector specific Arguments
    stride_err_mean::AbstractMatrix{Float64}
end

function StaticCorrector(N::Int)::StaticCorrector
    @assert N > 1 "Invalid number of "
    return StaticCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 10, N),
        zeros(Float64, 10, 10, N), repeat(Matrix{Float64}(I, 10, 6), 1, 1, N), zeros(Float64, 4, 10, N), 1,
        zeros(Float64, 4, N)
    )
end

function initialize_corrector!(c::StaticCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6, 1] = Σpq_init
    c.Σ[7:10, 7:10, 1] = Matrix{Float64}(I, 4, 4) * 1e-1
    c.G[:, :, 1] = Matrix{Float64}(I, 10, 6)
    c.i = 1
end

function dynamic_update!(c::StaticCorrector; t::Float64, Δp::AbstractVector{Float64}, Δq::AbstractVector{Float64}, Σpq::AbstractMatrix{Float64}, kwargs...)
    c.i += 1
    c.t[c.i] = t

    R_prev = quat_to_matrix(c.quat[:, c.i-1])

    # Nominal update
    c.pos[:, c.i] = c.pos[:, c.i-1] + R_prev * Δp
    c.quat[:, c.i] = quat_multiply(c.quat[:, c.i-1], Δq)
    # c.β = c.β
    c.G[:, :, c.i] .= 0.0
    c.G[1:3, 1:3, c.i] = R_prev
    c.G[4:6, 4:6, c.i] = quat_to_matrix(c.quat[:, c.i])
    # c.G[7:10, :, c.i] = Matrix{Float64}(I, 4, 4)

    c.F[:, :, c.i] .= 0.0
    c.F[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6, c.i] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6, c.i] = -skew(R_prev * Δp)
    c.F[7:10, 7:10, c.i] = Matrix{Float64}(I, 4, 4)

    # Covariance update
    c.δx[:, c.i] .= 0.0 # c.F[:, :, c.i] * c.δx[:, c.i-1]
    c.Σ[:, :, c.i] = c.F[:, :, c.i] * c.Σ[:, :, c.i] * c.F[:, :, c.i]' + c.G[:, :, c.i] * Σpq * c.G[:, :, c.i]'

end

function stride_measurement_update!(c::StaticCorrector;
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64}, R_aug_wl::AbstractMatrix{Float64},
    kwargs...)
    c.H[1:3, 1:3, c.i] = R_aug_wl[1:3, 1:3]' # H[1:3, 1:3] = R_bw = R_wb^⊤
    # c.H[4, 4:6, c.i] = jacobian_∂θ3_∂δθ_left(c.quat[:, c.i])
    c.H[4, 6, c.i] = 1.0
    c.H[1:4, 7:10, c.i] = Matrix{Float64}(I, 4, 4)

    stride_err -= c.stride_err_mean[:, c.i]

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        stride_err,
        c.H[:, :, c.i],
        Σ_err
    )
    return nothing, nothing
end


function posyaw_measurement_update!(c::StaticCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    c.H[1:4, 7:10, c.i] .= 0.0
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        vcat(curr_pos .- c.pos[:, c.i], atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))),
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::StaticCorrector; ref_frame::ReferenceFrame, kwargs...)
    c.H[1:3, 1:3, c.i] = if ref_frame == HEADING
        # Extract heading angle and create heading-aligned rotation
        yaw = matrix_to_euler(quat_to_matrix(c.quat[:, c.i-1]))[3]
        euler_to_matrix([0.0, 0.0, yaw])'
    elseif ref_frame == BODY
        # Use body-to-world rotation transpose
        quat_to_matrix(c.quat[:, c.i-1])'
    else
        error("Unknown reference frame: $ref_frame")
    end

    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    c.H[1:4, 7:10, c.i] .= 0.0

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        c.stride_err_mean[:, c.i],
        c.H[:, :, c.i],
        c.Σ[7:10, 7:10, c.i]
    )
    return nothing, nothing, c.stride_err_mean[:, c.i], diag(c.Σ[7:10, 7:10, c.i])
end

function relinearize!(c::StaticCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.stride_err_mean[:, c.i] += c.δx[7:10, c.i]
    c.δx[:, c.i] .= 0.0
end

mutable struct SplitHybridCorrector <: AbstractCorrector
    t::AbstractVector{Float64}
    pos::AbstractMatrix{Float64}
    quat::AbstractMatrix{Float64}
    δx::AbstractMatrix{Float64}
    Σ::AbstractArray{Float64,3}
    G::AbstractArray{Float64,3}
    H::AbstractArray{Float64,3}
    i::Int

    # Corrector specific Arguments
    params::HsgpParameters
    β::AbstractVector{Float64}
    Σβ::AbstractMatrix{Float64}
    ∂y∂z::AbstractMatrix{Float64}
    Φ::AbstractMatrix{Float64}
    per_dim_eigvals::AbstractMatrix{Float64}
    d::Int
    σ_n::AbstractVector{Float64}
end

function SplitHybridCorrector(N::Int, params::HsgpParameters, feature_type::FeatureType)::SplitHybridCorrector
    @assert N > 1 "Invalid number of "
    return SplitHybridCorrector(
        zeros(Float64, N), zeros(Float64, 3, N), zeros(Float64, 4, N), zeros(Float64, 6, N),
        zeros(Float64, 6, 6, N), repeat(Matrix{Float64}(I, 6, 6), 1, 1, N), zeros(Float64, 4, 6, N), 1,
        params, zeros(Float64, params.m * 4), zeros(Float64, params.m * 4, params.m * 4), zeros(Float64, 4, FEATURE_DIMS[feature_type]),
        zeros(Float64, 4, params.m * 4), zeros(Float64, params.m, FEATURE_DIMS[feature_type]), FEATURE_DIMS[feature_type], zeros(Float64, 4)
    )
end

function initialize_corrector!(c::SplitHybridCorrector; t::Float64, pos_init::AbstractVector{Float64}, quat_init::AbstractVector{Float64}, Σpq_init::AbstractMatrix{Float64}, kwargs...)
    c.t[1] = t
    c.pos[:, 1] = pos_init
    c.quat[:, 1] = quat_init
    c.δx[:, 1] .= 0.0
    c.Σ[1:6, 1:6, 1] = Σpq_init
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
    c.G[:, :, c.i] .= 0.0
    c.G[1:3, 1:3, c.i] = R_prev
    c.G[4:6, 4:6, c.i] = quat_to_matrix(c.quat[:, c.i])

    c.F[:, :, c.i] .= 0.0
    c.F[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.F[4:6, 4:6, c.i] = Matrix{Float64}(I, 3, 3)
    c.F[1:3, 4:6, c.i] = -skew(R_prev * Δp)

    # Covariance update
    c.δx[:, c.i] .= 0.0 # c.F[:, :, c.i] * c.δx[:, c.i-1]
    c.Σ[:, :, c.i] = c.F[:, :, c.i] * c.Σ[:, :, c.i] * c.F[:, :, c.i]' + c.G[:, :, c.i] * Σpq * c.G[:, :, c.i]'
end

function stride_measurement_update!(c::SplitHybridCorrector;
    feature_type::FeatureType,
    stride_err::AbstractVector{Float64}, Σ_err::AbstractMatrix{Float64},
    ins_stride::AbstractVector{Float64}, Σ_ins_stride::AbstractMatrix{Float64},
    kwargs...)

    # Normalise target and target covariance
    stride_err -= c.params.output_stats[1] # remove output mean
    stride_err ./= c.params.output_stats[2] # normalize with output std deviation
    Σ_err = Diagonal(1 ./ c.params.output_stats[2]) * (Σ_err) * Diagonal(1 ./ c.params.output_stats[2])

    # Compute input feature and normalize
    feature, Σ_feature = compute_feature(feature_type;
        stride_local_aug=ins_stride, Σ_stride_local_aug=Σ_ins_stride, ΔT=c.t[c.i] - c.t[c.i-1])
    feat_denorm, Σ_feat_denorm = copy(feature), copy(Σ_feature)
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

    return feat_denorm, diag(Σ_feat_denorm)
end


function posyaw_measurement_update!(c::SplitHybridCorrector; curr_pos::AbstractVector{Float64}, curr_θ3::Float64, Σy::AbstractMatrix{Float64}, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    θ3_estim = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3]
    # c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]
    # @info "Measurement matrix H" c.H[:, :, c.i]
    # @info "Covariance matrix before meas upd" c.Σ[:, :, c.i]

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        [curr_pos .- c.pos[:, c.i]; atan(sin(curr_θ3 - θ3_estim), cos(curr_θ3 - θ3_estim))],
        c.H[:, :, c.i],
        Σy
    )
end

function learned_measurement_update!(c::SplitHybridCorrector;
    ref_frame::ReferenceFrame, feature_type::FeatureType, kwargs...)
    c.H[1:3, 1:3, c.i] = Matrix{Float64}(I, 3, 3)
    c.H[4, 4:6, c.i] = [0.0, 0.0, 1.0]

    Δθ3 = matrix_to_euler(quat_to_matrix(c.quat[:, c.i]))[3] - matrix_to_euler(quat_to_matrix(c.quat[:, c.i-1]))[3]
    Δθ3 = atan(sin(Δθ3), cos(Δθ3))

    stride_local_aug, Σ_ins_stride, R_aug_wl = stride_local(ref_frame;
        R_wb=c.H[1:3, 1:3, c.i]',
        ΔpΔθ3=[c.pos[:, c.i] - c.pos[:, c.i-1]; Δθ3],
        Σ_ΔpΔθ3=c.Σ[[1:3; 6], [1:3; 6], c.i]
    )

    feature, Σ_feature = compute_feature(feature_type;
        stride_local_aug=stride_local_aug, Σ_stride_local_aug=Σ_ins_stride, ΔT=c.t[c.i] - c.t[c.i-1])
    feature_denorm, Σ_feature_denorm = copy(feature), copy(Σ_feature)
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

    c.δx[:, c.i], c.Σ[:, :, c.i] = measurement_update(
        c.δx[:, c.i], c.Σ[:, :, c.i],
        pred,
        c.H[:, :, c.i],
        Σ_pred
    )
    return feature_denorm, diag(Σ_feature_denorm), pred, diag(Σ_pred)
end

function relinearize!(c::SplitHybridCorrector)
    c.pos[:, c.i] += c.δx[1:3, c.i]
    c.quat[:, c.i] = quat_multiply(quat_exp(c.δx[4:6, c.i]), c.quat[:, c.i])
    c.δx[:, c.i] .= 0.0
end