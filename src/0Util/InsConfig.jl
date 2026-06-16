@enum ReferenceFrame begin
    BODY = 1
    HEADING = 2
    NAVIGATION = 3
end

@enum FeatureType begin
    THREED_STEP = 1
    TWOD_STEP_DT = 2
    THREED_STEP_DT = 3
    AUG_STEP = 4
    TWOD_STEP_YAW = 5
    TWOD_STEP_DT_YAW = 6
end

const FEATURE_DIMS = Dict{FeatureType,Int}(
    THREED_STEP => 3,
    TWOD_STEP_DT => 3,
    THREED_STEP_DT => 4,
    AUG_STEP => 4,
    TWOD_STEP_YAW => 3,
    TWOD_STEP_DT_YAW => 4
)

function string_to_enum(::Type{T}, s::AbstractString) where T<:Enum
    for v in instances(T)
        string(v) == s && return v
    end
    throw(ArgumentError("\"$s\" is not a valid $(T)"))
end

"""
    compute_gravity(latitude, altitude)

Compute local gravity magnitude using the WGS84 model.

# Arguments
- `latitude`: Latitude in degrees.
- `altitude`: Altitude in metres.

# Returns
- Local gravity magnitude in m/s².
"""
function compute_gravity(latitude::Real, altitude::Real)::Float64
    λ = deg2rad(latitude)
    γ = 9.780327 * (1 + 0.0053024 * sin(λ)^2 - 0.0000058 * sin(2λ)^2)
    g = γ - ((3.0877e-6) - (0.004e-6) * sin(λ)^2) * altitude + (0.072e-12) * altitude^2
    return g
end
mutable struct InsConfig
    # General
    altitude::Float64
    latitude::Float64
    Ts::Float64
    init_heading::Float64
    init_pos::NTuple{3,Float64}
    # Detector
    sigma_a::Float64
    sigma_g::Float64
    window_size::Int
    gamma::Float64
    # Process noise
    sigma_acc::NTuple{3,Float64}
    sigma_gyro::NTuple{3,Float64}
    # Measurement noise
    sigma_vel::NTuple{3,Float64}
    # Initial covariance
    sigma_initial_pos::NTuple{3,Float64}
    sigma_initial_vel::NTuple{3,Float64}
    sigma_initial_att::NTuple{3,Float64}
    # Gravity
    g::Union{Float64,Vector{Float64}}
    # ZUPT segmentation
    segmentation_thrsld::Float64
    calibration_distance_m::Float64
    # Hybrid
    sigma_groundtruth::NTuple{4,Float64}
end

function InsConfig(;
    altitude=100.0,
    latitude=58.0,
    Ts=1 / 100,
    init_heading=0.0,
    init_pos=(0.0, 0.0, 0.0),
    sigma_a=0.01,
    sigma_g=0.2 * π / 180,
    window_size=3,
    gamma=0.5e5,
    sigma_acc=(1.3, 1.3, 1.3),
    sigma_gyro=(0.1 * π / 180, 0.1 * π / 180, 0.1 * π / 180),
    sigma_vel=(0.1, 0.1, 0.1),
    sigma_initial_pos=(1e-5, 1e-5, 1e-5),
    sigma_initial_vel=(1e-5, 1e-5, 1e-5),
    sigma_initial_att=(100 * π / 180, 100 * π / 180, 0.1 * π / 180),
    g=nothing,
    segmentation_thrsld=0.1e-3,
    calibration_distance_m=1.55,
    sigma_groundtruth=(1e-2, 1e-2, 1e-2, 1e-3)
)
    g_val = isnothing(g) ? compute_gravity(latitude, altitude) : Float64(g)

    return InsConfig(
        altitude, latitude, Ts, init_heading, init_pos,
        sigma_a, sigma_g, window_size, gamma,
        sigma_acc, sigma_gyro, sigma_vel,
        sigma_initial_pos, sigma_initial_vel, sigma_initial_att,
        g_val, segmentation_thrsld, calibration_distance_m, sigma_groundtruth
    )
end

# --------------------------------------------------------------------------
# Convenience accessors that return mutable vectors from the tuple fields
# --------------------------------------------------------------------------
init_pos_array(cfg::InsConfig) = collect(cfg.init_pos)
sigma_acc_array(cfg::InsConfig) = collect(cfg.sigma_acc)
sigma_gyro_array(cfg::InsConfig) = collect(cfg.sigma_gyro)
sigma_vel_array(cfg::InsConfig) = collect(cfg.sigma_vel)
sigma_initial_pos_array(cfg::InsConfig) = collect(cfg.sigma_initial_pos)
sigma_initial_vel_array(cfg::InsConfig) = collect(cfg.sigma_initial_vel)
sigma_initial_att_array(cfg::InsConfig) = collect(cfg.sigma_initial_att)
sigma_groundtruth_array(cfg::InsConfig) = collect(cfg.sigma_groundtruth)