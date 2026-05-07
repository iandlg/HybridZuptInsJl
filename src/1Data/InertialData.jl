struct InertialData <: AbstractTimeSeries
    t::Vector{Float64}    # (N,)
    u::Matrix{Float64}    # (6, N)  [accel; gyro]

    function InertialData(t, u)
        N = length(t)
        all(diff(t) .> 0) || throw(ArgumentError("t must be strictly increasing"))
        size(u) == (6, N) || throw(DimensionMismatch("u must be (6, $N)"))
        new(t, u)
    end
end

Base.getindex(id::InertialData, mask) =
    InertialData(id.t[mask], id.u[:, mask])

function InertialData(path::AbstractString)
    df = CSV.read(path, DataFrame)
    t = df[:, 3] |> Vector{Float64}
    accel = Matrix(df[:, 4:6])'       # (3, N)
    gyro = Matrix(df[:, 7:9])'       # (3, N)
    return InertialData(t, [accel; gyro])
end

InertialData(dir::AbstractString, num::Int) =
    InertialData(joinpath(dir, "$(num)_IMURaw.csv"))

# Convenience accessors
accel(id::InertialData) = id.u[1:3, :]
gyro(id::InertialData) = id.u[4:6, :]