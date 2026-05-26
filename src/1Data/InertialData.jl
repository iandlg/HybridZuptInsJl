@enum DataDirectory begin
    ANG = 1
    MT = 2
end

const data_directories = Dict{DataDirectory,String}(
    ANG => "data/angermann_high_precision",
    MT => "data/mti-100-recordings"
)

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
    df = CSV.read(path, DataFrame, comment="//")
    rows_to_remove = ["PacketCounter", "", "SystemTimestamp", "Column2"] # add more if needed
    cols_to_drop = intersect(names(df), (rows_to_remove))
    select!(df, Not(cols_to_drop))

    # Now the collumns should be t, acc_x, acc_y, acc_z, rot_x, rot_y, rot_z ....
    t = df[:, 1] |> Vector{Float64}
    accel = Matrix(df[:, 2:4])'       # (3, N)
    gyro = Matrix(df[:, 5:7])'       # (3, N)
    return InertialData(t, [accel; gyro])
end

InertialData(dir::AbstractString, num::Int) =
    InertialData(joinpath(dir, "$(num)_IMURaw.csv"))


function InertialData(dir::DataDirectory, trial_id::Int)
    path = joinpath(data_directories[dir], "$(trial_id)_IMURaw.csv")
    df = CSV.read(path, DataFrame, comment="//")
    rows_to_remove = ["PacketCounter", "", "SystemTimestamp", "Column2"] # add more if needed
    cols_to_drop = intersect(names(df), (rows_to_remove))
    select!(df, Not(cols_to_drop))

    to_seconds = Dict(
        ANG => 1.0,
        MT => 1e-5
    )[dir]

    # Now the collumns should be t, acc_x, acc_y, acc_z, rot_x, rot_y, rot_z ....
    t = df[:, 1] |> Vector{Float64}
    accel = Matrix(df[:, 2:4])'       # (3, N)
    gyro = Matrix(df[:, 5:7])'       # (3, N)
    return InertialData(t .* to_seconds, [accel; gyro])
end