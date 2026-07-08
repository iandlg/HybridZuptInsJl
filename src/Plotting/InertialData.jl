# Generic optional type alias (if not already defined)
"""
    plot_inertial_data(t, ω, α; start=1, stop=nothing)

Core plotting function for inertial data. Creates a two‑panel figure:
  - Top: gyroscope (`ω`) angular rates.
  - Bottom: accelerometer (`α`) specific force.

# Arguments
- `t`: Time vector.
- `ω`: Gyroscope data (3×N matrix) or `nothing` – skipped if `nothing`.
- `α`: Accelerometer data (3×N matrix) or `nothing` – skipped if `nothing`.
- `start`, `stop`: indices to select a time window (1‑based).
"""
function plot_inertial_data(
    t::AbstractVector{T},
    ω::Optional{AbstractMatrix{T}}=nothing,
    α::Optional{AbstractMatrix{T}}=nothing;
    start::Int=1,
    stop::Union{Int,Nothing}=nothing
) where T<:Real
    N = length(t)
    start = min(max(start, 1), N)
    stop = isnothing(stop) ? N : min(stop, N)

    t_window = t[start:stop]

    fig = Figure(size=(1000, 600))

    # Gyroscope (top)
    if ω !== nothing
        gyr = ω[:, start:stop]
        ax_gyr = Axis(fig[1, 1],
            xlabel="Time (s)",
            ylabel="Angular rate (rad/s)",
            title="Angular Velocity",
            titlesize=24)
        lines!(ax_gyr, t_window, gyr[1, :], label="Gyro X", color=:red)
        lines!(ax_gyr, t_window, gyr[2, :], label="Gyro Y", color=:green)
        lines!(ax_gyr, t_window, gyr[3, :], label="Gyro Z", color=:blue)
        axislegend(ax_gyr)
    end

    # Accelerometer (bottom)
    if α !== nothing
        acc = α[:, start:stop]
        ax_acc = Axis(fig[2, 1],
            xlabel="Time (s)",
            ylabel="Acceleration (m/s²)",
            title="External Specific Force",
            titlesize=24)
        lines!(ax_acc, t_window, acc[1, :], label="Acc X", color=:red)
        lines!(ax_acc, t_window, acc[2, :], label="Acc Y", color=:green)
        lines!(ax_acc, t_window, acc[3, :], label="Acc Z", color=:blue)
        axislegend(ax_acc)
    end

    return fig
end

"""
    plot_inertial_data(id::InertialData; start=1, stop=nothing)

Convenience method that extracts time, gyroscope (rows 4–6) and
accelerometer (rows 1–3) from an `InertialData` object and calls
the generic `plot_inertial_data(t, ω, α)` method.
"""
function plot_inertial_data(id::InertialData; start=1, stop=nothing)
    t = id.t
    ω = id.u[4:6, :]   # gyroscope
    α = id.u[1:3, :]   # accelerometer
    return plot_inertial_data(t, ω, α; start=start, stop=stop)
end

function plot_inertialdata_and_stepsegm(inertial::InertialData, segs::Vector{Int})
    # Create figure and axis with grid
    fig = Figure()
    ax = Axis(fig[1, 1];
        xgridvisible=true,
        ygridvisible=true,
        xlabel="Time",
        ylabel="Squared sum of u[1:3]"
    )

    # Prepare data for line: sum of squares of first three rows over time
    # inertial.u is expected to be a matrix (n_rows × n_samples)
    u_sub = inertial.u[1:3, :]          # first three rows
    y_line = vec(sum(u_sub .^ 2, dims=1))  # sum along rows, then flatten to vector
    t = inertial.t

    # Plot line (drawn after scatter to have it on top, mimicking zorder=2 > 1)
    lines!(ax, t, y_line; linewidth=0.5, color=:blue)

    # Prepare scatter data: x = time at segments, y = constant 100
    x_scatter = t[segs]
    y_scatter = fill(100.0, length(segs))

    # Add scatter (drawn first, so line will cover it if overlapping)
    scatter!(ax, x_scatter, y_scatter;
        marker=('x'),
        color=:red,
        markersize=5,
        strokewidth=1
    )

    # Display the figure (optional, depends on environment)
    display(fig)

    return fig
end