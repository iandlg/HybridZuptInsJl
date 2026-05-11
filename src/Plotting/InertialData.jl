"""
    plot_inertial_data(id::InertialData; kwargs...)

Plot accelerometer and gyroscope signals from an `InertialData` object.

# Arguments
- `id`: InertialData object with fields `t` (time vector) and `u` (6xN matrix).
- `kwargs...`: Additional keyword arguments passed to `plot` (e.g., `title`, `xlabel`, `legend`).

# Returns
- A `Plots.Plot` object with two subplots.
"""
function plot_inertial_data(id::InertialData)
    t = id.t
    acc = accel(id)
    gyr = gyro(id)

    fig = Figure(size=(900, 600))

    # Accelerometer axes
    ax1 = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Acceleration (m/s²)", title="Accelerometer")
    lines!(ax1, t, acc[1, :], label="Acc X", color=:red)
    lines!(ax1, t, acc[2, :], label="Acc Y", color=:green)
    lines!(ax1, t, acc[3, :], label="Acc Z", color=:blue)
    axislegend(ax1)

    # Gyroscope axes
    ax2 = Axis(fig[2, 1], xlabel="Time (s)", ylabel="Angular rate (rad/s)", title="Gyroscope")
    lines!(ax2, t, gyr[1, :], label="Gyro X", color=:red)
    lines!(ax2, t, gyr[2, :], label="Gyro Y", color=:green)
    lines!(ax2, t, gyr[3, :], label="Gyro Z", color=:blue)
    axislegend(ax2)

    return fig
end

using GLMakie  # or CairoMakie, depending on your backend

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
        marker='x',
        color=:red,
        markersize=5,
        strokewidth=1
    )

    # Display the figure (optional, depends on environment)
    display(fig)

    return fig
end