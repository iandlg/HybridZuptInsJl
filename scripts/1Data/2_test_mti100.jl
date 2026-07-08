# scripts/main.jl
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
import DataFrames
import CSV

imu_ang = HybridZuptInsJl.InertialData("data/angermann_high_precision/15_IMURaw.csv")
imu_mti = HybridZuptInsJl.InertialData("data/mti-100-recordings", 6)
imu_ang2 = HybridZuptInsJl.InertialData("data/angermann_v2", 5)
imu_dcsc = HybridZuptInsJl.InertialData("data/dcsc_optitrack", 8)

with_theme(theme_ggplot2()) do
    fig = HybridZuptInsJl.plot_inertial_data(imu_dcsc)
end