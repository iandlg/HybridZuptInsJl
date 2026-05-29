# scripts/main.jl
include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
import DataFrames
import CSV

imu_mti100 = HybridZuptInsJl.InertialData("data/mti-100-recordings/MT_01782205_002-000.txt")

imu_ang = HybridZuptInsJl.InertialData("data/angermann_high_precision/15_IMURaw.csv")

fig = HybridZuptInsJl.plot_inertial_data(imu_mti100)
display(fig)   # opens an interactive window
