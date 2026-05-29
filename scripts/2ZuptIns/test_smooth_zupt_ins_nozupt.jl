include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_key = "MT"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "MT" => "data/mti-100-recordings"
)[data_key]
trial_id = 2
inertial = HybridZuptInsJl.InertialData(HybridZuptInsJl.MT, trial_id)

# fig_gt = HybridZuptInsJl.plot_trajectory_xyz_euler(gt_traj)
# Δt = -0.0
# gt_traj = HybridZuptInsJl.Trajectory(
#     (gt.t .+ Δt),
#     gt.pos,
#     gt.R_nb,
#     gt.vel
# )

sim_config = HybridZuptInsJl.InsConfig(gamma=5e3, Ts=1 / 400)

# Compute INS trajectory from inertial data
zupt, ins_traj, segs = HybridZuptInsJl.smoothed_zupt_aided_ins(inertial, sim_config)
fig = HybridZuptInsJl.plot_groundtruth_vs_inertial_positions(ins_traj, nothing; samples=10000000000000)


##
fig_segs = HybridZuptInsJl.plot_inertialdata_and_stepsegm(inertial, segs; zupt=zupt)
fig_3d = HybridZuptInsJl.plot_trajectory_3d(ins_traj)
fig_steps = HybridZuptInsJl.plot_step_lengths(ins_traj, nothing, segs)
