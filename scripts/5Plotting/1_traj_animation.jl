include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

trial_id = 8

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)


fig = HybridZuptInsJl.animate_trajectory(gt_traj_aligned, segs; lifetime=0.8, filename="my_trajectory.mp4")
display(fig)   # or just run the function to create the video