include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using GLMakie

data_key = "ANG2"
data_dir = Dict{String,String}(
    "ANG" => "data/angermann_high_precision",
    "ANG2" => "data/angermann_v2"
)[data_key]

trial_id = 1

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir, trial_id
)


filepath = joinpath("out/Media", "trajectory.mp4")
with_theme() do
    fig = HybridZuptInsJl.animate_trajectory(gt_traj_aligned, segs; lifetime=1.2, filepath=filepath)
    display(fig)
end