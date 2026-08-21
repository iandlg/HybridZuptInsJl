# Media only: 3-D animated ground-truth trajectory with footfall rings.
# Genuinely needs GLMakie (it records an mp4 via a live GL context), so this is
# the one script here that does not use the shared CairoMakie figure theme.

include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
include("_common.jl")
using GLMakie

data_key = "ANG2"
trial_id = 1

ins_traj_aligned, gt_traj_aligned, zupt, segs, _, _ = HybridZuptInsJl.compute_aligned_ins_trajectory(
    data_dir(data_key), trial_id
)


mkpath("out/Media")
filepath = joinpath("out/Media", "trajectory.mp4")
with_theme() do
    fig = HybridZuptInsJl.animate_trajectory(gt_traj_aligned, segs; lifetime=1.2, filepath=filepath)
    display(fig)
end