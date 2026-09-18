# Shared by 10_yaw_prediction_diagnosis.jl and 11_stride_correction_benchmark.jl.
#
# Include after the module include and after ../5Results/_common.jl. Run from
# the repository root.

using OrderedCollections, Statistics, LinearAlgebra

"""
Hyperparameter key per dataset (see `HSGP_PARAM_PATHS` in _common.jl). Each set
was trained on its own dataset; they are not interchangeable.
"""
const DATASET_HSGP_KEYS = OrderedDict{String,Int}(
    "ANG2" => 42,
    "DCSC" => 47,
)

"""
    run_variant(filter, estimator_type, channels, res, params; frame, feature_type, train_ratio)

Run one filter/estimator pair on one aligned trial (an entry of
`collect_aligned_trajectories`) with the first `train_ratio` of the samples
under ground truth. Returns `(; diag, io, step_seg, traj, n_train_cutoff)`.
"""
function run_variant(filter::Function, estimator_type::Type, channels::Vector{Symbol},
    res::NamedTuple, params; frame, feature_type, train_ratio::Float64,
    estimator_alloc::Int=2000)

    N = length(res.inertial_updated)
    n_train_cutoff = floor(Int, train_ratio * N)
    corrector = estimator_type(estimator_alloc; params=params, corrected_channels=channels)
    diag = HybridZuptInsJl.CorrectorDiagnostics()

    _, step_seg, traj, io, _ = filter(
        res.inertial_updated, res.sim_config_updated, res.gt_traj_aligned, corrector;
        x_init=res.x_init,
        gt_available=[n <= n_train_cutoff for n in 1:N],
        ref_frame=frame, feature_type=feature_type,
        diagnostics=diag)

    return (; diag, io, step_seg, traj, n_train_cutoff)
end

"""
    footfall_yaw_error(diag, gt_traj) -> Vector{Float64}

Heading error `wrap(ψ_gt − ψ_corrector)` at each recorded footfall.
"""
footfall_yaw_error(diag, gt_traj) = [HybridZuptInsJl.wrap_pi(
    HybridZuptInsJl.matrix_to_euler(gt_traj.R_nb[:, :, k])[3] -
    HybridZuptInsJl.matrix_to_euler(HybridZuptInsJl.quat_to_matrix(q))[3])
                                     for (k, q) in zip(diag.k, diag.quat)]
