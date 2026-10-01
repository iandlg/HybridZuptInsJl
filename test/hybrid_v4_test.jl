# Tests the core functions the 5Results scripts run Hybrid V4 through, on mock ANG2 and
# DCSC trials (test/mock_trials.jl). Run from the repository root:
#
#     julialauncher --project=. -t 1 test/hybrid_v4_test.jl
#
# or the whole suite with `julialauncher --project=. -t 1 -e "using Pkg; Pkg.test()"`.

using StrideGP
include("../scripts/5Results/_common.jl")
include("mock_trials.jl")
using Test, OrderedCollections, DataFrames, LinearAlgebra, Statistics

const H = StrideGP
const SECONDS = 40
# DCSC 1 and 2 have a -33/-16 s mocap lag, which 40 s is too short to re-estimate, and
# 6 locks onto a wrong peak at 40 s; 4 and 8 re-estimate it to within one sample.
const MOCK_IDS = OrderedDict("ANG2" => [1, 2], "DCSC" => [4, 8])
const CHANNELS = [:pos_1, :pos_2, :yaw]
const ALLOC = 300

hsgp_p, FRAME, FEATURE_TYPE, _ = load_hsgp_params(42; m=200)
mock_dirs = OrderedDict(k => mock_dataset(k, ids; seconds=SECONDS) for (k, ids) in MOCK_IDS)
aligned = H.collect_aligned_trajectories(
    OrderedDict{String,Tuple{String,Vector{Int}}}(k => (mock_dirs[k], MOCK_IDS[k]) for k in keys(MOCK_IDS)))
n_trials = sum(length, values(aligned))

corrector(kind::Symbol) = CORRECTORS["V4"][kind](ALLOC;
    params=hsgp_p, corrected_channels=CHANNELS)

function run_v4(res, c; train_ratio::Real, kwargs...)
    N = length(res.inertial_updated)
    return CORRECTION_FILTERS["V4"](res.inertial_updated, res.sim_config_updated, res.gt_traj_aligned, c;
        x_init=res.x_init, gt_available=[n <= floor(Int, train_ratio * N) for n in 1:N],
        ref_frame=FRAME, feature_type=FEATURE_TYPE, kwargs...)
end

@testset "Hybrid V4" begin

    # DCSC refits the IMU-to-rigid-body rotation and the lag on the mock's own window,
    # so its ground truth only matches the full trial's to a few cm (orientation is not
    # compared); ANG2 is synchronised in the recording and matches exactly.
    @testset "mock trials match the full recordings" begin
        gt_atol = Dict("ANG2" => 1e-3, "DCSC" => 0.1)
        full = H.collect_aligned_trajectories(OrderedDict{String,Tuple{String,Vector{Int}}}(
            k => (DATA_DIRS[k], ids) for (k, ids) in MOCK_IDS))
        for (key, ids) in MOCK_IDS, id in ids
            m, f = aligned[key][id], full[key][id]
            n = length(m.inertial_updated)
            @test SECONDS - 10 < m.inertial_updated.t[end] - m.inertial_updated.t[1] <= SECONDS + 0.1
            @test m.inertial_updated.t == f.inertial_updated.t[1:n]
            @test maximum(abs, m.gt_traj_aligned.pos - f.gt_traj_aligned.pos[:, 1:n]) < gt_atol[key]
            @test maximum(abs, m.x_init - f.x_init) < 0.05
        end
    end

    @testset "stride noise σ_w = σ_n" begin
        σ_n = [getfield(hsgp_p.hp, Symbol(name))[1] * hsgp_p.output_stats[2][j]
               for (j, name) in enumerate(H._OUTPUT_NAMES)]
        @test H.stride_noise_std(hsgp_p) ≈ σ_n
        @test corrector(:static).σ_w == corrector(:hsgp).σ_w == H.stride_noise_std(hsgp_p)
    end

    @testset "BaseEstimator stride noise" begin
        σ = H.sigma_stride_array(H.InsConfig())
        ψ = 0.7
        q0 = H.quat_exp([0.0, 0.0, ψ])
        step(c; kw...) = H.propagate_stride!(c; t=1.0, Δp=[0.7, 0.0, 0.0], Δq=H.quat_exp([0.0, 0.0, 0.05]),
            Σpq=zeros(6, 6), kw...)

        c = H.BaseEstimator(ALLOC)
        H.initialize_corrector!(c; t=0.0, pos_init=zeros(3), quat_init=q0, Σpq_init=zeros(6, 6))
        step(c; σ_stride=σ)
        R_ψ = H.euler_to_matrix([0.0, 0.0, ψ])
        @test c.Σ[6, 6] ≈ σ[4]^2
        @test c.Σ[1:3, 1:3] ≈ R_ψ * Diagonal(σ[1:3] .^ 2) * R_ψ'
        @test all(iszero, c.Σ[1:3, 4:6])

        c = H.BaseEstimator(ALLOC)
        H.initialize_corrector!(c; t=0.0, pos_init=zeros(3), quat_init=q0, Σpq_init=zeros(6, 6))
        @test_throws UndefKeywordError step(c)

        # With the noise, ZUPT only follows the mocap yaw in the train phase instead of
        # trusting its own heading over it (notes/022).
        res = aligned["ANG2"][1]
        k0 = floor(Int, 0.5 * length(res.inertial_updated))
        d = H.CorrectorDiagnostics()
        run_v4(res, H.BaseEstimator(ALLOC); train_ratio=0.5, diagnostics=d)
        gt = res.gt_traj_aligned[d.k]
        eψ = [abs(H.rotmat_to_rotvec(gt.R_nb[:, :, i] * H.quat_to_matrix(d.quat[i])')[3]) for i in 1:length(d)]
        @test median(eψ[d.k .<= k0]) <= 5e-3
    end

    # A real stride feature, from the INS of the first mock trial.
    feature = run_v4(first(values(aligned["ANG2"])), H.BaseEstimator(ALLOC); train_ratio=0.0)[4]["input"].data[:, 1]

    @testset "joint corrector, $kind" for kind in (:static, :hsgp)
        m = hsgp_p.m
        nβ = kind === :static ? 3 : 3m
        c = corrector(kind)
        @test length(c.δx) == 6 + nβ && size(c.Σ) == (6 + nβ, 6 + nβ)

        H.initialize_corrector!(c; t=0.0, pos_init=zeros(3), quat_init=[1.0, 0.0, 0.0, 0.0],
            Σpq_init=Matrix(1e-4I, 6, 6))
        β, Σβ = H.get_model(c)
        pos_3 = kind === :static ? (3:3) : H._full_range(3, m)
        @test length(β) == (kind === :static ? 4 : 4m)
        @test all(iszero, β)
        @test all(iszero, Σβ[pos_3, :]) && all(iszero, Σβ[:, pos_3])
        @test all(>(0), diag(Σβ)[setdiff(axes(Σβ, 1), pos_3)])

        kind === :static && @test H.stride_model(c, FEATURE_TYPE, feature) == (zeros(3), Matrix(1.0I, 3, 3))

        Σββ = c.Σ[7:end, 7:end]
        y, Σy = H.propagate_stride!(c; t=1.0, Δp=[0.7, 0.0, 0.0], Δq=H.quat_exp([0.0, 0.0, 0.05]),
            Σpq=Matrix(1e-4I, 6, 6), R_bh=Matrix(1.0I, 3, 3), ins_stride=[0.7, 0.0, 0.0, 0.05],
            ref_frame=FRAME, feature_type=FEATURE_TYPE, feature=feature)
        @test c.i == 2
        @test c.Σ ≈ c.Σ'
        @test c.Σ[7:end, 7:end] == Σββ
        @test y[3] == 0 && all(iszero, Σy[3, :])
        if kind === :static
            @test y == zeros(4)
            @test c.pos[:, 2] ≈ [0.7, 0.0, 0.0]
            @test H.matrix_to_euler(H.quat_to_matrix(c.quat[:, 2]))[3] ≈ 0.05
        end

        tr_before = tr(c.Σ)
        x_before = c.pos[1, 2]
        H.posyaw_measurement_update!(c; curr_pos=c.pos[:, 2] + [0.05, 0.0, 0.0],
            curr_θ3=H.matrix_to_euler(H.quat_to_matrix(c.quat[:, 2]))[3] + 0.01,
            Σy=Diagonal(fill(1e-4, 4)))
        @test tr(c.Σ) < tr_before
        @test any(!iszero, c.δx)
        H.relinearize!(c)
        @test all(iszero, c.δx)
        @test c.pos[1, 2] > x_before
    end

    @testset "hybrid_zupt_aided_insv4: $key $id" for (key, ids) in MOCK_IDS, id in ids
        res = aligned[key][id]
        _, seg_base, traj_base, _, _ = run_v4(res, H.BaseEstimator(ALLOC); train_ratio=0.5)
        @test all(isfinite, traj_base.pos)

        for kind in (:static, :hsgp)
            _, _, _, _, (_, Σβ_prior) = run_v4(res, corrector(kind); train_ratio=0.0)
            zupt, step_seg, traj, io, (β, Σβ) = run_v4(res, corrector(kind); train_ratio=0.5)

            @test length(zupt) == length(res.inertial_updated)
            @test step_seg == seg_base
            @test step_seg[1] == 1 && all(>(0), diff(step_seg))
            @test length(traj) == length(step_seg)
            @test all(isfinite, traj.pos) && all(isfinite, β) && all(isfinite, Σβ)
            @test length(io["target"]) == length(io["input"]) == length(step_seg) - 1
            @test isfinite(H.rmse(traj, res.gt_traj_aligned[step_seg])[end])
            @test any(!iszero, β)
            @test tr(Σβ) < tr(Σβ_prior)
        end

        @test_throws ArgumentError CORRECTION_FILTERS["V4"](res.inertial_updated, res.sim_config_updated,
            res.gt_traj_aligned[1:(end-1)], corrector(:hsgp); x_init=res.x_init)
    end

    @testset "model carry-over across trials, $key $kind" for key in keys(MOCK_IDS), kind in (:static, :hsgp)
        res1, res2 = (aligned[key][id] for id in MOCK_IDS[key])
        model1 = run_v4(res1, corrector(kind); train_ratio=1.0)[5]
        model2 = run_v4(res2, corrector(kind); train_ratio=0.0, init_model=model1)[5]
        @test model2[1] == model1[1]
        @test model2[2] == model1[2]
    end

    @testset "corrector diagnostics: $key $id" for (key, ids) in MOCK_IDS, id in ids
        res = aligned[key][id]
        k0 = floor(Int, 0.3 * length(res.inertial_updated)) + 1
        d = H.CorrectorDiagnostics()
        step_seg = run_v4(res, corrector(:hsgp); train_ratio=0.3, diagnostics=d)[2]
        @test length(d) == length(step_seg) - 1 < ALLOC
        @test any(<(k0), d.k) && any(>=(k0), d.k)

        nees = H.corrector_nees_series(d, res.gt_traj_aligned)
        nyaw = H.nees_yaw_series(d, res.gt_traj_aligned)
        @test all(isfinite, nees.pos) && all(isfinite, nyaw.yaw)
        @test 0 <= H.consistency_ratio(nees.pos, nees.lower, nees.upper) <= 1
        @test 0 <= H.consistency_ratio(nyaw.yaw, nyaw.lower, nyaw.upper) <= 1
    end

    estimators = OrderedDict(
        "ZUPT only" => H.BaseEstimator,
        "Static" => CORRECTORS["V4"].static,
        "HSGP" => CORRECTORS["V4"].hsgp,
    )

    @testset "run_online_correction_sweep" begin
        train_ratios = [0.3, 0.6]
        specs = [H.NoiseSpec(; tag="No Noise"), H.NoiseSpec(; pos_std=0.1, att_std=10π / 180, tag="Noisy")]
        df = H.run_online_correction_sweep(aligned, FRAME, FEATURE_TYPE, hsgp_p, train_ratios,
            estimators, CHANNELS;
            estimator_alloc=ALLOC,
            correction_filter=CORRECTION_FILTERS["V4"],
            noise_specs=specs,
            match_gt_sigma=true,
            keep_artifacts=false)
        # The sweep swallows a failed run with a @warn: a missing row is the failure.
        @test nrow(df) == n_trials * length(train_ratios) * length(estimators) * length(specs)
        @test all(isfinite, df.rmse) && all(isfinite, df.rmse_rate) && all(isfinite, df.rmse_yaw)
        @test all(df.gt_sigma_pos[df.noise_spec_tag .== "Noisy"] .> df.gt_sigma_pos[df.noise_spec_tag .== "No Noise"])

        paired = H.paired_estimator_contrast(df; metric=:rmse, reference_estimator="ZUPT only",
            train_ratios=train_ratios)
        @test nrow(paired) == count(df.estimator .!= "ZUPT only")
    end

    @testset "run_online_nees_sweep" begin
        train_ratios = [0.3, 0.6]
        ff = H.run_online_nees_sweep(aligned, FRAME, FEATURE_TYPE, hsgp_p, train_ratios,
            estimators, CHANNELS;
            estimator_alloc=ALLOC,
            correction_filter=CORRECTION_FILTERS["V4"])
        @test all(isfinite, ff.nees_pos) && all(isfinite, ff.nees_yaw)
        for r in eachrow(ff)
            cutoff = floor(Int, r.train_ratio * length(aligned[r.dataset_name][r.trial_id].inertial_updated))
            @test (r.phase == "train") == (r.k <= cutoff)
        end

        s = H.nees_summary(ff)
        @test nrow(s) == n_trials * length(train_ratios) * length(estimators) * 2
        @test all(0 .<= s.inside_pos .<= 1) && all(0 .<= s.inside_yaw .<= 1)
        @test sum(s.n) == nrow(ff)
    end

    @testset "run_online_learning_curve" begin
        budgets = [0, 3, 6]
        lc = H.run_online_learning_curve(aligned, FRAME, FEATURE_TYPE, hsgp_p, budgets,
            estimators, CHANNELS;
            n_test_strides=10,
            estimator_alloc=ALLOC,
            correction_filter=CORRECTION_FILTERS["V4"])
        @test nrow(lc) == n_trials * length(budgets) * length(estimators)
        @test all(isfinite, lc.rmse) && all(isfinite, lc.rmse_yaw)
        @test all(==(10), lc.n_test_strides)

        contrast = H.learning_curve_contrast(lc; metric=:rmse, reference_estimator="ZUPT only")
        @test nrow(contrast) == count(lc.estimator .!= "ZUPT only")
        @test H.plot_learning_curve_absolute(lc, first(keys(aligned)); metric=:rmse) isa H.Figure
    end
end
