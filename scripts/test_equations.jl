include("../src/HybridZuptInsJl.jl")
using .HybridZuptInsJl
using LinearAlgebra, Test

# ── helpers exposed from the module ────────────────────────────────────────
const IQ = HybridZuptInsJl.integrate_quaternion
const NE = HybridZuptInsJl.navigation_equations
const SM = HybridZuptInsJl.state_matrix
const CI = HybridZuptInsJl.comp_internal_states
const QM = HybridZuptInsJl.quat_to_matrix
const MQ = HybridZuptInsJl.matrix_to_quat
const ME = HybridZuptInsJl.matrix_to_euler
const EM = HybridZuptInsJl.euler_to_matrix

# ═══════════════════════════════════════════════════════════════════════════
# 1.  integrate_quaternion
#     Ground truth: pure-z rotation of π/2 starting from identity.
#     q{[0,0,π/2]} = [cos(π/4), 0, 0, sin(π/4)]
# ═══════════════════════════════════════════════════════════════════════════
@testset "integrate_quaternion" begin

    @testset "identity stays identity for zero rate" begin
        q0 = [1.0, 0.0, 0.0, 0.0]
        q1 = IQ(q0, [0.0, 0.0, 0.0], 0.01)
        @test q1 ≈ q0 atol = 1e-12
    end

    @testset "pure-z rotation 90° in one step" begin
        # ω = [0, 0, π/2] rad/s, Ts = 1 s  → should rotate 90° around z
        q0 = [1.0, 0.0, 0.0, 0.0]   # identity
        w = [0.0, 0.0, π / 2]
        Ts = 1.0
        q1 = IQ(q0, w, Ts)

        # Hamilton: q{v} = [cos(|v|/2), (v/|v|)sin(|v|/2)]
        #   |v| = π/2, so q_expected = [cos(π/4), 0, 0, sin(π/4)]
        q_expected = [cos(π / 4), 0.0, 0.0, sin(π / 4)]
        @test q1 ≈ q_expected atol = 1e-10

        # Sanity: result must be unit quaternion
        @test norm(q1) ≈ 1.0 atol = 1e-12
    end

    @testset "Hamilton body-frame convention: q ⊗ dq not dq ⊗ q" begin
        # Rotate 90° around x then 90° around y IN BODY FRAME.
        # Hamilton body-frame means the second rotation is expressed in the
        # already-rotated frame, so composition is q_new = q_old ⊗ dq.
        # After x-rot: body z points in nav -y direction.
        # A subsequent y-rot in the new body frame rotates around nav -y.
        q0 = [1.0, 0.0, 0.0, 0.0]

        # step 1: 90° around body-x
        w1 = [π / 2, 0.0, 0.0]
        q1 = IQ(q0, w1, 1.0)
        @test q1 ≈ [cos(π / 4), sin(π / 4), 0.0, 0.0] atol = 1e-10

        # step 2: 90° around body-y (which is now rotated)
        w2 = [0.0, π / 2, 0.0]
        q2 = IQ(q1, w2, 1.0)

        # Verify via rotation matrix: apply q2 to nav e_z = [0,0,1]
        R2 = QM(q2)
        v_rotated = R2 * [0.0, 0.0, 1.0]

        # After 90° x-rot body-z → nav -y; then 90° y-rot in new body tilts -y → nav -x
        # So rotated e_z should end up at [-1, 0, 0] in nav frame (to within sign convention)
        # The exact value depends on convention; what matters is it is NOT [0,-1,0]
        # (which is what dq⊗q would give).
        @test abs(v_rotated[1]) > 0.9   # dominant component is x, not y
    end

    @testset "output is always unit norm" begin
        for _ in 1:20
            q = normalize(randn(4))
            w = randn(3) .* 5.0   # large rates
            q1 = IQ(q, w, 0.01)
            @test norm(q1) ≈ 1.0 atol = 1e-10
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 2.  navigation_equations
#     Test A: stationary IMU → position and velocity must not drift.
#     Test B: constant acceleration along nav-x → parabolic position.
#     Test C: constant yaw rate → attitude changes, position stays zero.
# ═══════════════════════════════════════════════════════════════════════════
@testset "navigation_equations" begin
    g = [0.0, 0.0, 9.81]    # NED: gravity is +z
    Ts = 0.01

    @testset "stationary: specific force cancels gravity, no drift" begin
        # Level, identity orientation.  Specific force = R'*(-g_nav) for static IMU.
        # For identity R and NED g=[0,0,9.81], am = [0, 0, -9.81] cancels gravity.
        q0 = [1.0, 0.0, 0.0, 0.0]
        x0 = zeros(9)   # pos=vel=euler all zero

        # am must equal R'*(−g) = −g for identity orientation in NED
        am = [0.0, 0.0, -9.81]
        u = [am; zeros(3)]   # no angular rate

        x1, q1 = NE(x0, u, q0, Ts, g)

        @test x1[1:3] ≈ zeros(3) atol = 1e-10   # no position change
        @test x1[4:6] ≈ zeros(3) atol = 1e-10   # no velocity change
        @test q1 ≈ q0 atol = 1e-10   # no attitude change
    end

    @testset "free-fall: zero specific force → downward acceleration" begin
        # am = 0 (free-fall), identity orientation, NED g=[0,0,+9.81]
        # Expected: v_z grows by g*Ts each step, pos_z by ½g*Ts²
        q0 = [1.0, 0.0, 0.0, 0.0]
        x0 = zeros(9)
        u = zeros(6)   # am=[0,0,0], wm=[0,0,0]

        x1, _ = NE(x0, u, q0, Ts, g)

        @test x1[6] ≈ g[3] * Ts atol = 1e-10   # v_z = g*Ts
        @test x1[3] ≈ 0.5 * g[3] * Ts^2 atol = 1e-10   # p_z = ½g*Ts²
        @test x1[1:2] ≈ zeros(2) atol = 1e-10   # no lateral movement
    end

    @testset "pure yaw rate: position/velocity unchanged" begin
        q0 = [1.0, 0.0, 0.0, 0.0]
        x0 = zeros(9)
        am = [0.0, 0.0, -9.81]   # static specific force
        wm = [0.0, 0.0, 0.1]     # yaw rate only
        u = [am; wm]

        x1, q1 = NE(x0, u, q0, Ts, g)

        @test x1[1:3] ≈ zeros(3) atol = 1e-8   # no position change
        @test x1[4:6] ≈ zeros(3) atol = 1e-8   # no velocity change
        @test q1 ≉ q0                     # attitude DID change
    end

    @testset "state vector position matches discrete integral" begin
        # x-acceleration: am_x = 1 m/s² (specific force in body = 1 m/s² minus gravity)
        # identity orientation, NED.  Expected after Ts:
        #   v_x = 1*Ts, p_x = 0.5*1*Ts²
        q0 = [1.0, 0.0, 0.0, 0.0]
        x0 = zeros(9)
        ax = 1.0
        am = [ax, 0.0, -9.81]   # cancel gravity in z, accelerate in x
        u = [am; zeros(3)]

        x1, _ = NE(x0, u, q0, Ts, g)

        @test x1[4] ≈ ax * Ts atol = 1e-10
        @test x1[1] ≈ 0.5 * ax * Ts^2 atol = 1e-10
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 3.  state_matrix  (F and G)
#     Test A: F[1:3,4:6] = I*Ts  (position from velocity)
#     Test B: F[4:6,7:9] = −R[f]× Ts  (velocity from attitude error, Solà eq.166)
#     Test C: F[7:9,7:9] = R'{ω*Ts}  (attitude propagation)
#     Test D: G[4:6,1:3] = R*Ts  (accel noise to velocity)
#     Test E: G[7:9,4:6] = +I*Ts or +R*Ts per convention (NOT −R*Ts)
# ═══════════════════════════════════════════════════════════════════════════
@testset "state_matrix" begin
    Ts = 0.01
    # Identity orientation, specific force along x, yaw rate
    q = [1.0, 0.0, 0.0, 0.0]
    f = [2.0, 0.5, -9.81]
    w = [0.0, 0.0, 0.3]
    u = [f; w]
    R = QM(q)   # = I for identity quaternion

    F, G = SM(q, u, Ts)

    @testset "F pos-vel block = I*Ts  (Solà eq.166 top-right)" begin
        @test F[1:3, 4:6] ≈ I(3) .* Ts atol = 1e-12
    end

    @testset "F vel-att block = −R[f]× Ts  (Solà eq.166)" begin
        # [f]× skew matrix
        fx = f
        skew_f = [0 -fx[3] fx[2]; fx[3] 0 -fx[1]; -fx[2] fx[1] 0]
        expected = -R * skew_f * Ts
        @test F[4:6, 7:9] ≈ expected atol = 1e-10
    end

    @testset "F att-att block = I (first-order Euler approx)" begin
        @test F[7:9, 7:9] ≈ I(3) atol = 1e-12
    end

    @testset "G vel block = R*Ts  (accel noise → velocity)" begin
        @test G[4:6, 1:3] ≈ R .* Ts atol = 1e-12
    end

    @testset "G att block sign: must be +I*Ts not −R*Ts  (Solà eq.167 Fi)" begin
        # Solà Fi att block maps gyro impulse θᵢ → δθ with coefficient +I
        # In G this corresponds to G[7:9, 4:6].
        # The sign must be POSITIVE.  A negative value here is the bug.
        att_gyro_block = G[7:9, 4:6]
        # Check that the diagonal is positive (not negative)
        @test all(diag(att_gyro_block) .> 0)
    end

    @testset "F is 9×9 and G is 9×6" begin
        @test size(F) == (9, 9)
        @test size(G) == (9, 6)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 4.  comp_internal_states
#     Test A: zero correction → state unchanged
#     Test B: position/velocity correction is purely additive
#     Test C: attitude correction rotates the quaternion correctly
#     Test D: returned quaternion is consistent with returned Euler angles
# ═══════════════════════════════════════════════════════════════════════════
@testset "comp_internal_states" begin

    @testset "zero correction leaves state unchanged" begin
        q0 = normalize([0.9, 0.1, 0.2, 0.1])
        x0 = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.05, -0.1, 0.4]
        dx = zeros(9)
        xout, qout = CI(x0, dx, q0)
        @test xout[1:6] ≈ x0[1:6] atol = 1e-12
        @test qout ≈ q0 atol = 1e-10
    end

    @testset "pos/vel correction is additive" begin
        q0 = [1.0, 0.0, 0.0, 0.0]
        x0 = zeros(9)
        dx = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.0, 0.0, 0.0]
        xout, _ = CI(x0, dx, q0)
        @test xout[1:6] ≈ dx[1:6] atol = 1e-12
    end

    @testset "small attitude correction rotates quaternion" begin
        q0 = [1.0, 0.0, 0.0, 0.0]
        x0 = zeros(9)
        ε = [0.01, 0.0, 0.0]   # small roll error
        dx = [zeros(6); ε]
        xout, qout = CI(x0, dx, q0)

        # quaternion must remain unit
        @test norm(qout) ≈ 1.0 atol = 1e-10

        # Euler angles in xout[7:9] must be consistent with qout
        R_from_q = QM(qout)
        euler_from_q = ME(R_from_q)
        @test xout[7:9] ≈ euler_from_q atol = 1e-10
    end
end

println("\n✓ All equation unit tests passed (or failures printed above).")