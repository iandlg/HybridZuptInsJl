include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;
using LinearAlgebra

# Single quaternion → DCM
q = [0.1, 0.5, 0.5, 0.4]   # qx,qy,qz,qw
q = HybridZuptInsJl.normalize_quat(q)
R = HybridZuptInsJl.quat_to_matrix(q)          # 3×3 matrix
eu = HybridZuptInsJl.matrix_to_euler(R)
R = HybridZuptInsJl.euler_to_matrix(eu)

# Single DCM → quaternion
q2 = HybridZuptInsJl.matrix_to_quat(R)         # 4‑element vector (should be ≈ normalized q)

# Batched: N = 10 quaternions
q_batch = rand(4, 10)
q_batch = HybridZuptInsJl.normalize_quat(q_batch)
R_batch = HybridZuptInsJl.quat_to_matrix(q_batch)      # 3×3×10
q_back = HybridZuptInsJl.matrix_to_quat(R_batch)      # 4×10 matrix


# Test rotations and quaternion
x = [0.1, 0.2, 0.3]
x_quat = [0, 0.1, 0.2, 0.3]
q = HybridZuptInsJl.normalize_quat([0.1, 0.5, 0.5, 0.4])
q_conj = HybridZuptInsJl.quat_conjugate(q)
x_prime_q = HybridZuptInsJl.quat_multiply(HybridZuptInsJl.quat_multiply(q, x_quat), q_conj)

x_R = HybridZuptInsJl.quat_to_matrix(q) * x


# Test using rotation vectors
u = [sqrt(2) / 2, sqrt(2) / 2, 0] ./ norm([sqrt(2) / 2, sqrt(2) / 2, 0])
phi = pi / 3
v = phi * u

q = [cos(phi / 2); sin(phi / 2) .* u]

R = I(3) + sin(phi) .* HybridZuptInsJl.skew(u) + (1 - cos(phi)) * HybridZuptInsJl.skew(u)^2
R_exp = exp(HybridZuptInsJl.skew(v))
q_R = HybridZuptInsJl.matrix_to_quat(R)
q_R = q_R ./ norm(q_R)


HybridZuptInsJl.quat_to_matrix([1, 0, 0, 0])
HybridZuptInsJl.quat_to_matrix([1, -1, 0, 0])
HybridZuptInsJl.quat_to_matrix([-1, 1, 0, 0])
HybridZuptInsJl.quat_to_matrix(HybridZuptInsJl.quat_conjugate([1, -1, 0, 0]))

q1 = [0.1, 0.2, 0.3, 0.4]
q1 = q1 ./ norm(q1)
q2 = [0.5, 0.2, 0.3, 0.4]

q_mul = HybridZuptInsJl.quat_multiply(q1, q2)
R_mul = HybridZuptInsJl.quat_to_matrix(q_mul)

R = HybridZuptInsJl.quat_to_matrix(q1) * HybridZuptInsJl.quat_to_matrix(q2)

q1 = [0.1, 0.2, 0.3, 0.4]
q1 = q1 ./ norm(q1)
R1 = HybridZuptInsJl.quat_to_matrix(q1)
R1_eq_65 = (q1[1]^2 - q1[2:4]' * q1[2:4]) * I(3) + 2 * q1[2:4] * q1[2:4]' + 2 * q1[1] * HybridZuptInsJl.skew(q1[2:4])