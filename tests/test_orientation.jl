include("../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;


# Single quaternion → DCM
q = [0.1, 0.5, 0.5, 0.4]   # qx,qy,qz,qw
q = HybridZuptInsJl.normalize_quat(q)
R = HybridZuptInsJl.quat_to_matrix(q)          # 3×3 matrix

# Single DCM → quaternion
q2 = HybridZuptInsJl.matrix_to_quat(R)         # 4‑element vector (should be ≈ normalized q)

# Batched: N = 10 quaternions
q_batch = rand(4, 10)
q_batch = HybridZuptInsJl.normalize_quat(q_batch)
R_batch = HybridZuptInsJl.quat_to_matrix(q_batch)      # 3×3×10
q_back = HybridZuptInsJl.matrix_to_quat(R_batch)      # 4×10 matrix