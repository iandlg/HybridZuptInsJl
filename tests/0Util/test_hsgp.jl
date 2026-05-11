include("../../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

m = 5
d = 3
ls = 0.5
var_f = 2.1
L = [1, 2, 3]

@show eig_per_dim = HybridZuptInsJl.calc_eigenvalues(L, m, d)
display(eig_per_dim)


x = [1 2 3; 4 5 6]
eig_vects = HybridZuptInsJl.calc_eigenvectors(x, L, eig_per_dim)
display(eig_vects)