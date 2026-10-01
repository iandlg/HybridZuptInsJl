# Entry point for `Pkg.test()`. The tests read `data/...` relative to the repo root.

using StrideGP
cd(pkgdir(StrideGP))

include("test_equations.jl")
include("hybrid_v4_test.jl")
