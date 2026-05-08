include("../src/HybridZuptInsJl.jl");
using .HybridZuptInsJl;

cfg = HybridZuptInsJl.InsConfig(g=9.81, window_size=3, gamma=0.5e5, sigma_a=0.01, sigma_g=0.2 * π / 180)
u = HybridZuptInsJl.randn(6, 1000)
zupt, logL = HybridZuptInsJl.detect_zupt(u, cfg)
println(sum(zupt))
