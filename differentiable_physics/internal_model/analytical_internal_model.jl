using BoreholeNetworksSimulator
using Distributions
using DataFrames

Δt = 3600 * 24.
Nt = 182

α = 1e-6
λ = 3.

D = 0.
H = 145.

T0 = 9.

borehole = SingleUPipeBorehole(H=H, D=D)
fluid = GlycolMix()

a_in_ref = zeros(Nt)
for t in 1:Nt
    mf_t = mf[t]
    Tf_t = (Tfin[1, t] + Tfout[1, t])/2
    a_in_ref[t] = BoreholeNetworksSimulator.uniform_Tb_coeffs(borehole, λ, mf_t, Tf_t, fluid)[1]
end

Tb_ref = @. (- a_in_ref * Tfin[1, :] + Tfout[1, :]) / (1 - a_in_ref) 