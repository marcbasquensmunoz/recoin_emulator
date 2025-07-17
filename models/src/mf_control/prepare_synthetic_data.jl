
# Data preparation
include("../../../data/src/synthetic/generate_simulated_data.jl")

data = zeros(Float32, 4, Nt, N)

data[1, :, :] .= Tin_synth
data[2, :, :] .= Tb_synth
data[3, :, :] .= Q_synth
data[4, :, :] .= mf_synth