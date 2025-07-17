using CSV
using DataFrames

file = "$(@__DIR__)/../../../data/data/lausen/internal_model_data.csv"
raw_data = CSV.read(file, DataFrame; delim = ',')
raw_data = dropmissing(raw_data)

N_real = 1
Nt_real = length(raw_data._time)
data = zeros(4, Nt_real, N_real)

ρf = 1051.2 
Nb = 3
h_to_s = 3600.

data[1, :, 1] .= raw_data[!, :Tfin1]
data[2, :, 1] .= raw_data[!, :meanTb1]
data[3, :, 1] .= raw_data[!, :q_all] ./ Nb
data[4, :, 1] .= raw_data[!, :flowRate] .* ρf / h_to_s / Nb
