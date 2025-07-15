using Flux
using CSV
using DataFrames

file = "$(@__DIR__)/../../lausen_data_fetching/sample_data.csv"
raw_data = CSV.read(file, DataFrame; delim = ',')

N = length(raw_data.t)
ρ_f(T) = 1051.2 - 0.4719 * T
cp_f(T) = 3710 + 0.002 * T

Tair = raw_data[!, :airT]
Tb = Matrix(raw_data[!, [:meanTb1, :meanTb2, :meanTb3]])'
dTf = Matrix(raw_data[!, [:dT1, :dT2, :dT3]])'
Tfin = Matrix(raw_data[!, [:Tfin1, :Tfin2, :Tfin3]])'
Tfout = Matrix(raw_data[!, [:Tfout1, :Tfout2, :Tfout3]])'
q = raw_data[!, :q_all] .* 1000
q_calc = raw_data[!, :q_all_calc] .* 1000
mf = raw_data[!, :flowRate] .* ρ_f.(Tfin[1,:]) / 3600  # Assume same mass flow to each borehole

X = zeros(11, N)

X[1:3, :] .= Tfin
X[4:6, :] .= Tfout
X[7:9, :] .= Tb
X[10, :] .= q_calc
X[11, :] .= mf



# Data loaders declaration
batchsize = N
data_1      = Flux.DataLoader(X[1:9, :], batchsize=batchsize, shuffle=true)
data_2      = Flux.DataLoader(X[:, :], batchsize=batchsize, shuffle=true)


fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, Tfin[1, :], label="Tfin")
lines!(ax, Tfout[1, :], label="Tfout")
lines!(ax, Tb[1, :], label="Tb")
axislegend(position = :rt)
fig


q_ref = vec(sum((Tfout-Tfin), dims=1)) .* cp_f.(Tfin[1,:]) .* mf
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, q, label="q")
lines!(ax, q_calc, label="Database computed q")
lines!(ax, q_ref, label="Computed q")
axislegend(position = :rt)
fig