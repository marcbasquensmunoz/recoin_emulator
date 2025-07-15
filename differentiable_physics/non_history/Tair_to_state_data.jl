using Flux
using CSV
using DataFrames

file = "$(@__DIR__)/../../lausen_data_fetching/sample_data.csv"
raw_data = CSV.read(file, DataFrame; delim = ',')

N = length(raw_data.t)
ρ_f = 1051.2

Tair = raw_data[!, :airT]
Tb = Matrix(raw_data[!, [:meanTb1, :meanTb2, :meanTb3]])'
dTf = Matrix(raw_data[!, [:dT1, :dT2, :dT3]])'
q = raw_data[!, :q_all] .* 1000
mf_t = raw_data[!, :flowRate] * ρ_f / 3600

#####
plot(Tb[1, :])
plot!(Tb[2, :])
plot!(Tb[3, :])

plot(dTf[1, :])
plot!(dTf[2, :])
plot!(dTf[3, :])

#=
scatter( raw_data[!, :airT][1:24*30*2],  raw_data[!, :q_all][1:24*30*2])
scatter( raw_data[!, :airT][24*30*2:24*30*4],  raw_data[!, :q_all][24*30*2:24*30*4])
scatter( raw_data[!, :airT][24*30*4:24*30*6],  raw_data[!, :q_all][24*30*4:24*30*6])
scatter( raw_data[!, :airT][24*30*6:24*30*8],  raw_data[!, :q_all][24*30*6:24*30*8])
=#
#####

train_range = 1:round(Int, 0.8*N_data)
test_range = round(Int, 0.8*N_data):N_data

Nb = 3
X = zeros(1, N)
Y = zeros(2Nb+1, N)

for n in 1:N
    window = n-div(w, 2):n+div(w, 2)-1
    #=@views T_window = airT[window]
    μ = mean(T_window)
    σ = std(T_window)
    @views X[1, n-div(w, 2)] = (airT[n] - μ) / σ  # Normalized air temperature at time t
    @views X[2, n-div(w, 2)] = μ=#
    X[1, n] = q[n]
end
Y[1:3, :] = Tb[:, :]
Y[4:6, :] = dTf[:, :]
Y[7, :] = q

#μ_T = mean(X[2, :])

X_norm = X
#X_norm[2, :] .= Flux.normalise(X[2, :])
Y_norm = Flux.normalise(Y, dims=2)

μ_Y = mean(Y, dims=2)
σ_Y = std(Y, dims=2)

# Data loaders declaration
batchsize = 32
data      = Flux.DataLoader((X_norm[:, train_range], Y_norm[:, train_range]), batchsize=batchsize, shuffle=true)
test_data = Flux.DataLoader((X_norm[:, test_range],  Y_norm[:, test_range]),  batchsize=32,  shuffle=true)


