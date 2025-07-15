using Flux
using CSV
using DataFrames

file = "$(@__DIR__)/../lausen_data_fetching/sample_data.csv"
data = CSV.read(file, DataFrame; delim = ',')

T = 1
N = length(data.t) - T

train_range = 1:round(Int, 0.8*N)
test_range = round(Int, 0.8*N):N

Nb = 3
# dT_i, Tb_i, q, T_air at time t
X = zeros(2Nb+1+T, N)
# dT_i, Tb_i, q at time t+T
Y = zeros(2Nb+1, N)

@views @. X[1, :] .= data[1:end-T, :dT1]
@views @. X[2, :] .= data[1:end-T, :dT2]
@views @. X[3, :] .= data[1:end-T, :dT3]
@views @. X[4, :] .= data[1:end-T, :meanTb1]
@views @. X[5, :] .= data[1:end-T, :meanTb2]
@views @. X[6, :] .= data[1:end-T, :meanTb3]
@views @. X[7, :] .= data[1:end-T, :q_all]
for t in 1:N
    @views @. X[8:end, t] .= data[t:t+T-1, :airT]
end
@views @. Y[1, :] .= data[T+1:end, :dT1]
@views @. Y[2, :] .= data[T+1:end, :dT2]
@views @. Y[3, :] .= data[T+1:end, :dT3]
@views @. Y[4, :] .= data[T+1:end, :meanTb1]
@views @. Y[5, :] .= data[T+1:end, :meanTb2]
@views @. Y[6, :] .= data[T+1:end, :meanTb3]
@views @. Y[7, :] .= data[T+1:end, :q_all]

X_norm = Flux.normalise(X, dims=2)
Y_norm = Flux.normalise(Y, dims=2)

μ_X = mean(X, dims=2)
σ_X = std(X, dims=2)
μ_Y = mean(Y, dims=2)
σ_Y = std(Y, dims=2)

# Data loaders declaration
batchsize = 32
#data      = Flux.DataLoader((X_norm[:, train_range], Y_norm[:, train_range]), batchsize=batchsize, shuffle=true)
#test_data = Flux.DataLoader((X_norm[:, test_range],  Y_norm[:, test_range]),  batchsize=32,  shuffle=true)
data      = Flux.DataLoader((X[:, train_range], Y[:, train_range]), batchsize=batchsize, shuffle=true)
test_data = Flux.DataLoader((X[:, test_range],  Y[:, test_range]),  batchsize=32,  shuffle=true)


