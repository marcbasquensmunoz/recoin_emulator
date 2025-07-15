using Flux
using CSV
using DataFrames

file = "$(@__DIR__)/../lausen_data_fetching/sample_data.csv"
raw_data = CSV.read(file, DataFrame; delim = ',')

N = length(raw_data.t)
w = 24*7

airT = raw_data[!, :airT]
q = raw_data[!, :q_all]

plot(airT); plot!(q)


N_data = N-w

train_range = 1:round(Int, 0.8*N_data)
test_range = round(Int, 0.8*N_data):N_data

Nb = 3
X = zeros(2, N_data)
Y = zeros(2, N_data)

for n in div(w, 2)+1:N-div(w, 2) 
    window = n-div(w, 2):n+div(w, 2)-1
    @views T_window = airT[window]
    μ = mean(T_window)
    σ = std(T_window)
    @views X[1, n-div(w, 2)] = (airT[n] - μ) / σ  # Normalized air temperature at time t
    @views X[2, n-div(w, 2)] = μ

    @views q_window = q[window]
    μ_q = mean(q_window)
    σ_q = std(q_window)
    Y[1, n-div(w, 2)] = (q[n] - μ_q) / σ_q
    Y[2, n-div(w, 2)] = μ_q
end

μ_T = mean(X[2, :])
q_T = mean(Y[2, :])

X_norm = X
X_norm[2, :] .= Flux.normalise(X[2, :])
Y_norm = Y
Y_norm[2, :] .= Flux.normalise(Y[2, :])

# Data loaders declaration
batchsize = 32
data      = Flux.DataLoader((X_norm[:, train_range], Y_norm[:, train_range]), batchsize=batchsize, shuffle=true)
test_data = Flux.DataLoader((X_norm[:, test_range],  Y_norm[:, test_range]),  batchsize=32,  shuffle=true)


