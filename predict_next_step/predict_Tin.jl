using Flux
using Flux: normalise
using Makie
using CairoMakie
include("../generate_simulated_data.jl")

# Generate training data
Nt_train = 24*21           # Number of previous values in each sample 
N_train = N*(Nt-Nt_train)  # Number of training samples

train_range = 1:Int(0.8*N_train)
test_range = Int(0.8*N_train):N_train

X = zeros(Float32, 1, Nt_train, N_train) # Input 
Y = zeros(Float32, 1, N_train)           # Output

# Rolling window sampling
i = 1
for j in 1:N
    for k in 1:Nt-Nt_train
        @views data = normalise(T_synth[k:k+Nt_train, j])
        @views @. X[1, :, i] = Float32.(data[1:Nt_train])
        Y[1, i] = Float32(data[end])
        i += 1
    end
end

# Data loaders declaration
batchsize = 512
data      = Flux.DataLoader((X[:,:, train_range], Y[:, train_range]), batchsize=batchsize, shuffle=true)
test_data = Flux.DataLoader((X[:,:, test_range],  Y[:, test_range]),  batchsize=32,  shuffle=true)

# Model
hidden_size = 20
model = Chain(
    LSTM(1 => hidden_size),
    LSTM(hidden_size => hidden_size),
    Dense(hidden_size => 1)
) 
#=
hidden_size = 50
model = Chain(
    LSTM(1 => hidden_size),
    Dense(hidden_size => 1)
) 
=#
loss(y_hat, y) = Flux.mse(y_hat, y)
opt_state = Flux.setup(
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam(1e-3))
    , model
)
# Flux.adjust!(opt_state, η = 1e-4)

function evaluate(model, input)
    Flux.reset!(model)
    @views model(input)[:, end, :]
end

function estimate_loss(model, data)
    Ndata = size(data.data[1])[end]
    perm = rand(1:Ndata, 500)
    @views result = evaluate(model, data.data[1][:,:,perm])
    @views output = data.data[2][:,perm]
    loss(result, output)
end

# Training
loss_train = zeros(0)
loss_test = zeros(0)

@time for epoch in 1:10
    @info "Epoch $(epoch)"
    count = 1
    for batch in data
        @info "Processed $(round(100*count*batchsize/length(train_range), digits=2))% data points"
        input, output = batch
        grads = Flux.gradient(model) do m
            result = evaluate(m, input)
            loss(result, output)
        end
        Flux.update!(opt_state, model, grads[1])
        count += 1
        if count % 10 == 0
            push!(loss_train, estimate_loss(model, data))
            push!(loss_test, estimate_loss(model, test_data))
        end
    end
end


# Loss plot
fig_loss = Figure()
ax = Axis(fig_loss[1, 1], xlabel = "Training step", ylabel =  L"\log_{10} L", title="Loss evolution during training")
lines!(ax, log10.(loss_train), label="Train")
lines!(ax, log10.(loss_test), label="Test")
axislegend(""; position= :rt, backgroundcolor = (:grey90, 0.25));
fig_loss


################################
# Prediction (from the test set)
################################
K = N
Tin_original = Tin_synth[:, K:K]'

# Using the truth data to predict each time step
prediction = zeros(Nt-Nt_train)
for i in 1:Nt-Nt_train
    input_data = Tin_original[:, i:i+Nt_train-1]
    μ_T = mean(input_data)
    σ_T = std(input_data)
    prediction[i] = μ_T + evaluate(model, normalise(input_data))[end] * σ_T
end

# Using the previously predicted data to predict the next time step
input = hcat(Tin_original[:, 1:Nt_train], zeros(1, Nt-Nt_train))
for i in 1:Nt-Nt_train
    input_data = input[:, i:i+Nt_train-1]
    μ_T = mean(input_data)
    σ_T = std(input_data)
    input[i+Nt_train] = μ_T + evaluate(model, normalise(input_data))[end] * σ_T
end

fig_example = Figure()
ax = Axis(fig_example[1, 1], xlabel = L"t \ (h)", ylabel = L"T_{in} \ (K)", title = "Example $K")
lines!(ax, 1:Nt, Tin_original[1,:], label="Truth")
lines!(ax, Nt_train+1:Nt, prediction, label="Prediction (truth until last step)")
lines!(ax, 1:Nt, input[1,:], label="Prediction (previous predictions)")
axislegend(""; position= :rt, backgroundcolor = (:grey90, 0.25));
fig_example

save("$(@__DIR__)/example.png", fig_example)
