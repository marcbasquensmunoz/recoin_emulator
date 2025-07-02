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

X = zeros(Float32, 4, Nt_train, N_train) # Input 
Y = zeros(Float32, 3, N_train)           # Output

# Rolling window sampling
i = 1
for j in 1:N
    for k in 1:Nt-Nt_train
        @views norm_Tin = normalise(Tin_synth[k:k+Nt_train, j])
        @views norm_Tb = normalise(Tb_synth[k:k+Nt_train, j])
        @views norm_Q = normalise(Q_synth[k:k+Nt_train, j])
        @views @. X[1, :, i] = Float32.(norm_Tin[1:Nt_train])
        @views @. X[2, :, i] = Float32.(norm_Tb[1:Nt_train])
        @views @. X[3, :, i] = Float32.(norm_Q[1:Nt_train])
        @views @. X[4, :, i] = Float32.(mf_synth[k:k+Nt_train-1, j])
        Y[1, i] = Float32(norm_Tin[end])
        Y[2, i] = Float32(norm_Tb[end])
        Y[3, i] = Float32(norm_Q[end])
        i += 1
    end
end

# Data loaders declaration
batchsize = 512
data      = Flux.DataLoader((X[:,:, train_range], Y[:, train_range]), batchsize=batchsize, shuffle=true)
test_data = Flux.DataLoader((X[:,:, test_range],  Y[:, test_range]),  batchsize=32,        shuffle=true)

# Model
hidden_size = 40
model = Chain(
    LSTM(4 => hidden_size),
    LSTM(hidden_size => hidden_size),
    Dense(hidden_size => 3)
) 
loss(y_hat, y) = Flux.mse(y_hat, y)
opt_state = Flux.setup(
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam(1e-3))
    , model
)

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

@time for epoch in 1:1
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

@views Tin_original = Tin_synth[:, K:K]'
@views Tb_original = Tb_synth[:, K:K]'
@views Q_original = Q_synth[:, K:K]'
@views mf_original = mf_synth[:, K:K]'
input_full = vcat(Tin_original, Tb_original, Q_original, mf_original)

# Using the truth data to predict each time step
prediction = zeros(3, Nt-Nt_train)

for i in 1:Nt-Nt_train
    @views input = input_full[:, i:i+Nt_train-1]
    @views μ_input = mean(input, dims=2)[1:3]
    @views σ_input = std(input, dims=2)[1:3]
    prediction[:, i] .= μ_input .+ evaluate(model, normalise(input, dims=2)) .* σ_input
end

fig_example = Figure(size=(2000, 600))
ax_Tin = Axis(fig_example[1, 1], xlabel = L"t \ (h)", ylabel = L"T_{in} \ (K)", title = "Example $K")
lines!(ax_Tin, 1:Nt, Tin_original[1,:], label="Truth")
lines!(ax_Tin, Nt_train+1:Nt, prediction[1,:], label="Prediction")

ax_Tb = Axis(fig_example[1, 2], xlabel = L"t \ (h)", ylabel = L"T_{b} \ (K)", title = "Example $K")
lines!(ax_Tb, 1:Nt, Tb_original[1,:], label="Truth")
lines!(ax_Tb, Nt_train+1:Nt, prediction[2,:], label="Prediction")

ax_Q = Axis(fig_example[1, 3], xlabel = L"t \ (h)", ylabel = L"Q \ (W/m)", title = "Example $K")
lines!(ax_Q, 1:Nt, Q_original[1,:], label="Truth")
lines!(ax_Q, Nt_train+1:Nt, prediction[3,:], label="Prediction")
fig_example
