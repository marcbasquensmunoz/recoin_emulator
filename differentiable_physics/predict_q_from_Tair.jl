using LinearAlgebra
using Zygote
using Flux

state_dim = 2
hidden_dim = 32
model = Chain(
    Dense(state_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, state_dim)
)

function evaluate(model, input)
    model(input)
end

####
# Define the loss function
loss(y_hat, y) = Flux.mse(y_hat, y)

function estimate_loss(model, data)
    Ndata = size(data.data[1])[end]
    perm = rand(1:Ndata, 32)
    @views result = evaluate(model, data.data[1][:,perm])
    @views output = data.data[2][:,perm]
    loss(result, output)
end

####
# Define the optimizer
opt_state = Flux.setup( 
    Flux.OptimiserChain(
        #WeightDecay(), 
        Flux.Adam())
    , model
)
# Flux.adjust!(opt_state, η = 1e-3)

####
# Training
loss_train = zeros(0)
loss_test = zeros(0)

Flux.trainmode!(model)
@time for epoch in 1:100
    @info "Epoch $(epoch)"
    count = 1
    for batch in data
        #@info "Processed $(round(100*count*batchsize/length(train_range), digits=2))% data points"
        input, output = batch
        grads = Flux.gradient(model) do m
            result = evaluate(m, input)
            loss(result, output)
        end
        Flux.update!(opt_state, model, grads[1])
        count += 1
        push!(loss_train, estimate_loss(model, data))
        push!(loss_test, estimate_loss(model, test_data))
    end
end

plot(log10.(loss_train))

