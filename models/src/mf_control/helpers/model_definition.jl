
function create_model(;lstm_hidden, mlp_hidden, prediction_steps) 
    mlp_input = lstm_hidden[end] + prediction_steps
    mlp_output = 3 * prediction_steps
    mlp_layers = []

    for h in mlp_hidden
        push!(mlp_layers, Dense(mlp_input => h, tanh))
        mlp_input = h
    end
    push!(mlp_layers, Dense(mlp_input => mlp_output))  # Output layer without activation

    lstm_input = 4                                     # Number of state variables
    lstm_layers = []

    for h in lstm_hidden
        push!(lstm_layers, LSTM(lstm_input => h))
        lstm_input = h
    end

    Chain(
        Parallel(
            (x, y) -> vcat(x[:, end, :], y),
            Chain(lstm_layers...), 
            identity
        ),
        mlp_layers...,
        x -> reshape(x, 3, prediction_steps, :)
    )
end

function create_optimizer(model; η = 1e-3)
    Flux.setup(
        Flux.OptimiserChain(
            WeightDecay(), 
            Flux.Adam(η)
        ), 
        model
    )
end