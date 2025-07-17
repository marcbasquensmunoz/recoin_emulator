using Random
include("initialize.jl")

use_real_data = true
if use_real_data
    include("prepare_measured_data.jl")
else
    include("prepare_synthetic_data.jl")
end

prediction_steps = 1            # How many time steps to predict in the future
Nt_train = 24                   # How many previous time steps to use for prediction

train_data, test_data = prepare_data(data, Nt_train, prediction_steps)

# Model definition
model = create_model(lstm_hidden = [50, 50], mlp_hidden = [100, 100], prediction_steps=prediction_steps) 
opt_state = create_optimizer(model)

loss(y_hat, y) = Flux.mse(y_hat, y)

# Training
loss_train = Float64[]
loss_test = Float64[]
epochs = 5
train!(model, opt_state, train_data, test_data, epochs, loss_train, loss_test)

plot_training_loss(loss_train, loss_test)

# Test predictions
Random.seed!(123) # For reproducibility
plot_prediction(model, test_data, Nt_train, prediction_steps)
