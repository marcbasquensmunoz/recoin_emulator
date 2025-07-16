include("initialize.jl")

prediction_steps = 12           # How many time steps to predict in the future
Nt_train = 72                   # How many previous time steps to use for prediction

# Data preparation
include("../../../data/src/synthetic/generate_simulated_data.jl")

data = zeros(Float32, 4, Nt, N)  # Real data

data[1, :, :] .= Tin_synth
data[2, :, :] .= Tb_synth
data[3, :, :] .= Q_synth
data[4, :, :] .= mf_synth

train_data, test_data = prepare_data(data, Nt_train, prediction_steps)

# Model definition
model = create_model(lstm_hidden = [30, 30], mlp_hidden = [50, 50], prediction_steps=prediction_steps) 
opt_state = create_optimizer(model)

loss(y_hat, y) = Flux.mse(y_hat, y)

# Training
loss_train = Float64[]
loss_test = Float64[]
epochs = 1
train!(model, opt_state, train_data, test_data, epochs, loss_train, loss_test)

plot_training_loss(loss_train, loss_test)

# Test predictions
if prediction_steps == 1
    # TODO: Execute the prediction many times, just for the plo
else
    plot_prediction(model, test_data)
end
