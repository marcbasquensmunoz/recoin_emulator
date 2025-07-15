using Flux
using Flux: normalise
using Makie
using CairoMakie

prediction_steps = 1

normalize_min(x) = (x .- minimum(x, dims=2)) ./ (maximum(x, dims=2) .- minimum(x, dims=2))
normalise_data(data) = normalize_min(data), minimum(data, dims=2), maximum(data, dims=2)
denormalise_data(data, m, M) = hcat([@. data[:, i] .* (M - m) .+ m for i in 1:size(data, 2)]...)

file = "$(@__DIR__)/../lausen_data_fetching/lausen_data.csv"
raw_data = CSV.read(file, DataFrame; delim = ',')

Nt_real = length(raw_data.t)
data_real = zeros(4, Nt_real)
ρf = 1051.2 

data_real[1, :] .= raw_data[!, :Tfin1]
data_real[2, :] .= raw_data[!, :meanTb1]
data_real[3, :] .= raw_data[!, :q_all] ./ 3
data_real[4, :] .= raw_data[!, :flowRate] .* ρf / 3600 / 3 

N_real = 1  # Number of real data series samples
data_real = reshape(data_real, 4, Nt_real, N_real)

# Generate training data
Nt_train = 24                         # Number of previous values in each sample 
N_train_real = N_real*(Nt_real-Nt_train-prediction_steps)  # Number of real training samples 

# Take 80% of both simulated and real data sets for training
train_range = 1:Int(floor(0.8*N_train_real))
test_range = Int(floor(0.8*N_train_real))+1:N_train_real

X_l = zeros(Float32, 4, Nt_train, N_train_real)                # Input LSTM
X_m = zeros(Float32, prediction_steps, N_train_real)           # Input mf
Y = zeros(Float32, 3, prediction_steps, N_train_real)          # Output

# Rolling window sampling
i = 1
for j in 1:N_real
    for k in 1:Nt_real-Nt_train-prediction_steps
        # Includes training sample and target
        data_window = k:k+Nt_train+prediction_steps-1

        normalized_data, m, M = normalise_data(data_real[:, data_window, j])

        # Normalize the windowed data
        @views norm_Tin = normalized_data[1, :]
        @views norm_Tb  = normalized_data[2, :]
        @views norm_Q   = normalized_data[3, :]
        @views norm_mf  = normalized_data[4, :]

        # Input
        @views @. X_l[1, :, i] = Float32.(norm_Tin[1:Nt_train])
        @views @. X_l[2, :, i] = Float32.(norm_Tb[1:Nt_train])
        @views @. X_l[3, :, i] = Float32.(norm_Q[1:Nt_train])
        @views @. X_l[4, :, i] = Float32.(norm_mf[1:Nt_train])
        @views @. X_m[:, i] =    Float32.(norm_mf[Nt_train+1:end])

        # Target
        Y[1, :, i] .= Float32.(norm_Tin[Nt_train+1:end])
        Y[2, :, i] .= Float32.(norm_Tb[Nt_train+1:end])
        Y[3, :, i] .= Float32.(norm_Q[Nt_train+1:end])
        i += 1
    end
end

# Data loaders declaration
batchsize = 512
data      = Flux.DataLoader(((X_l[:,:, train_range], X_m[:, train_range]), Y[:, :, train_range]), batchsize=batchsize, shuffle=true)
test_data = Flux.DataLoader(((X_l[:,:, test_range],  X_m[:, test_range]),  Y[:, :, test_range]),  batchsize=32,        shuffle=true)

# Model
hidden_size = 20
mlp_hidden_size = 50
model = Chain(
    Parallel(
        (x, y) -> vcat(x[:, end, :], y),
        Chain(
            LSTM(4 => hidden_size),
            LSTM(hidden_size => hidden_size),
        ), 
        identity
    ),
    Dense(hidden_size + prediction_steps => mlp_hidden_size, tanh),
    Dense(mlp_hidden_size => mlp_hidden_size, tanh),
    Dense(mlp_hidden_size => 3*prediction_steps)
)

################################
# Load the model
################################
# Flux.loadmodel!(model, JLD2.load(model_save_file, "model_state"))
# @load model_loss_save_file loss_train loss_test

function loss(y_hat, y) 
    Flux.mse(y_hat, y)
end

opt_state = Flux.setup(
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam(1e-3))
    , model
)

function evaluate(model, input)
    Flux.reset!(model.layers[1].layers[1].layers[1])
    Flux.reset!(model.layers[1].layers[1].layers[2])
    output = reshape(model(input), 3, prediction_steps, :)
    return vcat(output[1:2, : ,:], relu(output[3:3, :, :])) 
end

function estimate_loss(model, data)
    borefield, control = data.data[1]
    Ndata = size(borefield)[end]
    perm = rand(1:Ndata, 1000)
    @views result = evaluate(model, (borefield[:,:,perm], control[:,perm]))
    @views output = data.data[2][:,:,perm]
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
        push!(loss_train, estimate_loss(model, data))
        push!(loss_test, estimate_loss(model, test_data))
    end
end

################################
# Save the model
################################
# JLD2.jldsave(model_save_file, model_state = Flux.state(model))
# @save model_loss_save_file loss_train loss_test


# Loss plot
fig_loss = Figure()
ax = Axis(fig_loss[1, 1], xlabel = "Training step", ylabel =  L"\log_{10} L", title="Loss evolution during training")
lines!(ax, log10.(loss_train), label="Train")
lines!(ax, log10.(loss_test), label="Test")
axislegend(""; position= :rt, backgroundcolor = (:grey90, 0.25));
fig_loss


################################
# Prediction 1 step at a time (from the test set)
################################
K1 = 100
NK = 20
J = 1

full_prediction = zeros(3, NK+1)
i = 1
for K in test_range[K1]:test_range[K1+NK]
    all_timesteps = K:K+Nt_train+prediction_steps-1
    input_timesteps = K:K+Nt_train-1
    predicted_timesteps = K+Nt_train:K+Nt_train+prediction_steps-1

    normalised_data, m, M = normalise_data(data_real[:, all_timesteps, J])

    input_bh_state = normalised_data[:, 1:end-prediction_steps]
    input_mf = normalised_data[4, end-prediction_steps+1:end]

    @views prediction = denormalise_data(evaluate(model, (input_bh_state, input_mf))[:,:,end], m[1:3], M[1:3])
    full_prediction[:, i] .= prediction
    i += 1
end

prediction_color = Makie.wong_colors()[2]
plot_timesteps = test_range[K1]:test_range[K1+NK]+Nt_train+prediction_steps-1
plot_prediction = test_range[K1]+Nt_train:test_range[K1+NK]+Nt_train+prediction_steps-1
plot_t_timesteps = (plot_timesteps .- plot_timesteps.start) ./ 3 
plot_t_prediction = (plot_prediction .- plot_timesteps.start) ./ 3

fig_example = Figure(size=(600, 800))
ax_Tin = Axis(fig_example[1, 1], xlabel = L"t \ (h)", ylabel = L"T_{in} \ (K)", title = "Prediction of the next 20 min")
lines!(ax_Tin, plot_t_timesteps, data_real[1, plot_timesteps, J], label="Truth")
plot!(ax_Tin, plot_t_prediction, full_prediction[1,:], label="Prediction", color=prediction_color)
axislegend(""; position= :lt, backgroundcolor = (:grey90, 0.25));

ax_Tb = Axis(fig_example[2, 1], xlabel = L"t \ (h)", ylabel = L"T_{b} \ (K)")
lines!(ax_Tb, plot_t_timesteps, data_real[2, plot_timesteps, J], label="Truth")
plot!(ax_Tb, plot_t_prediction, full_prediction[2,:], label="Prediction", color=prediction_color)

ax_Q = Axis(fig_example[3, 1], xlabel = L"t \ (h)", ylabel = L"Q \ (W/m)")
lines!(ax_Q, plot_t_timesteps, data_real[3, plot_timesteps, J], label="Truth")
plot!(ax_Q, plot_t_prediction, full_prediction[3,:], label="Prediction", color=prediction_color)

ax_mf = Axis(fig_example[4, 1], xlabel = L"t \ (h)", ylabel = L"\dot{m}_f \ (kg/m^3)")
lines!(ax_mf, plot_t_timesteps, data_real[4, plot_timesteps, J], label="Truth")

linkxaxes!(ax_Tin, ax_Tb, ax_Q, ax_mf)
hidexdecorations!(ax_Tin)
hidexdecorations!(ax_Tb)
hidexdecorations!(ax_Q)
fig_example

# save("$(@__DIR__)/mf_control_$(prediction_steps)_steps_example_3.png", fig_example)



################################
# Prediction N steps  (from the test set)
################################
K = test_range[200]
J = 1

all_timesteps = K:K+Nt_train+prediction_steps-1
input_timesteps = K:K+Nt_train-1
predicted_timesteps = K+Nt_train:K+Nt_train+prediction_steps-1

normalised_data, m, M = normalise_data(data_real[:, all_timesteps, J])
input_bh_state = normalised_data[:, 1:end-prediction_steps]
input_mf = normalised_data[4, end-prediction_steps+1:end]

@views full_prediction = denormalise_data(evaluate(model, (input_bh_state, input_mf))[:,:,end], m[1:3], M[1:3])

prediction_color = Makie.wong_colors()[2]
plot_timesteps = K+Nt_train-20:K+Nt_train+prediction_steps-1
plot_prediction = K+Nt_train:K+Nt_train+prediction_steps-1
plot_t_timesteps = (plot_timesteps .- plot_timesteps.start) ./ 3 
plot_t_prediction = (plot_prediction .- plot_timesteps.start) ./ 3

fig_example = Figure(size=(600, 800))
ax_Tin = Axis(fig_example[1, 1], xlabel = L"t \ (h)", ylabel = L"T_{in} \ (K)", title = "Prediction of the next $(Int(ceil(prediction_steps/3))) hours")
lines!(ax_Tin, plot_t_timesteps, data_real[1, plot_timesteps, J], label="Truth")
plot!(ax_Tin, plot_t_prediction, full_prediction[1,:], label="Prediction", color=prediction_color)
axislegend(""; position= :lt);

ax_Tb = Axis(fig_example[2, 1], xlabel = L"t \ (h)", ylabel = L"T_{b} \ (K)")
lines!(ax_Tb, plot_t_timesteps, data_real[2, plot_timesteps, J], label="Truth")
plot!(ax_Tb, plot_t_prediction, full_prediction[2,:], label="Prediction", color=prediction_color)

ax_Q = Axis(fig_example[3, 1], xlabel = L"t \ (h)", ylabel = L"Q \ (W/m)")
lines!(ax_Q, plot_t_timesteps, data_real[3, plot_timesteps, J], label="Truth")
plot!(ax_Q, plot_t_prediction, full_prediction[3,:], label="Prediction", color=prediction_color)

ax_mf = Axis(fig_example[4, 1], xlabel = L"t \ (h)", ylabel = L"\dot{m}_f \ (kg/m^3)")
lines!(ax_mf, plot_t_timesteps, data_real[4, plot_timesteps, J], label="Truth")

linkxaxes!(ax_Tin, ax_Tb, ax_Q, ax_mf)
hidexdecorations!(ax_Tin)
hidexdecorations!(ax_Tb)
hidexdecorations!(ax_Q)
fig_example

# save("$(@__DIR__)/mf_control_$(prediction_steps)_steps_example_3.png", fig_example)
