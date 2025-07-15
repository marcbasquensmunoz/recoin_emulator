using Flux
using Flux: normalise
using Makie
using CairoMakie
#include("../generate_simulated_data.jl")

prediction_steps = 12
model_save_file = "$(@__DIR__)/current_mf_$(prediction_steps)_steps.jld2"
model_loss_save_file = "$(@__DIR__)/current_mf_loss_$(prediction_steps)_steps.jld2"

# Generate training data
Nt_train = 72                            # Number of previous values in each sample 
N_train = N*(Nt-Nt_train-prediction_steps)  # Number of training samples

train_range = 1:Int(0.8*N_train)
test_range = Int(0.8*N_train):N_train

X_l = zeros(Float32, 3, Nt_train, N_train)                # Input LSTM
X_m = zeros(Float32, prediction_steps, N_train)           # Input mf
Y = zeros(Float32, 3, prediction_steps, N_train)          # Output

# Rolling window sampling
i = 1
for j in 1:N
    for k in 1:Nt-Nt_train-prediction_steps
        @views norm_Tin = normalise(Tin_synth[k:k+Nt_train+prediction_steps-1, j])
        @views norm_Tb = normalise(Tb_synth[k:k+Nt_train+prediction_steps-1, j])
        @views norm_Q = normalise(Q_synth[k:k+Nt_train+prediction_steps-1, j])
        @views @. X_l[1, :, i] = Float32.(norm_Tin[1:Nt_train])
        @views @. X_l[2, :, i] = Float32.(norm_Tb[1:Nt_train])
        @views @. X_l[3, :, i] = Float32.(norm_Q[1:Nt_train])
        @views @. X_m[:, i] = Float32.(mf_synth[k+Nt_train+1:k+Nt_train+prediction_steps])
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
hidden_size = 30
model = Chain(
    Parallel(
        (x, y) -> vcat(x[:, end, :], y),
        Chain(
            LSTM(3 => hidden_size),
            LSTM(hidden_size => hidden_size),
        ), 
        identity
    ),
    Dense(hidden_size + prediction_steps => 3*prediction_steps)
)
################################
# Load the model
################################
# Flux.loadmodel!(model, JLD2.load(model_save_file, "model_state"))
# @load model_loss_save_file loss_train loss_test

loss(y_hat, y) = Flux.mse(y_hat, y)
opt_state = Flux.setup(
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam(1e-3))
    , model
)

function evaluate(model, input)
    Flux.reset!(model.layers[1])
    reshape(model(input), 3, prediction_steps, :)
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

@time for epoch in 1:5
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
# Prediction (from the test set)
################################
K = N-26

@views Tin_original = Tin_synth[:, K:K]'
@views Tb_original = Tb_synth[:, K:K]'
@views Q_original = Q_synth[:, K:K]'
@views mf_original = mf_synth[:]'
input = vcat(Tin_original, Tb_original, Q_original)

# Using the truth data to predict each time step
prediction = zeros(3, prediction_steps)

@views input_borehole = input[:, 1:Nt_train]
input_control = mf_original[Nt_train+1:Nt_train+prediction_steps]
@views μ_input = mean(input_borehole, dims=2)[1:3] * ones(prediction_steps)'
@views σ_input = std(input_borehole, dims=2)[1:3] * ones(prediction_steps)'
@views prediction .= μ_input .+ evaluate(model, (normalise(input_borehole, dims=2), input_control))[:,:,end] .* σ_input


fig_example = Figure(size=(600, 800))
t_axis_truth = (1:Nt_train+prediction_steps) ./ 24
t_axis_pred = (Nt_train+1:Nt_train+prediction_steps) ./ 24
ax_Tin = Axis(fig_example[1, 1], xlabel = L"t \ (days)", ylabel = L"T_{in} \ (K)", title = "Prediction of the next $(prediction_steps) hours")
lines!(ax_Tin, t_axis_truth, Tin_original[1,1:Nt_train+prediction_steps], label="Truth")
lines!(ax_Tin, t_axis_pred, prediction[1,:], label="Prediction")

ax_Tb = Axis(fig_example[2, 1], xlabel = L"t \ (days)", ylabel = L"T_{b} \ (K)")
lines!(ax_Tb, t_axis_truth, Tb_original[1,1:Nt_train+prediction_steps], label="Truth")
lines!(ax_Tb, t_axis_pred, prediction[2,:], label="Prediction")

axislegend(""; position= :rt, backgroundcolor = (:grey90, 0.25));

ax_Q = Axis(fig_example[3, 1], xlabel = L"t \ (days)", ylabel = L"Q \ (W/m)")
lines!(ax_Q, t_axis_truth, Q_original[1,1:Nt_train+prediction_steps], label="Truth")
lines!(ax_Q, t_axis_pred, prediction[3,:], label="Prediction")

linkxaxes!(ax_Tin, ax_Tb, ax_Q)
hidexdecorations!(ax_Tin)
hidexdecorations!(ax_Tb)
fig_example

# Save the figure
save("current_mf_$(prediction_steps)_steps_example_3.png", fig_example, px_per_unit=2.0)