using LinearAlgebra
using Zygote
using Flux
using Plots
include("lausen_simulation_nh.jl")


T0 = options.medium.T0
expΔt = options.method.expΔt
w = options.method.w
ζ = options.method.ζ
M = containers.M

function P(f, Q)
    @views b_nh = -T0*ones(Nb) - [sum([dot(w[:, Nb*(i-1)+j], expΔt .* f[:, Nb*(i-1)+j]) for j in 1:Nb]) for i in 1:Nb]

    b = [zeros(2Nb-1); Q; b_nh; zeros(Nb)]
    x = M\b
    new_f = [expΔt[j] * f[j, i] + x[3Nb+(i-1)%Nb+1] * (1 - expΔt[j]) / ζ[j] for j in 1:size(f, 1), i in 1:size(f, 2)]

    x, new_f
end
####
# Simulate the non-history method
T = Nt
f = zeros(size(options.method.F))
for i in 1:T
    x, f = P(f, options.constraint.Q_tot[i])
end
containers.X[:, T] - x[1:4Nb]
####
# Check differentiability
jacobian(x -> P(x, options.constraint.Q_tot[1])[1], f)
####


####
# Define the model
input_size = 2
output_size = 1
hidden_size = 20
T_to_q_model = Chain(
    Dense(input_size, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, output_size)
)
input_correction_size = 4Nb
output_correction_size = 2Nb+1
hidden_correction_size = 24
state_correction_model = Chain(
    Dense(input_correction_size, hidden_correction_size, relu),
    Dense(hidden_correction_size, hidden_correction_size, relu),
    Dense(hidden_correction_size, output_correction_size)
)

function evaluate(model_q, model_state, input)
    #q = model_q(input)
    #state = hcat([P(zeros(size(options.method.F)), 1000 * q[1, i])[1] for i in 1:size(q, 2)]...)
    state = hcat([P(zeros(size(options.method.F)), input[1, i])[1] for i in 1:size(input, 2)]...)
    (features_from_state(state) .- μ_Y) ./ σ_Y + model_state(state)
end

### Check differentiability
Flux.gradient((m1, m2) -> norm(evaluate(m1, m2, input)), T_to_q_model, state_correction_model)
###

####
# Define the loss function
loss(y_hat, y) = Flux.mse(y_hat, y)

function features_from_state(state)
    @views dT = state[1:2:2Nb, :] - state[2:2:2Nb, :]  # Tfin - Tfout
    @views q = sum(state[3Nb+1:4Nb, :], dims=1)
    @views Tb = state[2Nb+1:3Nb, :]

    vcat(Tb, dT, q)
end

function estimate_loss(data)
    Ndata = size(data.data[1])[end]
    perm = rand(1:Ndata, 100)
    @views result = evaluate(T_to_q_model, state_correction_model, data.data[1][:,perm])
    @views output = data.data[2][:,perm]
    loss(result, output)
end

####
# Define the optimizer
opt_state = Flux.setup(
    Flux.OptimiserChain(
        #WeightDecay(), 
        Flux.Adam(1e-2))
    , (T_to_q_model, state_correction_model)
)
# Flux.adjust!(opt_state, η = 1e-4)


####
# Training
loss_train = zeros(0)
loss_test = zeros(0)

@time for epoch in 1:3
    @info "Epoch $(epoch)"
    count = 1
    for batch in data
        @info "Processed $(round(100*count*batchsize/length(train_range), digits=2))% data points"
        input, output = batch
        grads = Flux.gradient(T_to_q_model, state_correction_model) do m1, m2
            result = evaluate(m1, m2, input)
            loss(result, output)
        end
        Flux.update!(opt_state, (T_to_q_model,state_correction_model), grads)
        count += 1
        push!(loss_train, estimate_loss(data))
        push!(loss_test, estimate_loss(test_data))
    end
end

plot(log10.(loss_train))
plot(log10.(loss_test))


####################
# Inspection 
####################

data_input, data_output = data.data
k = 10
test_input = data_input[:, k]
test_output = data_output[:, k]

test_res = evaluate(T_to_q_model, state_correction_model, test_input)
loss(test_res, test_output)

#q_test = T_to_q_model(test_input)
state_test, f_test = P(zeros(size(options.method.F)), test_input)
(features_from_state(state_test) .- μ_Y) ./ σ_Y
state_correction_model(state_test)


physical_prediction = test_res .* σ_Y + μ_Y
Y[:,k]