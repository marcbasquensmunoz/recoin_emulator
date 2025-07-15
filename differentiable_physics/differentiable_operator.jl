using LinearAlgebra
using Zygote
using Flux
include("lausen_simulation.jl")

g = options.method.g
M = containers.M

function get_past_influence(q)
    t = size(q, 2)
    if t == 1
        return zeros(Nb)
    end
    sum([(g[:, :, t - k + 1] - g[:, :, t - k])' * q[:, k] for k in 1:t-1])
end

function features_from_state(state)
    @views dT = state[1:2:2Nb, :] - state[2:2:2Nb, :] 
    @views q = sum(state[3Nb+1:4Nb, :], dims=1)
    @views Tb = state[2Nb+1:3Nb, :]

    vcat(dT, Tb, q)
end

function P(x, q, Q)
    b_g = get_past_influence(q)
    b = [zeros(2Nb-1); Q; -T0*ones(Nb) - b_g; zeros(Nb)]

    state_new = M\b
    q_new = cat(q, state_new[3Nb+1:4Nb], dims=2)

    return state_new, q_new
end
####
# Simulate the non-history method
x = zeros(4*Nb)
q = zeros(Nb, 1)
for i in 1:Nt
    x, q = P(x, q, options.constraint.Q_tot[i])
end
containers.X[:, Nt] - x
####
# Check differentiability
jacobian(x -> P(x, zeros(Nb, 1), options.constraint.Q_tot[1])[1], x)
####

state_dim = 1
hidden_dim = 10
model = Chain(
  Dense(state_dim, hidden_dim, relu),
  Dense(hidden_dim, state_dim)
)

function evaluate(model, input)
    @views state = input[1:end-T, :]
    @views force = input[end-T+1:end, :]

    batch_N = size(input, 2)
    q = zeros(Nb, 1, batch_N)
    for i in 1:T
        force_t = force[i:i, :]
        u = model(force_t) 
        #@views corrected_state = u[1:end-T, :]
        @views corrected_force = u[end-T+1:end, :]
        results = [P(state[:, k], q[:, :, k], corrected_force[k]) for k in 1:batch_N]
        hidden_state = hcat(map(r->r[1], results)...)
        state = features_from_state(hidden_state)
        q = cat(map(r->reshape(r[2], Nb, size(r[2], 2), 1), results)..., dims=3)
        #state = corrected_state
    end
    return state
end

####
# Define the loss function
loss(y_hat, y) = Flux.mse(y_hat, y)

function estimate_loss(model, data)
    Ndata = size(data.data[1])[end]
    perm = rand(1:Ndata, 1)
    @views result = evaluate(model, data.data[1][:,perm])
    @views output = data.data[2][:,perm]
    loss(result, output)
end

####
# Define the optimizer
opt_state = Flux.setup(
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam(1e-4))
    , model
)
# Flux.adjust!(opt_state, η = 1e-4)

# J = jacobian(x -> loss(evaluate(model, ), output), input)


####
# Training
loss_train = zeros(0)
loss_test = zeros(0)

Flux.trainmode!(model)
@time for epoch in 1:10
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
        if count % 2 == 0
            push!(loss_train, estimate_loss(model, data))
            push!(loss_test, estimate_loss(model, test_data))
        end
    end
end

plot(log10.(loss_train))

### Testing
input, output = first(data)

input_state = input[1:end-T, :]
input_q = input[end-T+1:end, :]

test_model = Chain(
  Dense(state_dim, state_dim),
)

test_model(input_q)
res, q_res = P(input_state, zeros(3), input_q)
evaluate(model, input)
features = features_from_state(res)