using Flux
using Flux: mse

H = 145.
z = collect(0:0.1:H)'

#=
##################################
# Construct NN models for the Td, Tu data
##################################
k = 1
Td_ref = T_down_z[:, k]
Tu_ref = T_up_z[:, k]
data_d = Flux.DataLoader((z_down', Td_ref'), batchsize=length(z_down), shuffle=true)
data_u = Flux.DataLoader((z_up', Tu_ref'), batchsize=length(z_up), shuffle=true)

n_hidden = 10
T_d_model = Chain(
    Dense(1 => n_hidden, tanh),
    Dense(n_hidden => n_hidden, tanh),
    Dense(n_hidden => 1)
)
T_u_model = Chain(
    Dense(1 => n_hidden, tanh),
    Dense(n_hidden => n_hidden, tanh),
    Dense(n_hidden => 1)
)

T_model = (T_d_model, T_u_model)

opt_state = Flux.setup( 
    Flux.OptimiserChain(
        WeightDecay(1e-2), 
        SignDecay(),
        Flux.Adam(1e-4),
    ), T_model
)

loss_train_T = zeros(0)

@time for epoch in 1:40000
    zd, Tdz = first(data_d)
    zu, Tuz = first(data_u)

    lossval, grads = Flux.withgradient(T_model) do m
        Td_pred = m[1](zd)
        Tu_pred = m[2](zu)
        mse(Td_pred, Tdz) + mse(Tu_pred, Tuz) + mse(m[1]([H]), m[2]([H]))
    end
    Flux.update!(opt_state, T_model, grads[1])
    push!(loss_train_T, lossval)
end

lines(log10.(loss_train_T))


##################################
# Reconstruction of Td, Tu
##################################
Td_pred = vec(T_model[1](z))
Tu_pred = vec(T_model[2](z))

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, Td_pred, z', label="Td pred")
lines!(ax, Tu_pred, z', label="Tu pred")
lines!(ax, Td_ref, z_down, label="Td data")
lines!(ax, Tu_ref, z_up, label="Tu data")
axislegend(""; position= :rb)
fig
=#
#=
##################################
# Physical model
##################################
# T is the vector [Td, Tu]
## dT/dz(z) = A T(z) - A 1 Tb(z)
function loss_eq(Tb, A, T, dT)
    expected_A_sign = [-1 -1; 1 1]
    sign_penalization = sum(Int.(sign.(A)) .!= expected_A_sign)
    loss = 0.
    for i in 1:length(T)
        @views dT_i = dT[:,:,i]
        @views T_i = T[:,:,i]
        loss += Flux.mse(dT_i, A * T_i - A * ones(2) * Tb)
    end
    return loss + 10 * sign_penalization
end

#T(z) = [T_d_model([z])[1], T_u_model([z])[1]]
#dT(z) = [gradient(z -> T_d_model([z])[1], z)[1], gradient(z -> T_u_model([z])[1], z)[1]]

m_hidden = 100
Tb_model = Chain(
    Dense(1 => m_hidden, tanh),
    Dense(m_hidden => m_hidden, tanh),
    Dense(m_hidden => 1)
)

#A = vcat(-rand(2)', rand(2)')         
A = [.1 0.001; 0.001 .1]
model = (Tb_model, A)

opt_state_eq = Flux.setup( 
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam()
    ), 
    model
)

loss_train_eq = zeros(0)
#T_train = hcat(T.(z)...)
#dT_train = hcat(dT.(z)...)
@time for epoch in 1:5000
    lossval, grads = Flux.withgradient(model) do m
        m_Tb = m[1]
        #A = m[2]
        Tb = m_Tb(z)
        loss_eq(Tb, A, T_train, dT_train)
    end
    Flux.update!(opt_state_eq, model, grads[1])
    push!(loss_train_eq, lossval)
end
lines(log10.(loss_train_eq))


##################################
# Reconstruction
##################################
z = collect(0:0.1:145)'
Tdz = T_train[1, :]
Tuz = T_train[2, :]
Tb_pred = vec(Tb_model(z))

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, Tdz, z', label="Td")
lines!(ax, Tuz, z', label="Tu")
lines!(ax, Tb_pred, z', label="Tb pred")
axislegend(""; position= :rb)
fig
=#


###################################################
# Use avergaged data to train the internal model
###################################################

function temperature_gradient(T, z)
    n = length(T)
    dTdz = zeros(Float64, n)

    for i in 2:n-1
        dz = z[i+1] - z[i-1]
        dTdz[i] = (T[i+1] - T[i-1]) / dz
    end

    dTdz[1] = (T[2] - T[1]) / (z[2] - z[1])
    dTdz[end] = (T[end] - T[end-1]) / (z[end] - z[end-1])

    return dTdz
end

N_train = size(smooth_T_down_z, 2)
Tdz = smooth_T_down_z
Tuz = smooth_T_up_z
dTdz = hcat([temperature_gradient(smooth_T_down_z[:, k], z_down) for k in 1:N_train]...)
dTuz = hcat([temperature_gradient(smooth_T_up_z[:, k], z_up) for k in 1:N_train]...)

T_train = zeros(2, length(z_down), N_train)
dT_train = zeros(2, length(z_down), N_train)

for i in 1:N_train
    T_train[:, :, i] .= hcat(Tdz[:, i], vcat(Tuz[:, i], Tdz[end, i]))'
    dT_train[:, :, i] .= hcat(dTdz[:, i], vcat(dTuz[:, i], dTdz[end, i]))' 
end

function loss_eq(Tb, A, T, dT)
    loss = 0.
    for i in 1:size(T, 2)
        @views dT_i = dT[:, i]
        @views T_i = T[:, i]
        @views A_i = A[:, :, i]
        @views Tb_i = Tb[i]
        loss += Flux.mse(dT_i, A_i*T_i - A_i*ones(2) .* Tb[i]) 
    end
    return loss
end

m_hidden = 100
model = Chain(
    Dense(1+2 => m_hidden, tanh),
    Dense(m_hidden => m_hidden, tanh),
    Dense(m_hidden => 1+4),
    x -> vcat(x[1:1, :], sigmoid.(x[2:5, :]) .* [-1, 1, -1, 1])
) 

opt_state_eq = Flux.setup( 
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam()
    ), 
    model
)

loss_train_eq = zeros(0)
@time for epoch in 1:200
    @show epoch
    for i in 1:N_train
        @views T_data = T_train[:, :, i]
        @views dT_data = dT_train[:, :, i]

        lossval, grads = Flux.withgradient(model) do m
            res = m(vcat(z_down', T_data))
            @views Tb = res[1, :]
            @views A = reshape(res[2:end, :], 2, 2, size(res, 2))
            loss_eq(Tb, A, T_data, dT_data)
        end
        Flux.update!(opt_state_eq, model, grads[1])
        push!(loss_train_eq, lossval)
    end
end
lines(log10.(loss_train_eq))


##################################
# Reconstruction
##################################
k = 50
z = collect(0:0.1:145)'
Tdz = T_train[1, :, k]
Tuz = T_train[2, :, k]
test_input = hcat(z_down, Tdz, Tuz)'
prediction = model(test_input)
@views Tb_pred = prediction[1,:]
@views A_pred = reshape(prediction[2:end, :], 2, 2, size(prediction, 2))

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, Tdz, -z_down, label="Td")
lines!(ax, Tuz, -z_down, label="Tu")
lines!(ax, Tb_pred, -z_down, label="Tb pred")
axislegend(""; position= :lb)
fig
