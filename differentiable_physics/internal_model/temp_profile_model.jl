using Flux
using Flux: mse

H = 145.
z = collect(0:0.1:H)'
Δz = 1.2

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

#N_train = size(smooth_T_down_z, 2)
#Tdz = smooth_T_down_z
#Tuz = smooth_T_up_z
#Q_train = smooth_Q_train
N_train = size(T_down_z, 2)
Tdz = T_down_z
Tuz = T_up_z
Q_train = Q_data

dTdz = hcat([temperature_gradient(Tdz[:, k], z_down) for k in 1:N_train]...)
dTuz = hcat([temperature_gradient(Tuz[:, k], z_up) for k in 1:N_train]...)

T_train = zeros(2, length(z_down), N_train)
dT_train = zeros(2, length(z_down), N_train)

for i in 1:N_train
    T_train[:, :, i] .= hcat(Tdz[:, i], vcat(Tuz[:, i], Tdz[end, i]))'
    dT_train[:, :, i] .= hcat(dTdz[:, i], vcat(dTuz[:, i], dTdz[end, i]))' 
end

#=
function loss_eq(Tb, A, T, dT, Q)
    λ1 = 1.
    λ2 = 0.01
    loss = 0.
    eq_1_res = 0.
    Q_calc = zeros(Float64, 2)
    for i in 1:size(T, 2)
        @views dT_i = dT[:, i]
        @views T_i = T[:, i]
        @views A_i = A[:, :, i]
        @views Tb_i = Tb[i]
        eq_1_res += Flux.mse(dT_i, A_i*T_i - A_i*ones(2) .* Tb_i) 

        # q = R^-1 Tf(z) - R^-1 * 1  Tb(z)
        Rinv_i = (mf*cp) .* [-1 0; 0 1] * A_i
        Q_calc += Δz * (Rinv_i * T_i - Rinv_i * ones(2) * Tb_i)
    end
    loss += λ1 * eq_1_res
    #loss += λ2 * Flux.mse(Q, Q_calc[1] - Q_calc[2])
    return loss
end
=#

barrier_term(x) = relu(-x)^2
penalty_term(f) = min(f, 0.)^2
augemented_loss(f, λ = 1., ρ = 1.) = λ * f + ρ * min(f, 0.)^2

function loss_eq(Tb, A, T, dT, Q, μ)
    NB = size(T, 2)
    λ1 = 1.
    λ2 = 1e-6
    λ3 = 1e-1
    λ4 = 1e-2
    λ5 = 1e-2
    λ6 = 1e-5

    dT_residual = 0.
    dT_norm = 0.
    Q_calc = 0.
    for i in 1:NB
        @views dT_i = dT[:, i]
        @views T_i = T[:, i]
        @views A_i = A[:, :, i]
        @views Tb_i = Tb[i]
        dT_residual += Flux.mse(dT_i, A_i * (T_i - Tb_i .* ones(2)))
        dT_norm += sum(abs2.(A_i * (T_i - Tb_i .* ones(2))))
        Rinv_i = (mf * cp) .* [-1 0; 0 1] * A_i

        Q_calc_point_i = Rinv_i * (T_i - Tb_i .* ones(2))
        Q_calc += (Q_calc_point_i[1] - Q_calc_point_i[2]) * Δz * (i == 1 || i == NB ? 0.5 : 1.0) # Trapezoidal quadrature rule
    end
    Q_residual = Flux.mse(Q, Q_calc)

    #Tb_high = μ * barrier_term(sum(Tb .- max.(T[1,:], T[2,:])))
    Tb_high = penalty_term(sum(Tb .- T[1, :])) + penalty_term(sum(Tb .- T[2, :]))

    A_stability = penalty_term(-sum(A[1, 1, :] .+ A[1, 2, :])) + penalty_term(sum(A[2, 1, :] .+ A[2, 2, :]))
    #A_stability = μ * barrier_term(-sum(A[1, 1, :] .+ A[1, 2, :])) + μ * barrier_term(sum(A[2, 1, :] .+ A[2, 2, :]))

    A_big = sum([sum(1 ./ abs.(A[:, :, k])) for k in 1:NB])

    #@show dT_residual, Q_residual, dT_norm, Tb_high, A_stability, A_big
    return λ1 * dT_residual + λ2 * Q_residual + λ3 * dT_norm + λ4 * Tb_high + λ5 * A_stability + λ6 * A_big
end

m_hidden = 20
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
@time for epoch in 1:20
    @show epoch
    for i in 1:N_train
        @views T_data = T_train[:, :, i]
        @views dT_data = dT_train[:, :, i]
        Q_data_samp = Q_train[i]

        lossval, grads = Flux.withgradient(model) do m
            res = m(vcat(z_down', T_data))
            @views Tb = res[1, :]
            @views A = reshape(res[2:end, :], 2, 2, size(res, 2))
            loss_eq(Tb, A, T_data, dT_data, Q_data_samp, μ)
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
ax = Axis(fig[1, 1], title="Physics-informed internal model", xlabel=L"T  \ [^\circ C]", ylabel=L"z \ [m]")
lines!(ax, Tdz, -z_down, label=L"T_\text{down}")
lines!(ax, Tuz, -z_down, label=L"T_\text{up}")
lines!(ax, Tb_pred, -z_down, label=L"T_\text{b} \text{ predidcted}")
axislegend(""; position= :lb)
fig

# save("$(@__DIR__)/pinn_internal_model.png", fig)

### Check residuals
sum([abs.(dT_train[:, l, k] .- A_pred[:,:,k] * (T_train[:,l,k] - ones(2) .* Tb_pred[l])) for l in 1:length(z_down)])

Q_pred = 0.
for i in 1:120
    Rinv_pred = mf * cp .* [-1 0; 0 1] * A_pred[:,:,i]
    Q_pred_i = Rinv_pred * (T_train[:,i,k] - ones(2) .* Tb_pred[i])
    Q_pred += (Q_pred_i[1] - Q_pred_i[2]) * Δz * (i == 1 || i == N_train ? 0.5 : 1.0) # Trapezoidal quadrature rule
end
Q_pred - Q_train[1]
