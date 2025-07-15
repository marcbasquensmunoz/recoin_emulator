using Flux
using Flux: Scale

# input = (Tin_i, Tout_i, Tb_i, Q_T, mf_T)
# output = (ain_i, cp*mf_i)

n = 100
model_2 = Chain(
    Parallel(vcat, 
        Chain(
            Dense(11 => n, relu),
            Parallel(vcat,
                Dense(n => 3, σ),
                Dense(n => 3)
            )
        ),
        Chain( 
            x -> x[7:9, :],
            Scale(3)
        )
    )
)
opt_state = Flux.setup( 
    Flux.OptimiserChain(
        WeightDecay(), 
        Flux.Adam())
    , model_2
)
#Flux.adjust!(opt_state, η=1e-3)

Optimisers.freeze!(opt_state.layers[1].layers[2].layers[2].scale)


function loss_2(model, x, λ1=1., λ2=0.1) 
    y_hat = model(x)
    
    @views a_in = y_hat[1:3,:]     
    @views k = y_hat[4:6,:] 
    @views Tb = y_hat[7:9,:]

    @views Tfin = x[1:3, :] 
    @views Tfout = x[4:6, :] 
    
    @views Q_T = x[10, :] 
    @views mf_T = x[11, :] 

    internal_model_eqs = Flux.mse(Tfout, a_in .* Tfin + (1 .- a_in) .* Tb) 
    energy_balance_eqs = Flux.mse(Q_T, vec(sum(k .* (Tfout - Tfin), dims=1)))
    return λ1 * internal_model_eqs + λ2 * energy_balance_eqs
end

####
# Training
loss_train = zeros(0)
loss_test = zeros(0)

Flux.trainmode!(model_2)
@time for epoch in 1:10000
    if epoch % 1000 == 0 
        @info "Epoch $(epoch)"
    end
    for batch in data_2
        input = batch
        grads = Flux.gradient(model_2) do m
            loss_2(m, input)
        end
        Flux.update!(opt_state, model_2, grads[1])
    end
    if epoch%100 == 0
        push!(loss_train, loss_2(model_2, data_2.data))
    end
end
lines(log10.(loss_train))


#####
# Check results
result = model_2(data_2.data)
Tfin_ref = data_2.data[1:3, :]
Tfout_ref = data_2.data[4:6, :]
a_in_pred = result[1:3, :]
Tb_pred = result[4:6, :]



fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, Tfin_ref[1, :], label="Tfin")
lines!(ax, Tfout_ref[1, :], label="Tfout")
lines!(ax, Tb_pred[1, :], label="Corrected Tb")
axislegend(position = :rt)
fig


fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, a_in_pred[1, :], label="Fitted")
lines!(ax, a_in_ref, label="Analytical")
axislegend(position = :rt)
fig
