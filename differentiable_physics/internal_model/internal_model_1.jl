using Flux
using Flux: Scale

# input = (Tin_i, Tout_i, Tb_i)
# output = (ain_i, Tb_corrected)

n = 100


#=model_1 = Chain(
    Parallel(vcat, 
        Chain(
            x -> x[1:6, :],
            Dense(6 => n, relu),
            Dense(n => 3, σ)  
        ),
        Chain( 
            x -> x[7:9, :],
            Scale(3)
        )
    )
)=#

model_1 = Chain(
    Parallel(vcat, 
        x -> x[1:6, :],
        Chain(
            x -> x[7:9, :],
            Scale(3)
        )
    ),
    Parallel(vcat,
        Chain(
            Dense(9 => n, relu),
            Dense(n => 3, σ)  
        ),
        x -> x[7:9, :]
    )
)

opt_state = Flux.setup( 
    Flux.OptimiserChain(
        #WeightDecay(), 
        Flux.Adam())
    , model_1
)
#Optimisers.freeze!(opt_state.layers[1].layers[2].layers[2].scale)
Optimisers.freeze!(opt_state.layers[1].layers[2].layers[2].scale)
#Flux.adjust!(opt_state, η=1e-3)



function loss_1(model, x, α = 0.1) 
    y_hat = model(x)
    
    @views a_in = y_hat[1:3,:] 
    @views Tb = y_hat[4:6,:]

    @views Tfin = x[1:3, :] 
    @views Tfout = x[4:6, :] 
    
    internal_eqs = Flux.mse(Tfout, a_in .* Tfin + (1 .- a_in) .* Tb) 

    not_smooth_penalty = sum(abs2, std(a_in))#sum(abs.(a_in[2:end]-a_in[1:end-1]))

    @show internal_eqs, not_smooth_penalty
    return (1 - α) * internal_eqs + α * not_smooth_penalty
end

####
# Training
loss_train = zeros(0)

Flux.trainmode!(model_1)
@time for epoch in 1:10000
    if epoch % 1000 == 0 
        @info "Epoch $(epoch)"
    end
    for batch in data_1
        input = batch
        grads = Flux.gradient(model_1) do m
            loss_1(m, input)
        end
        Flux.update!(opt_state, model_1, grads[1])
    end
    if epoch%100 == 0
        push!(loss_train, loss_1(model_1, data_1.data))
    end
end
lines(log10.(loss_train))


#####
# Check results
result = model_1(data_1.data)
Tfin_ref = data_1.data[1:3, :]
Tfout_ref = data_1.data[4:6, :]
a_in_pred = result[1:3, :]
Tb_pred = result[4:6, :]


fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, Tfin_ref[1, :], label="Tfin")
lines!(ax, Tfout_ref[1, :], label="Tfout")
lines!(ax, Tb_pred[1, :], label="Corrected Tb")
lines!(ax, Tb_ref, label="Analytical Tb")
axislegend(position = :rt)
fig


fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, a_in_pred[1, :], label="Fitted")
lines!(ax, a_in_ref, label="Analytical")
axislegend(position = :rt)
fig
