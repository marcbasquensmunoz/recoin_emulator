
function train!(model, opt_state, train_data, test_data, epochs, loss_train, loss_test)
    @time for epoch in 1:epochs
        @info "Epoch $(epoch)"
        n_batches = length(train_data)
        for (i, batch) in enumerate(train_data)
            @info "Progess $(round(i / n_batches * 100, digits=2))%"
            input, output = batch
            grads = Flux.gradient(model) do m
                result = m(input)
                loss(result, output)
            end
            Flux.update!(opt_state, model, grads[1])
            if (i-1) % Int(floor(n_batches / 20)) == 0
                push!(loss_train, estimate_loss(model, train_data))
                push!(loss_test, estimate_loss(model, test_data))
            end
        end
    end
end

function estimate_loss(model, data)
    state, control = data.data[1]
    Ndata = size(state)[end]
    perm = rand(1:Ndata, 1000)
    @views result = model((state[:,:,perm], control[:,perm]))
    @views output = data.data[2][:,:,perm]
    loss(result, output)
end
