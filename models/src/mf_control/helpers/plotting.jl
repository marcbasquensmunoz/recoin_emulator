
function plot_training_loss(loss_train, loss_test) 
    fig_loss = Figure()
    ax = Axis(fig_loss[1, 1], xlabel = "Training step", ylabel =  L"\log_{10} L", title="Loss evolution during training")
    lines!(ax, log10.(loss_train), label="Train")
    lines!(ax, log10.(loss_test), label="Test")
    axislegend(""; position= :rt, backgroundcolor = (:grey90, 0.25));
    fig_loss
end

function plot_prediction(model, test_data)
    test_input_batch, test_target_batch = rand(collect(test_data))
    test_state_batch, test_control_batch, test_m_batch, test_M_batch = test_input_batch

    K = rand(1:size(test_state_batch, 3))
    @views test_state = test_state_batch[:, :, K]
    @views test_control = test_control_batch[:, K]
    @views test_m = test_m_batch[:, K]
    @views test_M = test_M_batch[:, K]
    @views test_target = test_target_batch[:, :, K]

    result = evaluate(model, (test_state, test_control))[:, :, end]

    @views test_truth = denormalise_data(vcat(hcat(test_state[1:3, :], test_target), hcat(test_state[4:4, :], test_control')), test_m, test_M) 
    @views prediction = denormalise_data(vcat(result, test_control'), test_m, test_M)

    t_axis_truth = 1:Nt_train+prediction_steps
    t_axis_pred = Nt_train+1:Nt_train+prediction_steps
    @views test_mse = Flux.mse(test_truth[1:3, t_axis_pred], prediction[1:3, :])

    y_labels = [L"T_{in} \ (K)", L"T_{b} \ (K)", L"Q \ (W/m)", L"m_f \ (kg/s)"]
    fig_example = Figure(size=(600, 800))
    figure_axes = []
    for (i, ylabel) in enumerate(y_labels)
        ax = Axis(fig_example[i, 1], xlabel = L"t \ (days)", ylabel = ylabel, title = i == 1 ? "Prediction of the next $(prediction_steps) hours; MSE = $(test_mse)" : "")
        push!(figure_axes, ax)
        lines!(ax, t_axis_truth, test_truth[i,:], label="Truth")
        lines!(ax, t_axis_pred, prediction[i,:], label="Prediction")
    end
    axislegend(""; position= :lb, backgroundcolor = (:grey90, 0.25));
    linkxaxes!(figure_axes...)
    for ax in figure_axes[1:end-1]
        hidexdecorations!(ax)
    end
    fig_example
end

function plot_many_predictions(model, test_data) 

    # Using the truth data to predict each time step
    prediction = zeros(3, Nt-Nt_train)

    for i in 1:Nt-Nt_train
        @views input_borehole = input[:, i:i+Nt_train-1]
        input_control = mf_original[i+Nt_train]
        @views μ_input = mean(input_borehole, dims=2)[1:3]
        @views σ_input = std(input_borehole, dims=2)[1:3]
        prediction[:, i] .= μ_input .+ evaluate(model, (normalise(input_borehole, dims=2), input_control)) .* σ_input
    end
end
