
function extract_test_sample(test_data)
    input_batch, truth_batch = first(test_data)
    K = rand(1:size(truth_batch, 3))
    
    state_batch, control_batch, m_batch, M_batch = input_batch

    @views state = state_batch[:, :, K]
    @views control = control_batch[:, K]
    @views m = m_batch[:, K]
    @views M = M_batch[:, K]
    @views truth = truth_batch[:, :, K]

    return denormalise_min(hcat(state, vcat(control, truth)), m, M)
end

function plot_training_loss(loss_train, loss_test) 
    fig_loss = Figure()
    ax = Axis(fig_loss[1, 1], xlabel = "Training step", ylabel =  L"\log_{10} L", title="Loss evolution during training")
    lines!(ax, log10.(loss_train), label="Train")
    lines!(ax, log10.(loss_test), label="Test")
    axislegend(""; position= :rt, backgroundcolor = (:grey90, 0.25));
    fig_loss
end

function plot_result(prediction, truth, Nt_train, prediction_steps)
    t_axis_truth = 1:Nt_train+prediction_steps
    t_axis_pred = Nt_train+1:Nt_train+prediction_steps
    @views test_mse = Flux.mse(truth[1:3, t_axis_pred], prediction[1:3, :])

    y_labels = [L"T_{in} \ (K)", L"T_{b} \ (K)", L"Q \ (W/m)", L"m_f \ (kg/s)"]
    prediction_color = Makie.wong_colors()[2]
    fig_example = Figure(size=(600, 800))
    figure_axes = []
    for (i, ylabel) in enumerate(y_labels)
        ax = Axis(fig_example[i, 1], xlabel = L"t \ (days)", ylabel = ylabel, title = i == 1 ? "Prediction of the next $(prediction_steps) hours; MSE = $(test_mse)" : "")
        push!(figure_axes, ax)
        lines!(ax, t_axis_truth, truth[i,:], label="Truth")
        plot!(ax, t_axis_pred, prediction[i,:], label="Prediction", color=prediction_color)
    end
    axislegend(""; position= :lb, backgroundcolor = (:grey90, 0.25));
    linkxaxes!(figure_axes...)
    for ax in figure_axes[1:end-1]
        hidexdecorations!(ax)
    end
    fig_example
end

function plot_prediction(model, test_data, Nt_train, prediction_steps)
    test_input_batch, test_target_batch = rand(collect(test_data))
    test_state_batch, test_control_batch, test_m_batch, test_M_batch = test_input_batch

    K = rand(1:size(test_state_batch, 3))
    @views test_state = test_state_batch[:, :, K]
    @views test_control = test_control_batch[:, K]
    @views test_m = test_m_batch[:, K]
    @views test_M = test_M_batch[:, K]
    @views test_target = test_target_batch[:, :, K]

    result = model((test_state, test_control))[:, :, end]

    @views test_truth = denormalise_min(vcat(hcat(test_state[1:3, :], test_target), hcat(test_state[4:4, :], test_control')), test_m, test_M) 
    @views prediction = denormalise_min(vcat(result, test_control'), test_m, test_M)

    plot_result(prediction, test_truth, Nt_train, prediction_steps)
end

function plot_many_predictions(model, data, Nt_train) 
    Nt = size(data, 2)
    # Using the truth data to predict each time step
    prediction = zeros(4, Nt-Nt_train)

    for i in 1:Nt-Nt_train
        @views window_input = data[:, i:i+Nt_train-1]
        @views input_control = data[4, i+Nt_train:i+Nt_train]
        normalised_input, m, M = normalise_min(window_input)
        @views normalized_control, _, _ = normalise_min(data[4, i+Nt_train:i+Nt_train], m[4], M[4])

        result = model((normalised_input, normalized_control))[:, :, end]
        @views prediction[1:3, i] .= denormalise_min(result, m[1:3], M[1:3])
        prediction[4:4, i] .= input_control
    end
    
    plot_result(prediction, data, Nt_train, prediction_steps)
end
