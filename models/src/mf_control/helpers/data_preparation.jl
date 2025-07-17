
function normalise_min(x, m=nothing, M=nothing)
    if (isnothing(M)) M = vec(maximum(x, dims=2)) end
    if (isnothing(m)) m = vec(minimum(x, dims=2)) end

    norm = (x .- m) ./ (M .- m)
    @views norm[findall(==(0), M .- m), :] .= 0.5
    return norm, m, M
end

denormalise_min(data, m, M) = hcat([@. data[:, i] .* (M - m) .+ m for i in 1:size(data, 2)]...)

function prepare_data(data, Nt_train, prediction_steps)

    Nt = size(data, 2)    # Length of each time series
    N = size(data, 3)     # Number of time series
    N_train = N*(Nt-Nt_train-prediction_steps) 

    X_state = zeros(Float32, 4, Nt_train, N_train)                
    X_control = zeros(Float32, prediction_steps, N_train)        

    X_m = zeros(Float32, 4, N_train)           
    X_M = zeros(Float32, 4, N_train)    

    Y = zeros(Float32, 3, prediction_steps, N_train)              

    i = 1
    for j in 1:N
        for k in 1:Nt-Nt_train-prediction_steps
            window = k:k+Nt_train+prediction_steps-1
            train_subwindow = 1:Nt_train
            prediction_subwindow = Nt_train+1:Nt_train+prediction_steps

            @views window_data = data[:, window, j]

            # Normalize the train data
            @views normalized_data, mins, maxs = normalise_min(window_data[:, train_subwindow])

            # Normalize the control and target data using the same normalization as the train data 
            # (at prediction time we don't have information about the future values)
            @views normalized_target, _, _ = normalise_min(window_data[1:3, prediction_subwindow], mins[1:3], maxs[1:3])
            @views normalized_control, _, _ = normalise_min(window_data[4:4, prediction_subwindow], mins[4:4], maxs[4:4])

            # Input
            @views @. X_state[:, :, i] = normalized_data
            @views X_control[:, i] .= vec(normalized_control)
            @views @. X_m[:, i] = mins
            @views @. X_M[:, i] = maxs

            # Target
            @views @. Y[:, :, i] = normalized_target
            i += 1
        end
    end

    train_range = 1:Int(ceil(0.8*N_train))
    test_range = Int(ceil(0.8*N_train)):N_train

    # Data loaders declaration
    batchsize = 512
    train_data = Flux.DataLoader(((X_state[:,:, train_range], X_control[:, train_range]), Y[:, :, train_range]), batchsize=batchsize, shuffle=true)
    test_data  = Flux.DataLoader(((X_state[:,:, test_range],  X_control[:, test_range], X_m[:, test_range], X_M[:, test_range]),  Y[:, :, test_range]),  batchsize=32,        shuffle=true)

    return train_data, test_data
end

