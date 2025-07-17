using CSV
using DataFrames
using CairoMakie
using Statistics

file = "$(@__DIR__)/../../lausen_data_fetching/temp_profile.csv"
profile_raw_data = CSV.read(file, DataFrame; delim = ',')

file_power = "$(@__DIR__)/../../lausen_data_fetching/power_data.csv"
power_raw_data = CSV.read(file_power, DataFrame; delim = ',')

times = unique(profile_raw_data[!, :_time])
z_down_raw = unique(subset(profile_raw_data, :Direction => x -> x .== "Down")[!, :Deep])
z_down = sort(parse.(Float64, replace.(z_down_raw, "m" => "", "_" => ".")))
z_up_raw = unique(subset(profile_raw_data, :Direction => x -> x .== "Up")[!, :Deep])
z_up = sort(parse.(Float64, replace.(z_up_raw, "m" => "", "_" => ".")))

N_down = 120
N_up = 119
N_samples = length(times)
T_down_z = zeros(N_down, N_samples)
T_up_z = zeros(N_up, N_samples)

for (i, time) in enumerate(times)
    down_t = subset(profile_raw_data, :_time => x -> x .== time, :Direction => x -> x .== "Down")
    up_t = subset(profile_raw_data, :_time => x -> x .== time, :Direction => x -> x .== "Up")

    T_down_z[:, i] = sort(down_t, :DeepCounter)[!, :ChValue]
    T_up_z[:, i] = sort(up_t, :DeepCounter)[!, :ChValue]
end

Q_data = power_raw_data[!, :q_all][2:end] .* (1000 / 3) # Convert from kW to W and correct for the number of boreholes
ρf = 1051.2 # Density (kg / m^3)
mf = 3.5 * ρf / 3600 / 3 # Convert from m^3/h to kg/s and correct for the number of boreholes
cp = 3700 # J / (Kg * K)

# Fluid temperature profile at a particular time
k = 410
fig = Figure()
ax = Axis(fig[1, 1], xlabel="T (°C)", ylabel="Depth (m)", title="Fluid temperature profile")
lines!(ax, T_down_z[:, k], -z_down, label="Downward pipe")
lines!(ax, T_up_z[:, k], -z_up, label="Upward pipe")
axislegend(position = :rt)
fig
#save("temp_profile.png", fig)


function moving_average(arr, window_size::Int=3)
    n = length(arr)
    res = similar(arr)
    pad = div(window_size, 2)
    for i in 1:n
        start_idx = max(1, i - pad)
        end_idx = min(n, i + pad)
        res[i] = mean(arr[start_idx:end_idx])
    end
    return res
end

function coupled_moving_average(up, down, window_size=3)
    res_up = moving_average(up, window_size)
    res_down = moving_average(down, window_size)

    bottom_temp = mean([res_up[end], res_down[end]])
    res_up[end] = bottom_temp
    res_down[end] = bottom_temp

    return res_up, res_down
end

k = 410
N_ma = 15
smooth_Tu, smooth_Td = coupled_moving_average(T_up_z[:, k], T_down_z[:, k], N_ma)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="T (°C)", ylabel="Depth (m)", title="Smoothed fluid temperature profile")
lines!(ax, smooth_Td, -z_down, label="Downward pipe")
lines!(ax, smooth_Tu, -z_up, label="Upward pipe")
axislegend(position = :rt)
fig
#save("smooth_temp_profile.png", fig)

noisy_T_down_z = zeros(N_down, 0)
noisy_T_up_z = zeros(N_up, 0)
smooth_T_down_z = zeros(N_down, 0)
smooth_T_up_z = zeros(N_up, 0)
Q_train_smooth = zeros(0)

# Discard samples such that outlet temp is lower than inlet
for i in 1:N_samples
    smooth_Tu, smooth_Td = coupled_moving_average(T_up_z[:, i], T_down_z[:, i], 50)
    if smooth_Tu[1] < smooth_Td[1] continue end
    smooth_T_down_z = hcat(smooth_T_down_z, smooth_Td)
    smooth_T_up_z = hcat(smooth_T_up_z, smooth_Tu)
    Q_train_smooth = vcat(Q_train_smooth, Q_data[i])
end
