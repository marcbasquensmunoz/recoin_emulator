using BoreholeNetworksSimulator
using Distributions

Δt = 60 * 20.
Nt = 24*30*3   # Number of time steps
N = 500      # Number of samples

α = 1e-6
λ = 3.

D = 0.
H = 100.

σ = 5.
T0 = 10.

network = all_parallel_network(2)
positions = [(0., 0.), (σ, 0.)]
Nb = length(positions)
configurations = [network]

method = NonHistoryMethod()
medium = GroundMedium(λ=λ, α=α, T0=T0)
borehole = SingleUPipeBorehole(H=H, D=D)
borefield = EqualBoreholesBorefield(borehole_prototype=borehole, positions=positions)
constraint = TotalHeatLoadConstraint(ones(Nt))
fluid = Water()

options_Q_tot = SimulationOptions(
    method = method,
    constraint = constraint,
    borefield = borefield,
    fluid = fluid,
    medium = medium,
    boundary_condition = DirichletBoundaryCondition(),
    Δt = Δt,
    Nt = Nt,
    configurations = configurations
)
containers_Q_tot = initialize(options_Q_tot)


struct MFOperator{Arr <: AbstractArray, T <: Number} <: Operator
    mass_flow_series::Arr
    mass_flows::Vector{T}
end

function BoreholeNetworksSimulator.operate(operator::MFOperator, step, options, X)
    operator.mass_flows .= operator.mass_flow_series[step]
    BoreholeOperation(network = options.configurations[1], mass_flows = operator.mass_flows)
end

function generate_T_in!(Q, mf, options_Q_tot, containers_Q_tot)
    operator = MFOperator(mf, zeros(n_branches(network)))
    reset!(options_Q_tot)
    options_Q_tot.constraint.Q_tot .= Q
    simulate!(operator=operator, options=options_Q_tot, containers=containers_Q_tot)
    return containers_Q_tot
end

# Generate T_in that make sense
t = 1:Nt
Tin_synth = zeros(Nt, N)
Tb_synth = zeros(Nt, N)
Q_synth = zeros(Nt, N)
Q_tot_synth = zeros(Nt, N)
mf = 0.34 .* ones(Nt)
mf_synth = mf
for i in 1:N
    yearly_mean = rand(Uniform(-10, 10)) 
    yearly_amplitude = rand(Uniform(5, 15))
    daily_amplitude = rand(Uniform(0, 1)) 
    random_phase = rand(Uniform(0, 2π))
    @. Q_tot_synth[:, i] = H * Nb * (yearly_mean + yearly_amplitude * sin(2π/8760 * t + random_phase) + daily_amplitude * sin(2π/24 * t + random_phase))
    containers = generate_T_in!(Q_tot_synth[:, i], mf, options_Q_tot, containers_Q_tot)
    Tin_synth[:, i] = containers.X[1, :]
    Tb_synth[:, i] = containers.X[5, :]
    Q_synth[:, i] = containers.X[7, :]
end
