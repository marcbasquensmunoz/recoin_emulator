using BoreholeNetworksSimulator
using Distributions
using DataFrames

Δt = 3600*24.
Nt = 10    # Number of time steps

α = 1e-6
λ = 3.

D = 0.
H = 145.

T0 = 9.

positions = [(0., 0.), (14., 0.), (24., 0.)]
Nb = length(positions)
network = all_parallel_network(Nb)

constraint = TotalHeatLoadConstraint(ones(Nt))

options = SimulationOptions(
    method = NonHistoryMethod(),
    constraint = constraint,
    borefield = EqualBoreholesBorefield(borehole_prototype=SingleUPipeBorehole(H=H, D=D), positions=positions),
    fluid = EthanolMix(),
    medium = GroundMedium(λ=λ, α=α, T0=T0),
    boundary_condition = DirichletBoundaryCondition(),
    Δt = Δt,
    Nt = Nt,
    configurations = [network]
)
containers = initialize(options)

flow_rate = 3. # m³/h
ρ = 1051.2     # kg/m³
mass_flow_rate = flow_rate * ρ / 3600. # kg/s

operator = ConstantOperator(network, mass_flows = mass_flow_rate / 3 * ones(3))
simulate!(operator=operator, options=options, containers=containers)
