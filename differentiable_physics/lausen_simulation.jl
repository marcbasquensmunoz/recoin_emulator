using BoreholeNetworksSimulator
using Distributions
using DataFrames

df = DataFrame(CSV.File("$(@__DIR__)/../lausen_data_fetching/lausen_data_from_2023-08-01T00:00:00Z_to_2024-08-01T00:00:00Z.csv"))

Δt = 3600.
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
    method = ConvolutionMethod(),#NonHistoryMethod(),
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

operator = ConstantOperator(network, mass_flows = 1 * ones(3))
simulate!(operator=operator, options=options, containers=containers)
