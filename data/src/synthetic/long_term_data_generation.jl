using BoreholeNetworksSimulator
using BNSPlots
using Distributions
using CoolProp
using Interpolations
using LambertW

# Create the set up for the simulation
function configuration()
    Δt = 3600.
    Nt = 8760*20

    T0 = 9.
    α = 1e-6
    λ = 3.

    D = 0.
    H = 145.

    positions = [(0., 0.), (14., 0.), (24., 0.)]
    Nb = length(positions)
    network = all_parallel_network(Nb)

    SimulationOptions(
        method = NonHistoryMethod(),
        constraint = TotalHeatLoadConstraint(ones(Nt)),
        borefield = EqualBoreholesBorefield(borehole_prototype=SingleUPipeBorehole(H=H, D=D), positions=positions),
        fluid = GlycolMix(),
        medium = GroundMedium(λ=λ, α=α, T0=T0),
        boundary_condition = DirichletBoundaryCondition(),
        Δt = Δt,
        Nt = Nt,
        configurations = [network]
    )
end

# Simple linear relation for the energy demand depending on air T
Q_demand(T) = 5183.03 - 158.91 *  T

# Generate a randomized energy demand based on temperature cycles
function generate_energy_demand(options)
    t = options.t
    T0 = options.medium.T0 + rand(Normal(0., 5.))
    Tin_synth = T0 .+ rand(Normal(0., 2.)) .* sin.(2π*t/(8760 * 3600)) .+ rand(Normal(0., 4)) .* sin.(2π*t/(24 * 3600))
    Q_demand.(Tin_synth)
end

fluid_density(options) = mean(options.fluid.stored_properties.ρ) / 1000.
function pumping_power(η, H, V, D, fluidproperties, T)
    Δp = pressure_drop(H,V,D,fluidproperties,T)
    P = Δp * V / η
    return P
end
fDarcy(Re) = Re<2300. ? 64/Re : (1/(0.838*lambertw(0.629*Re)))^2
function pressure_drop(H,V,D,fluidproperties,T)
    ρ = fluidproperties.ρ(T)
    μ = fluidproperties.μ(T)

    v = V/(π*D^2/4)
    Re = ρ*v*D/μ
    fD = fDarcy(Re) 
    
    Δp = fD * H/D * v^2 * ρ/2
    return Δp
end
struct SecondaryFluid
    ρ
    μ
    cp
    λ
 end

function SecondaryFluid(fluid)
    spl = secondary_fluid(fluid)
    return SecondaryFluid(spl.ρ, spl.μ, spl.cp, spl.λ)
end
function secondary_fluid(fluid)
    T = fluid.stored_properties.T
    Trange = range(start=minimum(T), stop=maximum(T), step=T[2]-T[1]) 
    ρs_T = fluid.stored_properties.ρ
    μs_T = fluid.stored_properties.μ
    cps_T = fluid.stored_properties.cp
    λ_T = fluid.stored_properties.k

    spl_ρs  = cubic_spline_interpolation(Trange, ρs_T, extrapolation_bc = Flat())
    spl_μs  = cubic_spline_interpolation(Trange, μs_T, extrapolation_bc = Flat())
    spl_cps = cubic_spline_interpolation(Trange, cps_T, extrapolation_bc = Flat())
    spl_λs  = cubic_spline_interpolation(Trange, λ_T, extrapolation_bc = Flat())

    spl = (ρ=spl_ρs, μ = spl_μs, cp=spl_cps, λ= spl_λs)
    return spl
end

function extract_features(options, containers)
    Tfin = BNSPlots.get_Tfin(containers)
    Tfout = BNSPlots.get_Tfout(containers)
    Tb = BNSPlots.get_Tb(containers)
    #q = BNSPlots.get_q(containers)
    q = containers.X[3options.Nb+1:end, :]
    mf = containers.mf
    ΔTf = Tfout - Tfin
    ΔTf_tot = sum((Tfout - Tfin) .* mf, dims=1) ./ sum(mf, dims=1)

    H = BoreholeNetworksSimulator.get_H(options.borefield, 1)

    Q_tot = sum(q, dims=1) .* H

    ρ = fluid_density(options)
    d = BoreholeNetworksSimulator.get_rp(options.borefield.borehole_prototype)
    Tb_mean = mean(Tb, dims=1)
    Vf = sum(mf, dims=1) ./ ρ
    fluidproperties = SecondaryFluid(options.fluid)
    P_pump = [pumping_power(0.5, H, Vf[i], d, fluidproperties, Tb_mean[i]) for i in 1:options.Nt]
    vcat(ΔTf, ΔTf_tot, Q_tot, P_pump')
end

# Initialize 
Vf_per_branch = 1.
options = configuration()
containers = initialize(options)
operator = ConstantOperator(options.configurations[1], mass_flows=Vf_per_branch * fluid_density(options) * ones(options.Nb)) 

# Generate data
N = 100 # Samples
# Rows: ΔTf_1, ΔTf_2, ΔTf_3, ΔTf_tot, Q_tot, P_pump
data = zeros(6, options.Nt, N)

for i in 1:N
    @info "Generating sample $i"
    reset!(options)
    options.constraint.Q_tot .= generate_energy_demand(options)
    simulate!(containers=containers, options=options, operator=operator)
    data[:, :, i] = extract_features(options, containers)
end
