using ModelingToolkit
using DifferentialEquations
using Plots

@variables t u(t)
@parameters gL C I EL Vr Vth
D = Differential(t)

equation = D(u) ~ (-gL * (u - EL) + I) / C

continuous_events = [u ~ -50.0] => [u ~ -65.0]
@named lif_system = ODESystem([equation],t; continuous_events = [u ~ -50.0] => [u ~ -65.0])

u0 = [u => -70.0]
p = [gL => 1.0 , EL => -65.0 , C => 1.0 , Vth => -50.0 , I => 30.0 , Vr => -70.0]
tspan = (0.0, 100.0)

lif_system = complete(lif_system)
problem = ODEProblem(lif_system, u0, tspan, p)
solution = solve(problem, Tsit5())

gr()
plot(solution, xlabel="Time", ylabel="Membrane Potential", title="LIF Neuron Model",xlim=(0,1))