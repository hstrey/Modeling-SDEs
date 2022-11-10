using ModelingToolkit, Plots, DifferentialEquations
using Distributions

N = 10

@parameters K ω[1:N]
@parameters t
D = Differential(t)

@variables θ(t)[1:N]

eqs = []
for (θ_i,ω_i) in zip(θ,ω)
    push!(eqs, D(θ_i) ~ ω_i + K/N*sum([sin(θ_j-θ_i) for θ_j in θ]))
end

# initial conditions for the omegas
w_dist = Normal(0,1)
ω_dist = Normal(0,3)
w_0 = rand(w_dist,N)
θ_0 = rand(ω_dist,N)
@named kuramoto = ODESystem(eqs, t)

