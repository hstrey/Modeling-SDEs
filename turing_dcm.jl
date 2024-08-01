using Turing
using DifferentialEquations
using ModelingToolkit
using Random
using Neuroblox
using MetaGraphs
using LinearAlgebra

function stochasticLM_3node(;connection_adj=zeros(3, 3), sim_len=10.0, noise_weight=1.0)
    @named lm1 = LinearNeuralMass()
    @named lm2 = LinearNeuralMass()
    @named lm3 = LinearNeuralMass()
    @named ou1 = OUBlox(τ=1, σ=0.1)
    @named ou2 = OUBlox(τ=1, σ=0.1)
    @named ou3 = OUBlox(τ=1, σ=0.1)
    @named bold1 = BalloonModel()
    @named bold2 = BalloonModel()
    @named bold3 = BalloonModel()

    g = MetaDiGraph()
    add_blox!.(Ref(g), [lm1, lm2, lm3, ou1, ou2, ou3, bold1, bold2, bold3])
    add_edge!(g, 1, 1, Dict(:weight => connection_adj[1, 1]))
    add_edge!(g, 1, 2, Dict(:weight => connection_adj[1, 2]))
    add_edge!(g, 1, 3, Dict(:weight => connection_adj[1, 3]))
    add_edge!(g, 2, 1, Dict(:weight => connection_adj[2, 1]))
    add_edge!(g, 2, 2, Dict(:weight => connection_adj[2, 2]))
    add_edge!(g, 2, 3, Dict(:weight => connection_adj[2, 3]))
    add_edge!(g, 3, 1, Dict(:weight => connection_adj[3, 1]))
    add_edge!(g, 3, 2, Dict(:weight => connection_adj[3, 2]))
    add_edge!(g, 3, 3, Dict(:weight => connection_adj[3, 3]))

    add_edge!(g, 4, 1, Dict(:weight => noise_weight)) # ou1 -> lm1
    add_edge!(g, 5, 2, Dict(:weight => noise_weight)) # ou2 -> lm2
    add_edge!(g, 6, 3, Dict(:weight => noise_weight)) # ou3 -> lm3
    add_edge!(g, 1, 7, Dict(:weight => 0.1)) # lm1 -> bold1
    add_edge!(g, 2, 8, Dict(:weight => 0.1)) # lm2 -> bold2
    add_edge!(g, 3, 9, Dict(:weight => 0.1)) # lm3 -> bold3

    @named sys = system_from_graph(g)
    sys = structural_simplify(sys)

    prob = SDEProblem(sys, [], (0.0, sim_len))
    sol = solve(prob, saveat=1)
    return sol
end

initial_sol = stochasticLM_3node() # usually solves fine

# Parameters that I'd imagine are reasonable
sim_len = 600.0
initial_connections = (rand(3, 3).*2.0) .- 1.0
initial_connections[1, 1] = -1.0*sum((abs.(initial_connections[1, 2:3]))) + (rand()*0.01)
initial_connections[2, 2] = -1.0*sum(abs.(initial_connections[2, [1, 3]])) + (rand()*0.01)
initial_connections[3, 3] = -1.0*sum(abs.(initial_connections[3, 1:2])) + (rand()*0.01)
noise_weight = 1.0

initial_sol = stochasticLM_3node(connection_adj=initial_connections, sim_len=sim_len) # crashes regardless of noise weight
odedata = Array(initial_sol)
initial_data = odedata[end-2:end, :]

@model function fit_sDCM(data)

    σ ~ truncated(Normal(1, 1); lower=0.0, upper=3.0)
    w11 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w12 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w13 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w21 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w22 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w23 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w31 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w32 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)
    w33 ~ truncated(Normal(0, 0.5); lower=-1.0, upper=1.0)

    new_connections = [w11 w12 w13; w21 w22 w23; w31 w32 w33]
    predicted = stochasticLM_3node(connection_adj=new_connections, sim_len=600.0)
    predicted_data = Array(predicted)
    predicted_data = predicted_data[end-2:end, :]
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted_data[:, i], σ^2 * I)
    end

    return nothing
end

model = fit_sDCM(initial_data)
chain = sample(model, NUTS(), MCMCSerial(), 1000, 3; progress=true)