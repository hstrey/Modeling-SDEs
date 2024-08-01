using Turing
using DifferentialEquations
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random
using LinearAlgebra
using Distributions
using StatsPlots

# Parameters that I'd imagine are reasonable
sim_len = 600.0
dt = 0.1
nd = Normal(0,1)
dd = -Gamma(5,1) + 1
# Define some variables
@parameters w11 w12 w22 w21 γ
@variables x1(t) x2(t)
@brownian a b

eqs = [D(x1) ~ w11 * x1 + w12 * x2 + γ * a,
    D(x2) ~ w22 * x2 + w21 * x1 + γ * b]

@mtkbuild de = System(eqs, t)

u0map = [
    x1 => 0.0,
    x2 => 0.0
]

parammap = [
    w11 => rand(dd),
    w22 => rand(dd),
    w12 => rand(nd),
    w21 => rand(nd),
    γ => 1.0,
]

prob = SDEProblem(de, u0map, (0.0, sim_len), parammap)
sol = solve(prob, EM(), dt=dt)

plot(sol)

@model function fitlinear(x1,x2)
    m11 ~ -Gamma(5,1) + 1
    m22 ~ -Gamma(5,1) + 1
    m12 ~ Normal(0,1)
    m21 ~ Normal(0,1)

    for i in 2:length(x1)
        dx1 = m11 * x1[i-1] + m12 * x2[i-1]
        dx2 = m22 * x2[i-1] + m21 * x1[i-1]
        x1[i] ~ Normal(x1[i-1] + dx1 * dt,sqrt(dt))
        x2[i] ~ Normal(x2[i-1] + dx2 * dt,sqrt(dt))
    end
end

model = fitlinear(sol[1,:],sol[2,:])
chain = sample(model, NUTS(), 1000)
