using DifferentialEquations, ApproxFun, ModelingToolkit, DiffEqOperators, DomainSets, Interpolations

# Parameters, variables, and derivatives
@parameters t x γ d x0 σ
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq  = Dt(u(t,x)) ~ γ * x * Dx(u(t,x)) + d*u(t,x) + Dxx(u(t,x))
bcs = [u(0,x) ~ 1/sqrt(2π)/σ*exp(-(x-x0)^2/2/σ^2),
    u(t,-5.0) ~ 0.0,
    u(t,5.0) ~ 0.0]

# Space and time domains
domains = [t ∈ Interval(0.0,2.0),
           x ∈ Interval(-5.0,5.0)]

# PDE system
par = [γ => 2.0,d => 1.0,x0 => 1.0, σ => 0.3, x0 => -0.5, σ =>0.1]

@named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)],par)

# Method of lines discretization
dx = 0.1
order = 2
discretization = MOLFiniteDifference([x=>dx],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
# using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = (-5.0:dx:5.0)[2:end-1]
t = sol.t

using Plots
plt = plot()
plot!(x,sol.u[5],label="Numerical, t=1.0")
solution1itp = CubicSplineInterpolation(x,sol.u[5])
f = Fun(x->solution1itp(x), -4.9..4.9)
f = f/sum(f)
xx = ApproxFun.sample(f,10000)
histogram(xx;normed=true)
plot!(f)
