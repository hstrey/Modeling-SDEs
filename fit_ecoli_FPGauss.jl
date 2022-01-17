using DifferentialEquations, ApproxFun, ModelingToolkit
using DiffEqOperators, DomainSets, Interpolations
using Plots, StatsPlots
using Distributions
using Random, Statistics

# Parameters, variables, and derivatives
@parameters t y a k η0 η1 ϵ β σ y0 t0
@variables p(..)
Dt = Differential(t)
Dy = Differential(y)
Dyy = Differential(y)^2

# this is Uri's group guess
# η0=0.33
# η1=0.0048
# β=1.06
# ϵ=0.147
# a=0.33
# k=0.29

# 1D PDE and boundary conditions
eq  = Dt(p(t,y)) ~ p(t,y)*β*a* exp(a*(y+k))/(exp(a*y)+exp(a*k))^2 -
                 Dy(p(t,y))*(η0 + η1*(t+t0) -β*exp(a*y)/(exp(a*y)+exp(a*k))) +
                 ϵ*Dyy(p(t,y))

bcs = [p(0,y) ~ 1/sqrt(2π)/σ*exp(-(y-y0)^2/2/σ^2),
    p(t,-3.0) ~ 0.0,
    p(t,10.0) ~ 0.0]

# Space and time domains
domains = [t ∈ Interval(0.0,100.0),
           y ∈ Interval(-3.0,10.0)]

# PDE system
par = [η0 => 0.33, η1=>0.0048, β=>1.06, ϵ=>0.147, a=>0.33, k=>0.29, y0 => -0.5, σ =>0.1, t0=>40.0]
@named pdesys = PDESystem(eq,bcs,domains,[t,y],[p(t,y)],par)

# Method of lines discretization
dy = 0.1
order = 2
discretization = MOLFiniteDifference([y=>dy],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Plot results and compare with exact solution
y_range = (-3.0:dy:10.0)[2:end-1]

function FPsolveGauss(η0,η1,ϵ,β,a,k,y0,σ,t0,t1,y_range,prob)
    """
    solves the Fokker-Planck equation using the parameters and boundary conditions
    at t1 and then calculate the mean and standard deviation
    """
    par = [η0,η1,β,ϵ,σ,y0,t0]
    prob2 = remake(prob,p=par)
    sol2 = solve(prob2,Tsit5(),saveat=[t1-t0])
    p = sol2.u[1] / sum(sol2.u[1]) # calculate normalized p(y)
    m = sum(p .* y_range)
    s = sqrt(sum(p .* y_range .^2) - m^2)
    return m,s
end

function FPsolve(η0,η1,ϵ,β,a,k,y0,σ,t0,t1,y_range,prob)
    """
    solves the Fokker-Planck equation using the parameters and boundary conditions
    at t1 and then creates a normalized Fun approximation that can be used as
    probability distribution
    """
        par = [η0,η1,β,ϵ,σ,y0,t0]
        prob2 = remake(prob,p=par)
        sol2 = solve(prob2,Tsit5(),saveat=[t1-t0])
        solution1itp = CubicSplineInterpolation(y_range,sol2.u[1])
        f = Fun(y->solution1itp(y), -2.9..9.9)
        f = f/sum(f)
        return f, sol2.u[1]
end

struct FPDist{T<:Real} <: ContinuousUnivariateDistribution
    η0::T
    η1::T
    ϵ::T
    β::T
    a::T
    k::T
    y0::T
    σ::T
    t0::T
    t1::T
    y_range::StepRangeLen
    f::Fun
end

FPDist(η0,η1,ϵ,β,a,k,y0,σ,t0,t1,y_range) = FPDist(η0,η1,ϵ,β,a,k,y0,σ,t0,t1,y_range,FPsolve(η0,η1,ϵ,β,a,k,y0,σ,t0,t1,y_range,prob)[1])

Distributions.rand(rng::AbstractRNG, d::FPDist) = ApproxFun.sample(d.f)
Distributions.logpdf(d::FPDist, x::Real) = log(d.f(x))
Distributions.logpdf(d::FPDist, x::AbstractVector{<:Real}) = log.(d.f.(x))
Distributions.minimum(d::FPDist) = -3.0
Distributions.maximum(d::FPDist) = 10.0

f,sol = FPsolve(0.35,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,40.0,60.0,y_range,prob)
d = FPDist(0.35,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,40.0,60.0,y_range)
data = rand(d,50) # create test data

FPsolveGauss(0.35,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,40.0,60.0,y_range,prob)

using Turing
using FillArrays
using LinearAlgebra: I
using ReverseDiff
Turing.setadbackend(:reversediff)

@model function fpmodel(data,prob)
    eta0 ~ Normal(0.35,0.1)
    m,s = FPsolveGauss(eta0,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,40.0,60.0,y_range,prob)
    data ~ MvNormal(Fill(m, length(data)), s^2 * I)
end

chain = Turing.sample(fpmodel(data,prob), NUTS(0.65), 1000)
plot(chain)