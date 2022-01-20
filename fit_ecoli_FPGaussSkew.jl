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

y_min = -5.0
y_max = 8.0

# 1D PDE and boundary conditions
eq  = Dt(p(t,y)) ~ p(t,y)*β*a* exp(a*(y+k))/(exp(a*y)+exp(a*k))^2 -
                 Dy(p(t,y))*(η0 + η1*(t+t0) -β*exp(a*y)/(exp(a*y)+exp(a*k))) +
                 ϵ*Dyy(p(t,y))

bcs = [p(0,y) ~ 1/sqrt(2π)/σ*exp(-(y-y0)^2/2/σ^2),
    p(t,y_min) ~ 0.0,
    p(t,y_max) ~ 0.0]

# Space and time domains
domains = [t ∈ Interval(0.0,100.0),
           y ∈ Interval(y_min,y_max)]

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
y_range = (y_min:dy:y_max)[2:end-1]

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
    v = sum(p .* y_range .^2) - m^2
    s = sum(p .* (y_range .- m).^3)/v^1.5
    return m,v,s
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
        f = Fun(y->solution1itp(y), (y_min+dy)..(y_max-dy))
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
Distributions.logpdf(d::FPDist, x::Real) = log(d.f(x)<=0 ? 0.0 : d.f(x))
Distributions.logpdf(d::FPDist, x::AbstractVector{<:Real}) = log.(d.f.(x))
Distributions.minimum(d::FPDist) = y_min
Distributions.maximum(d::FPDist) = y_max

f,sol = FPsolve(0.35,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,0.0,1.0,y_range,prob)
d = FPDist(0.35,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,0.0,1.0,y_range)
data = rand(d,50) # create test data
mean(data)
std(data)

m_s,v_s,s_s = FPsolveGauss(0.3500,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,0.0,1.0,y_range,prob)
gammatt = min(0.99,abs(s_s)^(2/3))
delta = sqrt(π*gammatt/(gammatt+((4-π)/2)^(2/3))/2)*sign(s_s)
alpha = delta/sqrt(1-delta^2)
omega = sqrt(v_s/(1-2*delta^2/π))
xi = m_s - omega*delta*sqrt(2/π)
solplot = plot(y_range,sol)

plot!(y_range,pdf.(Normal(m_s,sqrt(v_s)),y_range))
plot!(y_range,pdf.(SkewNormal(xi,omega,alpha),y_range))

sum((sol .-pdf.(Normal(m_s,sqrt(v_s)),y_range)).^2)
sum((sol .-pdf.(SkewNormal(xi,omega,alpha),y_range)).^2)

using Turing
using FillArrays
using LinearAlgebra: I
#using ReverseDiff
Turing.setadbackend(:forwarddiff)

@model function fpmodel(data,t1)
    eta0 ~ Normal(0.35,0.2)
    par = [eta0,0.0048,1.06,0.147,0.33,0.29,-0.5,0.1,0.0]
    prob2 = remake(prob,p=par)
    sol2 = solve(prob2,Tsit5(),saveat=[t1])
    if sol2.retcode != :Success # don't consider if solver fails
        # @show "solver fail"
        Turing.@addlogprob! -Inf
        return
    end
    p = sol2.u[1] / sum(sol2.u[1]) # calculate normalized p(y)
    m = sum(p .* y_range)
    v = sum(p .* (y_range .^2)) - m^2
    if v<=0 # variance cannot be less than zero
        # @show "solver fail"
        Turing.@addlogprob! -Inf
        return
    end
    # now calculate the parameters for a skewed normal
    # using: https://en.wikipedia.org/wiki/Skew_normal_distribution
    s = sum(p .* (y_range .- m).^3)/v^1.5
    # @show s
    gammatt = min(0.99,abs(s)^(2/3))
    delta = sqrt(π*gammatt/(gammatt+((4-π)/2)^(2/3))/2)*sign(s)
    alpha = delta/sqrt(1-delta^2)
    omega = sqrt(v/(1-2*delta^2/π))
    xi = m - omega*delta*sqrt(2/π)
    # @show m,v,s
    data ~ filldist(SkewNormal(xi,omega,alpha),length(data))
end

chain = Turing.sample(fpmodel(data,1.0), NUTS(0.65), 1000)
plot(chain)