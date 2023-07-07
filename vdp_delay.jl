
using DifferentialEquations
using Plots

function vdpDelay!(du,u,h,p,t)
    x1,x2 = u
    θ,ϕ,tau = p
    x1hist = h(p,t-tau)[1]
    du[1] = x2
    du[2] = θ*(1-x1hist^2)*x2 - x1hist
end

h(p,t) = ones(2)

lags = [0.5]
p = [1.0,1.0,lags[1]]
tspan = (0.0,100.0)
u0 = [1.0,1.0,1.0]

prob = DDEProblem(vdpDelay!, u0, h, tspan, p; constant_lags = lags)

sol = solve(prob, MethodOfSteps(Tsit5()))
plot(sol)

