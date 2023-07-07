using ModelingToolkit

@parameters θ, ϕ, τ
@variables t x(t) y(t)
D = Differential(t)

eqs = [D(x) ~ y,
       D(y) ~ θ*(1-(x-τ)^2)*y - (x-τ)]

@named vdpDelay = DDESystem(eqs,t,lags = [τ])

u0map = [
           x => 0.1,
           y => 0.1
        ]
       
parammap = [
           θ => 1.0,
           ϕ => 0.1
           ]

lags = [ τ => 0.1 ]
       
tspan = [0.0, 20]

prob = DDEProblem(vdpDelay,u0map,tspan,parammap;constant_lags = lags)
solve(prob)
