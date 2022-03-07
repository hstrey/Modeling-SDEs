using DifferentialEquations
using Turing
using Distributions
using Random
using Turing: Variational

function vanderpol(du,u,p,t)
    x1,x2 = u
    θ,ϕ = p

    du[1] = x2
    du[2] = θ*(1-x1^2)*x2 - x1
end

function add_noise(du,u,p,t)
    x1,x2 = u
    θ,ϕ = p

    du[1] = ϕ #   0	
    du[2] = ϕ
end

u0 = [0.1, 0.1]
tspan = [0.0, 20]
p = [1.0,0.1]

dt=0.1
prob1 = SDEProblem(vanderpol, add_noise, u0, tspan, p)
sol = solve(prob1,EM(),dt=dt)
time = sol.t

# add noise to the solution
x1 = sol[1,:]
x1n = x1 .+ rand(Normal(0,.3),length(x1))
x2 = sol[2,:]
x2n = x2 .+ rand(Normal(0,.3),length(x2))

@model function fitvpEMn(datax, datay)
    σ = 0.3
    θ ~ Gamma(0.25,4.0)
    ϕ ~ Uniform(0,0.2)
    # create arrays for hidden variables
    xh = zero(datax)
    yh = zero(datay)
    xh[1] = 0.1
    yh[1] = 0.1
    for i in 2:length(datax)
        dxh = yh[i-1]
        dyh = θ * (1.0 - xh[i-1]^2) * yh[i-1] - xh[i-1]
        xh[i] ~ Normal(xh[i-1] + dxh * dt,ϕ*sqrt(dt))
        yh[i] ~ Normal(yh[i-1] + dyh * dt,ϕ*sqrt(dt))
    end
    datax ~ MvNormal(xh,σ)
    datay ~ MvNormal(yh,σ)
end

model = fitvpEMn(x1n,x2n)

# ADVI
advi = ADVI(10, 1000)
q = vi(model, advi)

# sampling
z = rand(q, 1000)
avg = vec(mean(z; dims = 2))
zstd = vec(std(z; dims = 2))
chainx_advi_mean = avg[3:202]
chainy_advi_mean = avg[203:end]
chainx_advi_std = zstd[3:202]
chainy_advi_std = zstd[203:end]
