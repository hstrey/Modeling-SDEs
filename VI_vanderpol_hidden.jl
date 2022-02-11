using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Turing
using Distributions
using FillArrays
#using MCMCChains
using StatsPlots
using Random
using Plots
#using Turing: Variational
#using DelimitedFiles

Random.seed!(08)

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
tspan = [0.0, 50]
p = [1.0,0.1]


prob1 = SDEProblem(vanderpol, add_noise, u0, tspan, p)
sol = solve(prob1,SOSRI(),saveat=0.1)
time = sol.t
dt=0.1

p1 = plot(sol)

#plot(sol, vars = (1,2))

ensembleprob = EnsembleProblem(prob1)
data = solve(ensembleprob,SOSRI(),EnsembleThreads(),saveat=0.1,trajectories=50)
#plot(EnsembleSummary(data))
#ar=Array(sol)
ar=Array(data) # first index x1,x2; second index time; third index trajectory
ns = rand(Normal(0,.1),2,size(ar,2),size(ar,3)) #this is the measurement noise

#simulation + noise
arn = ar + ns

slope = 5.0
scale = 50.0

function g_sigmoid(x,slope,scale)
    sig = scale ./ (1.0 .+ exp.(-slope*x))
    return sig
end

# x1 and x2 are the untransformed solutions
x1 = ar[1,:,:]
x2 = ar[2,:,:]

# x1_sig and x2_sig are the transformed solutions
x1_sig = g_sigmoid(ar[1,:,:],slope,scale)
x2_sig = g_sigmoid(ar[2,:,:],slope,scale)

# x1 and x2 are the untransformed solutions
x1n = arn[1,:,:]
x2n = arn[2,:,:]

# x1_sig and x2_sig are the transformed solutions
x1n_sig = g_sigmoid(arn[1,:,:],slope,scale)
x2n_sig = g_sigmoid(arn[2,:,:],slope,scale)

p = plot(x1n)
plot(x2n)

Turing.setadbackend(:forwarddiff)

@model function fitvp(data,prob1)
    σ = 0.1
    θ ~ TruncatedNormal(0,3,0,Inf)
    ϕ ~ Uniform(0,0.5)
    #ϕ ~ Gamma(1.1,0.1)
    #x01 ~ Normal(0.1,0.1)
    #x02 ~ Normal(0.1,0.1)

    p = [θ,ϕ]
    u0 = [0.1,0.1]
    #u0 = typeof(ϕ).(prob1.u0)
    #u0 = u0

    prob = remake(prob1,u0=u0,p=p)
    predicted = solve(prob,SOSRI(),saveat=0.1)#,maxiters=1e7)

    if predicted.retcode != :Success
        Turing.@addlogprob! -Inf
        return
    end
    for i in 1:size(data,2)
        data[:,i] ~ MvNormal(sol[1,:],σ)
    end
end

@model function fitvpEM(datax, datay)
    σ = 0.1
    θ ~ TruncatedNormal(0,3,0,Inf)
    ϕ ~ Uniform(0,0.5)
    for i in 2:length(datax)
        dx1 = datay[i-1]
        dx2 = θ*(1-datax[i-1]^2)*datay[i-1] - datax[i-1]
        datax[i] ~ Normal(datax[i-1]+dx1*dt,ϕ*sqrt(dt))
        datay[i] ~ Normal(datay[i-1]+dx2*dt,ϕ*sqrt(dt))
    end
end

@model function fitvpEMn(dataxn, datayn)
    σ = 0.1
    θ ~ TruncatedNormal(0,3,0,Inf)
    ϕ ~ Uniform(0,0.5)
    for i in 2:length(datax)
        dx1 = datay[i-1]
        dx2 = θ*(1-datax[i-1]^2)*datay[i-1] - datax[i-1]
        xh ~ Normal(datax[i-1]+dx1*dt,ϕ*sqrt(dt))
        yh ~ Normal(datay[i-1]+dx2*dt,ϕ*sqrt(dt))
        data
    end
end

#model = fitvp(y,prob1)

model = fitvpEM(sol[1,:],sol[2,:])

# model = fitvp(sol,prob1)

#chain = sample(model, NUTS(0.25), MCMCThreads(),1000, 1)#,init_theta = [0.1, 0.5, 0.1])
chain = sample(model, NUTS(0.65),2000)

plot(chain)
