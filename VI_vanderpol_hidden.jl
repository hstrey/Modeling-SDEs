using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Turing
using Distributions
using FillArrays
#using MCMCChains
using StatsPlots
using Random
using Plots
using Turing: Variational
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
tspan = [0.0, 20]
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

p = plot(x1n[:,1:5])
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
    θ ~ TruncatedNormal(0,3,0,Inf)
    ϕ ~ Uniform(0,0.5)
    dxh = datay[1:end-1]
    dyh = θ .* (1 .- datax[1:end-1] .^ 2) .* datay[1:end-1] .- datax[1:end-1]
    datax[2:end] ~ MvNormal(datax[1:end-1] .+ dxh * dt,ϕ*sqrt(dt))
    datay[2:end] ~ MvNormal(datay[1:end-1] .+ dyh * dt,ϕ*sqrt(dt))
end

@model function fitSDE(f,g,dt,datax,datay)
    θ ~ TruncatedNormal(0,3,0,Inf)
    ϕ ~ Uniform(0,0.5)
    p = [θ,ϕ]
    data = [datax[1:end-1] datay[1:end-1]]'
    du = zero(data)
    dun = zero(data)
    ff(du,u) = f(du,u,p,0)
    ff.(du,data)
    gg(du,u) = g(du,u,p,0)
    gg.(dun,data,p,0)
    data[2:end] ~ MvNormal(data .+ du * dt,dun * sqrt(dt))
end

@model function fitvpEMn(datax, datay)
    σ = 0.1
    θ ~ TruncatedNormal(0,3,0,Inf)
    ϕ ~ Uniform(0,0.5)
    dxh = datay[1:end-1]
    dyh = θ .* (1 .- datax[1:end-1] .^ 2) .* datay[1:end-1] .- datax[1:end-1]
    xh ~ MvNormal(datax[1:end-1] .+ dxh * dt,ϕ*sqrt(dt))
    yh ~ MvNormal(datay[1:end-1] .+ dyh * dt,ϕ*sqrt(dt))
    datax[2:end] ~ MvNormal(xh,σ)
    datay[2:end] ~ MvNormal(yh,σ)
end

#model = fitvp(y,prob1)

model = fitvpEMn(x1n[:,1],x2n[:,1])

plot(x1n[:,1])
plot!(x2n[:,1])
# model = fitvp(sol,prob1)

#chain = sample(model, NUTS(0.25), MCMCThreads(),1000, 1)#,init_theta = [0.1, 0.5, 0.1])
chain = sample(model, NUTS(0.65),1000)
chainarray = Array(chain)
chainx = chainarray[:,3:203]
chainy = chainarray[:,204:end]
chainx_avg = mean(chainx,dims=1)[:]
chainy_avg = mean(chainy,dims=1)[:]
chainx_std = std(chainx,dims=1)[:]
chainy_std = std(chainy,dims=1)[:]

plot(0:0.1:20,x1n[:,1],label="x data",xlabel="time in sec",ylabel="amplitude",linewidth=2)
plot!(0:0.1:20,x2n[:,1],label="y data",linewidth=2)
savefig("VPdata.png")

plot(0.1:0.1:20,chainx_avg,ribbon=chainx_std,label="x pred",xlabel="time in sec",ylabel="amplitude")
plot!(0.1:0.1:20,chainy_avg,ribbon=chainy_std,label="y pred")
savefig("VPpred.png")

model2 = fitvpEM(x1[:,1],x2[:,1])
chain2 = sample(model2, NUTS(0.65),2000)
plot(chain2)
savefig("ChainEM.png")

# ADVI
advi = ADVI(10, 1000)
q = vi(model, advi)

# sampling
z = rand(q, 10_000)
avg = vec(mean(z; dims = 2))
chainx_advi = avg[3:503]
chainy_advi = avg[504:end]
plot(chainx_advi)
plot!(chainy_advi)