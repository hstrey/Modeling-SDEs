using ModelingToolkit
using StochasticDiffEq
using Turing
using Distributions
using FillArrays
using StatsPlots
using Random
using Plots
using ReverseDiff
using Turing: Variational

Random.seed!(08)
# Define some variables
@parameters θ,ϕ
@variables t x(t) y(t)
D = Differential(t)

eqs = [D(x) ~ y,
       D(y) ~ θ*(1-x^2)*y - x]

noiseeqs = [ϕ,ϕ]

@named vdp = SDESystem(eqs,noiseeqs,t,[x,y],[θ,ϕ])

u0map = [
    x => 0.1,
    y => 0.1
]

parammap = [
    θ => 1.0,
    ϕ => 0.1]

tspan = [0.0, 20]

dt = 0.1
prob = SDEProblem(vdp,u0map,tspan,parammap)
sol = solve(prob,SOSRI(),dt=dt)

time = sol.t
p1 = plot(sol)

#plot(sol, vars = (1,2))

ensembleprob = EnsembleProblem(prob1)
data = solve(ensembleprob,EM(),EnsembleThreads(),dt=dt,trajectories=50)
#plot(EnsembleSummary(data))
#ar=Array(sol)
ar=Array(data) # first index x1,x2; second index time; third index trajectory
ns = rand(Normal(0,.3),2,size(ar,2),size(ar,3)) #this is the measurement noise

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
x1n_sig = g_sigmoid(ar[1,:,:],slope,scale) + ns[1,:,:]
x2n_sig = g_sigmoid(ar[2,:,:],slope,scale) + ns[2,:,:]

plot(x1n_sig[:,1])
plot!(x2n_sig[:,1])

Turing.setadbackend(:reversediff)

@model function fitvpEM(datax, datay)
    θ ~ Gamma(0.1,10.0)
    ϕ ~ Uniform(0,0.5)
    dxh = datay[1:end-1]
    dyh = θ .* (1 .- datax[1:end-1] .^ 2) .* datay[1:end-1] .- datax[1:end-1]
    datax[2:end] ~ MvNormal(datax[1:end-1] .+ dxh * dt,ϕ*sqrt(dt))
    datay[2:end] ~ MvNormal(datay[1:end-1] .+ dyh * dt,ϕ*sqrt(dt))
end

# here I am trying to fit and SDE with f(du,u,p,t) and g(du,u,p,t)
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

@model function fitvpEMnsig(datax, datay)
    σ = 0.1
    θ ~ Gamma(0.1,10.0)
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
        datax ~ MvNormal(g_sigmoid(xh,slope,scale),σ)
        datay ~ MvNormal(g_sigmoid(xh,slope,scale),σ)
end

#model = fitvp(y,prob1)

model = fitvpEMn(x1n[:,1],x2n[:,1])
model = fitvpEMnsig(x1n_sig[:,1],x2n_sig[:,1])
plot(x1n[:,1])
plot!(x2n[:,1])
# model = fitvp(sol,prob1)

#chain = sample(model, NUTS(0.25), MCMCThreads(),1000, 1)#,init_theta = [0.1, 0.5, 0.1])
chain = sample(model, NUTS(0.65),1000,init_theta = [1.0,0.1])
chainarray = Array(chain)
chainx = chainarray[:,3:202]
chainy = chainarray[:,203:end]
chainx_avg = mean(chainx,dims=1)[:]
chainy_avg = mean(chainy,dims=1)[:]
chainx_std = std(chainx,dims=1)[:]
chainy_std = std(chainy,dims=1)[:]

scatter(0:0.1:20,x1n[:,1],label="x data",xlabel="time in sec",ylabel="amplitude")
scatter!(0:0.1:20,x2n[:,1],label="y data")
plot!(0:0.1:20,x1[:,1],linewidth=2,ribbon=0.3,label="x")
plot!(0:0.1:20,x2[:,1],linewidth=2,ribbon=0.3,label="y")
savefig("VPdata.png")

plot(0:0.1:20,x1n_sig[:,1],label="x sig data",xlabel="time in sec",ylabel="amplitude",linewidth=2)
plot!(0:0.1:20,x2n_sig[:,1],label="y sig data",linewidth=2)
savefig("VPdatasig.png")

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
z = rand(q, 1000)
avg = vec(mean(z; dims = 2))
zstd = vec(std(z; dims = 2))
chainx_advi_mean = avg[3:202]
chainy_advi_mean = avg[203:end]
chainx_advi_std = zstd[3:202]
chainy_advi_std = zstd[203:end]

plot(0.1:0.1:20,chainx_advi_mean,ribbon=chainx_advi_std,label="x pred",xlabel="time in sec",ylabel="amplitude")
plot!(0.1:0.1:20,chainy_advi_mean,ribbon=chainy_advi_std,label="y pred")
savefig("VPpred.png")