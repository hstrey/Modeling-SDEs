using DifferentialEquations
using Plots
using StatsPlots
using DelimitedFiles
using DifferentialEquations
using Turing
using DifferentialEquations.EnsembleAnalysis

# reading in data
experiment = readdlm("Ecoli.csv", ',', Float64)

# create two lists
# time_list is the data for each time point
# time = [24.5,31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5]
time_list = []
for time in eachcol(experiment)
    
    cleantime = time[time .> 0.3]
    deleteat!(cleantime, findall(isnan,cleantime))
    push!(time_list,log.(cleantime))
end

time_mean = []
time_std = []
for t in time_list
    push!(time_mean,mean(t))
    push!(time_std,std(t))
end

p = [0.33,0.0048,1.06,0.147,0.33,0.29]

# η0=0.33
# η1=0.0048
# β=1.06
# ϵ=0.147
# a=0.33
# k=0.29
tspan = (24.5,110.0)
X0 = log(0.6)

function f(u,p,t)
    η0,η1,β,ϵ,a,k = p
    x = u
    η0 + η1*t -β*exp(a*x)/(exp(a*x)+exp(a*k))
end

function g(u,p,t)
    ϵ = p[4]
    sqrt(2ϵ)
end

prob = SDEProblem(f,g,X0,tspan,p)
#sol = solve(prob,SOSRI(),saveat = [31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5])
#plot(sol.t,exp.(sol.u))

Turing.setadbackend(:forwarddiff)
@model function fitlv(time_mean, time_std,prob)
    Y0 ~ Normal(time_mean[1],time_std[1]) # pick first data point as starting value
    η0 ~ Uniform(0.1,1.7)
    η1 ~ Uniform(0.001,0.02)
    β ~ Uniform(0.5,3.0)
    ϵ ~ Uniform(0.01,0.5)
    a ~ Uniform(0.1,1.0)
    k ~ Uniform(0.05,1.2)
 
    p = [η0,η1,β,ϵ,a,k]
    prob1 = remake(prob, u0=Y0, p=p)
    ensembleprob = EnsembleProblem(prob1)
    predicted = solve(ensembleprob,EM(),EnsembleThreads(),dt=0.1,saveat = [31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5],trajectories=50)
    pred_mean, pred_std = timeseries_steps_meanvar(predicted)
    for j in 2:length(time_mean)
        time_mean[j] ~ Normal(pred_mean.u[j-1],1.0)
        time_std[j] ~ Normal(pred_std.u[j-1],1.0)
    end
end

model = fitlv(time_mean, time_std,prob)

# This next command runs 3 independent chains without using multithreading.
chain = sample(model, NUTS(0.65), 2000, init_theta = [time_mean[1],0.6678,0.0108,2.3401,0.2369,0.6929,0.6958])
#summarystats(chain)
plot(chain)
