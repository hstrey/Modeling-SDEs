using DifferentialEquations
using Plots
using StatsPlots
using DelimitedFiles
using DifferentialEquations
using Turing

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
@model function fitlv(data, time_mean, time_std,prob)
    Y0 ~ Normal(time_mean[1],time_std[1]) # pick first data point as starting value
    η0 ~ Uniform(0.3,2.0)
    η1 ~ Uniform(0.003,0.02)
    β ~ Uniform(0.5,3)
    ϵ ~ Uniform(0.001,0.02)
    a ~ Uniform(0.5,3)
    k ~ Uniform(0.01,0.3)
 
    p = [η0,η1,β,ϵ,a,k]
    prob1 = remake(prob, u0=Y0, p=p)
    predicted = solve(prob1,
    SOSRI(),
    saveat = [31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5])

    for j in 2:length(data)
        for i = 1:length(data[j])
            data[j][i] ~ Normal(predicted[j-1],time_std[j])
        end
    end
end

model = fitlv(time_list, time_mean, time_std,prob)

# This next command runs 3 independent chains without using multithreading.
chain = sample(model, HMC(0.05, 10), 5000, init_theta = [time_mean[1],0.33,0.0048,1.06,0.147,0.33,0.29])
plot(chain)
