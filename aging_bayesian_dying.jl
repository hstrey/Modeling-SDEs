using DifferentialEquations
using Plots
using StatsPlots
using DelimitedFiles
using Turing
using Statistics
using DifferentialEquations.EnsembleAnalysis

# reading in data
experiment = readdlm("Ecoli.csv", ',', Float64)

# create two lists
# time_list is the data for each time point
time_exp = [24.5,31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5]
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
tspan = (24.5,108.5)
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

Turing.setadbackend(:forwarddiff)
@model function fitlv(time_mean, time_std,prob)
    Y0 ~ Normal(0.0420,0.7409) # pick first data point as starting value
    η0 ~ Uniform(0.1,1.7)
    η1 ~ Uniform(0.001,0.02)
    β ~ Uniform(0.5,3.0)
    ϵ ~ Uniform(0.01,0.5)
    a ~ Uniform(0.1,1.0)
    k ~ Uniform(0.05,1.2)
 
    p = [η0,η1,β,ϵ,a,k]
    prob1 = remake(prob, u0=Y0, p=p)
    ensembleprob = EnsembleProblem(prob1)
    predicted = solve(ensembleprob,EM(),EnsembleThreads(),dt=0.1,saveat = [31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5],trajectories=500)

    # predicted contains 50 samples of the SDE
    # but we need to take into account that a cell with damage larger than 4 dies
    # and further values cannot contribute to the average
    # lets first create the array of the correct length
    pred_sum = zeros(eltype(predicted[1].u[1]),length(time_exp)-1)
    pred_sumsq = zeros(eltype(predicted[1].u[1]),length(time_exp)-1)
    pred_count = zeros(length(time_exp)-1)
    for i in 1:length(predicted)
        sol = predicted[i].u
        for j in 1:length(sol)
            if sol[j]<4.0
                pred_sum[j] = pred_sum[j] + sol[j]
                pred_sumsq[j] = pred_sumsq[j] + sol[j]^2
                pred_count[j] += 1
            else
                break
            end
        end
    end

    pred_mean = pred_sum ./ pred_count
    pred_std = sqrt.(pred_sumsq ./ pred_count - pred_mean.^2)
    #println("mean: ",pred_mean)
    #println("std: ",pred_std)
    #println("pred_death: ",pred_death)
    for j in 2:length(pred_mean)
        time_mean[j] ~ Normal(pred_mean[j-1],1.0)
        time_std[j] ~ Normal(pred_std[j-1],1.0)
    end
end

model = fitlv(time_mean, time_std,prob)

# This next command runs 3 independent chains without using multithreading.
chain = sample(model, NUTS(0.65), 2000, init_theta = [time_mean[1],0.6336,0.0116,2.2637,0.2636,0.6738,0.6060])
#summarystats(chain)
plot(chain)
