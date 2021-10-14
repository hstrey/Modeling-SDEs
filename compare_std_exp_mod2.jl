using DifferentialEquations
using Plots
using StatsPlots
using DelimitedFiles
using Statistics
using Distributions
using Random

# reading in data
experiment = readdlm("Ecoli.csv", ',', Float64)

# create two lists
# time_list is a list of bacterial damage data for each time point
time = [24.5,31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5]
time_list = []
for time in eachcol(experiment)
    
    cleantime = time[time .> 0.3] # any value below 0.3 is considered bad
    deleteat!(cleantime, findall(isnan,cleantime)) # remove nans
    push!(time_list,log.(cleantime))
end

# since the list of data points are of different length we have
# to do it like that
time_mean = []
time_std = []
for t in time_list
    push!(time_mean,mean(t))
    push!(time_std,std(t))
end

# values of parameters
p = [1.0219,2.4352,0.0103,0.1459,0.6099,0.6535]

# we are simulating from 24.5 h to 110 h
tspan = (24.5,110.0)
Y0 = 0.06 # initial starting point

# here we are using model 2
# (b)  dY/dt = η- (β0 - β1*t) E^(a Y) / [E^(a Y)+E^(a k)] + Sqrt(2ε) ξ
# where  η, β0, β1, ε, a, k are model parameters.

function f(u,p,t)
    η,β0,β1,ϵ,a,k = p
    x = u
    η - (β0 - β1*t)*exp(a*x)/(exp(a*x)+exp(a*k))
end

function g(u,p,t)
    ϵ = p[4]
    sqrt(2ϵ)
end

# we are setting the range of starting values of using a normal
# distribution from MCMC simulation
starting = Normal(0.0197,0.7113)
function prob_func(prob,i,repeat)
    remake(prob,u0=rand(starting))
end

prob = SDEProblem(f,g,Y0,tspan,p) # set up SDE problem
ensembleprob = EnsembleProblem(prob,prob_func=prob_func) # define ensemble with varying starting values
predicted = solve(ensembleprob,SOSRI(),EnsembleThreads(),saveat = [24.5,31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5],trajectories=500)
pred_death = zeros(500,length(time))
for i in 1:length(predicted)
    sol = predicted[i].u
    for j in 1:length(sol)
        if sol[j]<4.0
            pred_death[i,j] = sol[j]
        else
            pred_death[i,j:end] .= NaN
            break
        end
    end
end
nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)
pred_mean = nanmean(pred_death,1)
using DifferentialEquations.EnsembleAnalysis
m,s = timeseries_steps_meanvar(predicted)
summ = EnsembleSummary(predicted)
plot(summ,labels="Ensemble 95% perc")
summ = EnsembleSummary(predicted,quantiles=[0.25,0.75])
plot!(summ,labels="Ensemble 50% perc",legend=true)
plot!(time,time_mean,seriestype = :scatter,yerror=time_std,labels="data")
plot!(time,pred_mean[:])

# at the end we want to calculate the likelihood of the optimal solution
# we can do this both for the unmodified solution and the solution that takes into account death.

#dev_square = sum(- (m .- time_mean) .^2 ./ time_std .^ 2)
#println("log likelihood: ",dev_square)
