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

p=plot(time,experiment[1,:],markershape = :circle, label=false,xlabel="t in hours",ylabel="Damage X")
for i in 2:size(experiment)[1]
    plot!(time,experiment[i,:],markershape = :circle,label=false)
end
savefig("IndividualEcoliData.png")

# since the list of data points are of different length we have
# to do it like that
time_mean = []
time_std = []
for t in time_list
    push!(time_mean,mean(t))
    push!(time_std,std(t))
end

# values of parameters
p = [0.6165,0.0119,2.2335,0.2800,0.6610,0.5933]

# just for reference.  These were the values from the earlier analysis
#η0 = 0.33
#η1 = 0.0048
#β = 1.06
#ε = 0.147
#a = 0.33
#k = 0.29

# we are simulating from 24.5 h to 110 h
tspan = (24.5,110.0)
Y0 = 1.0 # initial starting point

function f(u,p,t)
    η0,η1,β,ϵ,a,k = p
    x = u
    η0 + η1*t -β*exp(a*x)/(exp(a*x)+exp(a*k))
end

function g(u,p,t)
    ϵ = p[4]
    sqrt(2ϵ)
end

# we are setting the range of starting values of using a normal
# distribution from MCMC simulation
starting = Normal(0.0573,0.7395)
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
summ2 = EnsembleSummary(predicted,quantiles=[0.25,0.75])
plot!(summ2,labels="Ensemble 50% perc",legend=true)
plot!(time,time_mean,seriestype = :scatter,yerror=time_std,labels="Y simulated",
xlabel="t in hours",
ylabel="Y")
plot!(time,pred_mean[:],label="Y simulated - dead")
savefig("Model1parameters.png")

plot(time,time_mean,
    seriestype = :scatter,
    yerror=time_std,
    labels=false,
    xlabel="t in hours",
    ylabel="Y")
savefig("AverageEColiData.png")

#dev_square = sum(- (m .- time_mean) .^2 ./ time_std .^ 2)
#println("log likelihood: ",dev_square)
