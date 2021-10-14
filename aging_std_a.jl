using DifferentialEquations
using Plots

using DifferentialEquations
η0=0.33
η1=0.0048
β=1.06
ϵ=0.147
a=0.33
k=0.29
X0 = 0

f(u,p,t) = η0 + η1*t -β*exp(a*u)/(exp(a*u)+exp(a*k)) 
g(u,p,t) = sqrt(2ϵ)
dt = 0.5
tspan = (0.0,)
prob = SDEProblem(f,g,X0,(0.0,110.0))

sol = solve(prob,SOSRI(),saveat = [24.5,31.5,38.5,45.5,52.5,59.5,66.5,73.5,80.5,87.5,101.5,108.5])
plot(sol.t,exp.(sol.u))

# #We can plot using the classic Euler-Maruyama algorithm as follows:
# sol = solve(prob,EM(),dt=dt)
# plot(sol,plot_analytic=true)

# sol = solve(prob,SRIW1())
# plot(sol,plot_analytic=true)

ensembleprob = EnsembleProblem(prob)
sol = solve(ensembleprob,EnsembleThreads(),trajectories=1000)

using DifferentialEquations.EnsembleAnalysis
summ = EnsembleSummary(sol,0:1:110)
plot(summ,labels="Middle 95%")
summ = EnsembleSummary(sol,0:1:110;quantiles=[0.25,0.75])
plot!(summ,labels="Middle 50%",legend=true)
