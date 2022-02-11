# to test whether Turing.jl can be used we test it on a non-linear SDE
# that has an analytical solution Cox,Ingersol, Ross Econometrica 53,385-407(1985)
# dX = alpha(beta-X)dt + sigma*sqrt(X)dW
# model parameters:
# alpha = 0.2
# beta = 0.08
# sigma = 0.03
# dt = 1/12000 # 1000th of a year
# N = 50000

# Euler Maruyama method
# dW = np.random.normal(loc=0,scale=np.sqrt(dt),size=N)
# X = [0.1]
# for i,onedW in enumerate(dW):
#     n = X[i] + alpha*(beta-X[i])*dt + sigma*np.sqrt(X[i])*onedW
#     X.append(n)
# X = np.array(X)

using DifferentialEquations, Plots
using Turing
using StatsPlots

α = 0.2
β = 0.08
σ = 0.03
u₀=0.1
f(u,p,t) = α*(β-u)
g(u,p,t) = σ*sqrt(u)
dt = 1/12000
tspan = (0.0,4.1)
prob = SDEProblem(f,g,u₀,tspan)
sol = solve(prob,EM(),dt=dt)

plot(sol)

@model function fitcir(data)
    alpha ~ Uniform(0,0.5)
    beta ~ Uniform(0,0.2)
    sigma ~ Uniform(0,0.1)
    for i in 2:length(data)
        xn = data[i-1]
        xnp1 = xn + alpha*(beta-xn)*dt
        xsig = sigma*sqrt(xn)*sqrt(dt)
        data[i] ~ Normal(xnp1,xsig)
    end
end

chain = Turing.sample(fitcir(sol), NUTS(0.65), 1000)
plot(chain)