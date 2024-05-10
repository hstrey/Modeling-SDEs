using DifferentialEquations
using Turing
using ReverseDiff
Turing.setadbackend(:reversediff)
using Distributions
using Turing: Variational
using Random
using Statistics

# create Ornstein-Uhlenbeck process time-series
μ = 0.0 # mean is zero
σ = sqrt(2) # D=1
Θ1 = 1.0
W = OrnsteinUhlenbeckProcess(Θ1,μ,σ,0.0,1.0)
prob = NoiseProblem(W,(0.0,100.0))
sol = solve(prob;dt=0.1)

# Ornstein-Uhlenbeck process
@model ou_noise(rn,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl ~ Uniform(0.0,5.0)
    b ~ Beta(5.0,1.0)
    noise_ampl ~ Uniform(0.0,5.0)

    r = R(undef, T)
    r[1] ~ Normal(0,sqrt(ampl))
    for i=2:T
        r[i] ~ Normal(r[i-1]*b,sqrt(ampl*(1-b^2)))
    end
    rn ~ MvNormal(r,sqrt(noise_ampl))
end

modeloun = ou_noise(sol.u,length(sol.u),0.1)
chn = sample(modeloun, NUTS(0.65), 2000)

advi = ADVI(10, 1000)
q = vi(modeloun, advi)
q_sample = rand(q, 1000)

mean(q_sample[1,:]), std(q_sample[1,:])
mean(q_sample[2,:]), std(q_sample[2,:])
mean(q_sample[3,:]), std(q_sample[3,:])
