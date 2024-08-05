using Turing
using DifferentialEquations
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random
using LinearAlgebra
using Distributions
using StatsPlots
using DSP
using ReverseDiff
using ADTypes
using OptimizationOptimJL: NelderMead

# function f = spm_Gpdf(x,h,l)
#     % Probability Density Function (PDF) of Gamma distribution
#     % FORMAT f = spm_Gpdf(x,h,l)
#     %
#     % x - Gamma-variate   (Gamma has range [0,Inf) )
#     % h - Shape parameter (h>0)
#     % l - Scale parameter (l>0)
#     % f - PDF of Gamma-distribution with shape & scale parameters h & l

# function [hrf,p] = spm_hrf(RT,P,T)
#     % Return a hemodynamic response function
#     % FORMAT [hrf,p] = spm_hrf(RT,p,T)
#     % RT   - scan repeat time
#     % p    - parameters of the response function (two Gamma functions)
#     %
#     %                                                           defaults
#     %                                                          (seconds)
#     %        p(1) - delay of response (relative to onset)          6
#     %        p(2) - delay of undershoot (relative to onset)       16
#     %        p(3) - dispersion of response                         1
#     %        p(4) - dispersion of undershoot                       1
#     %        p(5) - ratio of response to undershoot                6
#     %        p(6) - onset (seconds)                                0
#     %        p(7) - length of kernel (seconds)                    32
#     %--------------------------------------------------------------------------
#     dt  = RT/fMRI_T;
#     u   = [0:ceil(p(7)/dt)] - p(6)/dt;
#     hrf = spm_Gpdf(u,p(1)/p(3),dt/p(3)) - spm_Gpdf(u,p(2)/p(4),dt/p(4))/p(5);
#     hrf = hrf([0:floor(p(7)/RT)]*fMRI_T + 1);
#     hrf = hrf'/sum(hrf);

# https://github.com/neurodebian/spm12/blob/master/spm_hrf.m
function HRF(x,rtu=6.0,δ1=6.0,δ2=16.0,τ1=1.0,τ2=1.0,C=0.0)
    return pdf.(Gamma(δ1,τ1),x)-pdf.(Gamma(δ2,τ2),x)/rtu + C
end

function myconv(x,y)
    [ sum(y .* circshift(x, i)) for i in 0:length(x)-1]
end

t_hrf = 0:0.1:60
hrf = HRF.(t_hrf)

plot(t_hrf, hrf)

# Parameters that I'd imagine are reasonable
sim_len = 60.0
dt = 0.1
nd = Normal(0,1)
dd = -Gamma(5,1)
# Define some variables
@parameters w11 w12 w22 w21 γ
@variables x1(t) x2(t)
@brownian a b

eqs = [D(x1) ~ w11 * x1 + w12 * x2 + γ * a,
    D(x2) ~ w22 * x2 + w21 * x1 + γ * b]

@mtkbuild de = System(eqs, t)

u0map = [
    x1 => 0.0,
    x2 => 0.0
]

parammap = [
    w11 => rand(dd),
    w22 => rand(dd),
    w12 => rand(nd),
    w21 => rand(nd),
    γ => 1.0,
]

prob = SDEProblem(de, u0map, (0.0, sim_len), parammap)
sol = solve(prob, EM(), dt=dt)

plot(sol)

fmrix1 = myconv(sol[1,:],hrf)
fmrix2 = myconv(sol[2,:],hrf)

plot(sol.t,sol[2,:])
plot!(sol.t,fmrix2)

@model function fitlinear(x1,x2,dt)
    m11 ~ -Gamma(5,1) + 1
    m22 ~ -Gamma(5,1) + 1
    m12 ~ Normal(0,1)
    m21 ~ Normal(0,1)

    for i in 2:length(x1)
        dx1 = m11 * x1[i-1] + m12 * x2[i-1]
        dx2 = m22 * x2[i-1] + m21 * x1[i-1]
        x1[i] ~ Normal(x1[i-1] + dx1 * dt,sqrt(dt))
        x2[i] ~ Normal(x2[i-1] + dx2 * dt,sqrt(dt))
    end
end

@model function fitlinearhrf(x1,x2,dt)
    m11 ~ -Gamma(5,1) + 1
    m22 ~ -Gamma(5,1) + 1
    m12 ~ Normal(0,1)
    m21 ~ Normal(0,1)

    # latent x1 and x2
    lx1 = tzeros(length(x1))
    lx2 = tzeros(length(x2))

    lx1[1] = 0.0 # initial condition (could be a parameter later)
    lx2[1] = 0.0

    for i in 2:length(x1)
        dx1 = m11 * lx1[i-1] + m12 * lx2[i-1]
        dx2 = m22 * lx2[i-1] + m21 * lx1[i-1]
        lx1[i] ~ Normal(lx1[i-1] + dx1 * dt,sqrt(dt))
        lx2[i] ~ Normal(lx2[i-1] + dx2 * dt,sqrt(dt))
    end
    x1 ~ MvNormal(myconv(hrf,lx1)[1:length(x1)],0.5) # observed width of Gaussian
    x2 ~ MvNormal(myconv(hrf,lx2)[1:length(x2)],0.5) # may be a parameter also
end

model = fitlinear(sol[1,:],sol[2,:],0.1)
chain = sample(model, NUTS(), 1000)

modelhrf = fitlinearhrf(sol[1,:],sol[2,:],0.1)

# we first get a MAP estimate to start the chain in a viable spot
map_estimate = maximum_a_posteriori(modelhrf, NelderMead(); adtype=AutoReverseDiff())
chainhrf = sample(modelhrf, NUTS(0.65, adtype=AutoReverseDiff()), 1000; initial_params=map_estimate.values.array, save_state=true)
chn2 = sample(modelhrf, NUTS(0.65, adtype=AutoReverseDiff()), 3000; resume_from=chainhrf)