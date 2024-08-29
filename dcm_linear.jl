using Turing
using DifferentialEquations
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random
using LinearAlgebra
using PDMats
using Distributions
using StatsPlots
using DSP
using ReverseDiff
using ADTypes
using OptimizationOptimJL: NelderMead
using FFTW
using Turing: Variational
using Optim

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
    [ sum(circshift(y,-i)[1:length(x)] .* x) for i in 0:length(y)-1]
end

function convfft(x,y)
    abs.(ifft(fft(x) .* fft(y)))
end

t_hrf = 0:0.2:32
hrf = HRF.(t_hrf)

plot(t_hrf, hrf)

# Parameters that I'd imagine are reasonable
sim_len = 200
dt = 0.2
nd = Normal(0,1)
dd = -LogNormal(0,1)
iw = Wishart(3.0,PDiagMat([.5,.5]))
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
cm = .-rand(iw)
eigen(cm)
parammap = [
    w11 => cm[1,1],
    w22 => cm[2,2],
    w12 => cm[1,2],
    w21 => cm[2,1],
    γ => 1.0,
]

prob = SDEProblem(de, u0map, (0.0, sim_len), parammap)
sol = solve(prob, EM(), dt=dt)

plot(sol)
sol.t[1:4:end]

fmrix1 = myconv(hrf, sol[1,:])
fmrix2 = myconv(hrf,sol[2,:])

fmrix1_exp = fmrix1[1:4:end]
fmrix2_exp = fmrix2[1:4:end]
fmrix1f = convfft(sol[1,:],hrf)
fmrix2f = convfft(sol[2,:],hrf)

plot(sol.t,sol[2,:])
plot!(sol.t,fmrix2)

@model function fitlinear(x1,x2,dt)
    m ~ Wishart(2, PDiagMat([1,1]))
    for i in 2:length(x1)
        dx = -m * [x1[i-1],x2[i-1]]
        x1[i] ~ Normal(x1[i-1] + dx[1] * dt,sqrt(dt))
        x2[i] ~ Normal(x2[i-1] + dx[2] * dt,sqrt(dt))
    end
end

@model function fitlinearhrf(x1,x2,N,dt)
    m ~ Wishart(3.0,PDiagMat([.1,.1]))

    # latent x1 and x2
    lx1 = tzeros(N)
    lx2 = tzeros(N)

    lx1[1] = 0.0 # initial condition (could be a parameter later)
    lx2[1] = 0.0

    for i in 2:N
        dx1 = m[1,1] * lx1[i-1] + m[1,2] * lx2[i-1]
        dx2 = m[2,2] * lx2[i-1] + m[2,1] * lx1[i-1]
        lx1[i] ~ Normal(lx1[i-1] + dx1 * dt,sqrt(dt))
        lx2[i] ~ Normal(lx2[i-1] + dx2 * dt,sqrt(dt))
    end
    x1 ~ MvNormal(myconv(hrf,lx1)[1:4:N],1.0) # observed width of Gaussian
    x2 ~ MvNormal(myconv(hrf,lx2)[1:4:N],1.0) # may be a parameter also
end

model = fitlinear(sol[1,:],sol[2,:],0.2)
chain = sample(model, NUTS(), 1000)

N = length(sol.t)
modelhrf = fitlinearhrf(fmrix1_exp,fmrix2_exp,N,0.2)

# we first get a MAP estimate to start the chain in a viable spot

# experimenting with Pathfinder - not compatible with 0.33
#fun = optim_function(modelhrf, MAP(); constrained=false)
#dim = length(fun.init())
#pathfinder(fun.func; dim=dim, optimizer=NelderMead(),adtype=AutoReverseDiff())
#result_single = pathfinder(modelhrf; ndraws=100,adtype=AutoReverseDiff())
#result_multi = multipathfinder(modelhrf, 1000; adtype=AutoReverseDiff(), nruns=8)

map_estimate = maximum_a_posteriori(modelhrf, NelderMead(); adtype=AutoReverseDiff())

# Turing 0.32
#map_estimate = optimize(modelhrf, MAP(), NelderMead(adtype=AutoReverseDiff()))

init_values = map_estimate.values.array
init_values[1:4] = [-0.6,-0.6,-1.3,0.27]
chainhrf = sample(modelhrf, NUTS(0.65, adtype=AutoReverseDiff()), 1000; initial_params=init_values, save_state=true)
chn2 = sample(modelhrf, NUTS(0.65, adtype=AutoReverseDiff()), 3000; resume_from=chainhrf)

dcmplot = plot(chainhrf[["m11","m22","m12","m21"]],legend=true)

chrf = Array(chainhrf)
x1pred = chrf[:,5:604]
x2pred = chrf[:,605:1204]

x1mean = vec(mean(x1pred, dims=1))
pushfirst!(x1mean,0.0)
x2mean = vec(mean(x2pred, dims=1))
pushfirst!(x2mean,0.0)

plot(sol.t, fmrix1)
plot!(sol.t, myconv(hrf,x1mean))

idx = 2
plot(sol.t, sol[1,:])
x1pred_idx = x1pred[idx,:]
pushfirst!(x1pred_idx,0.0)
plot(sol.t, fmrix1)
plot!(sol.t, myconv(hrf,x1pred_idx))

advi = ADVI(20, 5000, AutoReverseDiff())
q = vi(modelhrf, advi)
q_sample = rand(q, 5000)
para_sample = q_sample[1:4,:]
para_mean = mean(para_sample, dims=2)
para_std = std(para_sample, dims=2)