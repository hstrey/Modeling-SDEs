using DelayDiffEq
using Distributions
using Random
using BenchmarkTools

function f!(dθ, θ, h::H, p, t) where H
    ω, A = p
    n = length(θ)
    lags = reshape(lag, n,n)
    @inbounds for j in 1:n
        coupling = 0.0
        @inbounds for i in 1:n
            coupling += A[i,j]*sin(h(p, t-lags[i,j]; idxs=i) - θ[j])
        end
        dθ[j] = ω[j] + coupling
    end
    nothing
end

n = 20
Random.seed!(1)
ω = rand(n)
A = rand(n,n)
const lag = rand(n*n)
θ₀ = rand(Uniform(0, 2π), n)
p = (ω, A)
const past = rand(Uniform(0, 2π), n)
h(p, t; idxs=nothing) = typeof(idxs) <: Number ? past[idxs] : past

prob = DDEProblem(f!, θ₀, h, (0.0, 1.0), p, constant_lags=lag)
@btime solve(prob, MethodOfSteps(BS3()), saveat=0.01, reltol=0.0, abstol=1e-5)
@btime solve(prob, MethodOfSteps(TRBDF2()), saveat=0.01, reltol=0.0, abstol=1e-5)