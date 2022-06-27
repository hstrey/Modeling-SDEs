using Random
using StatsPlots
using Turing
using LinearAlgebra:I
using FFTW

# this is a test program to see whether ForwardDiff can handle FFT in Turing
# we are creating random data and then calculate the power spectrum using the p-Welch method
x = (rand(128) .- 0.5) .* 2.0
xfft = fft(x)
fp = real(xfft .* conj(xfft))[2:63]
plot(fp)

@model function fit_random(data)
    A ~ Normal(2.0,4.0)
    ϵ ~ Uniform(10,1000)
    s = (rand(128) .- 0.5) .* A
    sfft = fft(s)
    ps = real(sfft .* conj(sfft))[2:63]
    data ~ MvNormal(ps,I*ϵ)
end

rmodel = fit_random(fp)
chain = sample(rmodel, NUTS(),2000)