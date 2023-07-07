using Turing, Distributions, Random, Plots

sample1 = rand(Exponential(2.0),100)
sample2 = rand(Exponential(1.0),100)

sampletwo = sample1 .+ sample2

histogram(sampletwo, normalized=true)

@model function twoexp(sample)
    N = length(sample)
    λ_1 ~ Uniform(0.8,1.2)
    λ_2 ~ Uniform(1.8,2.2)
    first = tzeros(N)
    for i in 1:N
        first[i] ~ Exponential(λ_1)
        sample[i] ~ first[i] + Exponential(λ_2)
    end
end

mymodel = twoexp(sampletwo)

chain = sample(mymodel,NUTS(0.65),1000)
