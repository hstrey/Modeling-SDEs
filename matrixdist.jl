using Distributions
using LinearAlgebra
using Random
using Plots

m = rand(3,3)
eigen(m)

# construct a diagonally dominant matrix (negative)
a = m - Diagonal([sum(x) for x in eachrow(m)])
eigen(a)
# all eigenvalues are negative

#LKJ matrices are diagonally dominant
for i in 1:10
    lkj = -rand(LKJ(3, 10))
    ev = eigen(lkj).values
    if all(<=(0),ev)
        println(ev)
    end
end

# Can we construct a matrix that is most likely diagonally dominant?
p_unstable = []
for alpha in 1:15
    n_unstable = 0
    n_trial = 1000000
    for i in 1:n_trial
        nd = Normal(0,1)
        m = rand(nd,3,3)
        m[diagind(m)] = -rand(Gamma(alpha,1),3) .+ 1
        ev = real.(eigen(m).values)
        if !all(<=(0),ev)
            #println(ev)
            n_unstable += 1
        end
    end
    push!(p_unstable, n_unstable/n_trial)
end

plot(0:14,log10.(p_unstable),xlabel="mean of gamma distribution",ylabel="log10 prob unstable")
savefig("log10prob.png")

# Can we construct a 2-d matrix that is most likely diagonally dominant?
p_unstable = []
for alpha in 1:15
    n_unstable = 0
    n_trial = 1000000
    for i in 1:n_trial
        nd = Normal(0,1)
        m = rand(nd,2,2)
        m[diagind(m)] = -rand(Gamma(alpha,1),2) .+ 1
        ev = real.(eigen(m).values)
        if !all(<=(0),ev)
            #println(ev)
            n_unstable += 1
        end
    end
    push!(p_unstable, n_unstable/n_trial)
end
plot(0:14,log10.(p_unstable),xlabel="mean of gamma distribution",ylabel="log10 prob unstable")

# suggestion by Fleming to create a custom matrix distribution
function sample_stable_matrix(off_diagonal_scale, dimension)
    off_diagonal_dist = truncated(Normal(), -off_diagonal_scale, off_diagonal_scale)
    diagonal_dist = LogNormal(log(1), 0.5) # tune for margin of stability and breadth
    offset = (dimension - 1) * off_diagonal_scale
    M = zeros(dimension, dimension)
    for i in 1:dimension, j in 1:dimension
        M[i,j] = (i == j) ? offset + rand(diagonal_dist) : rand(off_diagonal_dist)
    end
    return -M 
end