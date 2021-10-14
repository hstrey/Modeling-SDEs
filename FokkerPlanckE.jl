# here I am trying to implement a Fokker-Planck equation (FPE) solver
# the FPE is a partial differential equation
# I would like to start with the Ornstein-Uhlenbeck since the solution to
# the FPE is known
# I am using https://nextjournal.com/sosiris-de/pde-2018 as a guide
# Second order approximation to the second derivative

using ApproxFun, Sundials, Plots, DifferentialEquations, ForwardDiff; gr()
ForwardDiff.can_dual(::Type{ComplexF64}) = true
S = Fourier(-20π..20π)
n = 100
x = points(S, n)
T = ApproxFun.plan_transform(S, n)
Ti = ApproxFun.plan_itransform(S, n)

# Convert the initial condition to Fourier space
u₀ = T*exp.(-((x .- 5).^2)) /sqrt(2π)
D1 = Derivative(S,1)
L = D1[1:n,1:n]

using LinearAlgebra
# The OU Fokker-Planck equation is easier in Fourier space
function ou_fp!(du,u,p,t)
    γ,D,L,x = p
    du1 = similar(u)
    mul!(du1, L, u)
    du1 = -γ .* du1 .* x
    du2 = -D .* x .^ 2 .* u
    du = du1 .+ du2
end

p = [1.0,1.0,L,x]
prob = ODEProblem(ou_fp!, u₀, (0.0,100.0),p)
# Specialize the linear solver on the diagonalness of the Jacobian
sol = solve(prob, KenCarp4(); reltol=1e-8,abstol=1e-8)

# The solution is in Fourier space, so use inverse to transform back
#plot(x,Ti*sol(0.0)) 
plot!(x,Ti*sol(0.5))
plot!(x,Ti*sol(2.0))
plot!(x,Ti*sol(100.0))
