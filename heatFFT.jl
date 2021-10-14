using ApproxFun, Sundials, Plots; gr()

S = Fourier()
n = 100
x = points(S, n)
T = ApproxFun.plan_transform(S, n)
Ti = ApproxFun.plan_itransform(S, n)

# Convert the initial condition to Fourier space
u₀ = T*cos.(cos.(x.-0.1)) 
D2 = Derivative(S,2)
L = D2[1:n,1:n]

using LinearAlgebra
# The equation is trivial in Fourier space
heat(du,u,L,t) = mul!(du, L, u) 

prob = ODEProblem(heat, u₀, (0.0,10.0),L)
# Specialize the linear solver on the diagonalness of the Jacobian
sol = solve(prob, CVODE_BDF(linear_solver=:Diagonal); reltol=1e-8,abstol=1e-8)

# The solution is in Fourier space, so use inverse to transform back
plot(x,Ti*sol(0.0)) 
plot!(x,Ti*sol(0.5))
plot!(x,Ti*sol(2.0))
plot!(x,Ti*sol(10.0))
