using ModelingToolkit
using Random
using SparseArrays

@variables t
D = Differential(t)
@variables u[1:100](t) = zeros(100)
u = collect(u)

for loop in 1:500
    println(loop);
    A = sprand(100,100,0.1)
    eqs = [D.(u) .~ A*u]
    @named randomODE = ODESystem(eqs[1],t,u,[])
    simply = structural_simplify(randomODE)
    prob = ODEProblem(simply, [], (0.0, 10.0), [])
    du = zeros(100)
    prob.f(du,rand(100),[],0.0)
    println(loop, du);
end
