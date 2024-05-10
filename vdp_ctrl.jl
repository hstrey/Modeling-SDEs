using ModelingToolkit
using DifferentialEquations


# Define some variables
@parameters θ
@variables t x(t) y(t)
D = Differential(t)

eqs = [D(x) ~ y,
       D(y) ~ θ*(1-x^2)*y - x]

@named vdp = ODESystem(eqs,t,[x,y],[θ],controls=[θ])
jac = generate_jacobian(vdp)
j = eval(jac[1])
j([1,1],1,0)
ctrl_jac = generate_control_jacobian(vdp; expression = Val{false})
cj = eval(ctrl_jac[1])
ctrl_jac[1]([0,3],1,0)

u0map = [
    x => 0.1,
    y => 0.1
]

parammap = [
    θ => 1.0
    ]
