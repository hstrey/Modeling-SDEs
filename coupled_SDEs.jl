using ModelingToolkit, DifferentialEquations, Plots

@variables t
D = Differential(t)

function van_der_pol(;name, θ=1.0)
    @parameters θ=θ
    @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
    @brownian ϕ

    eqs = [D(x) ~ y + jcn + ϕ,
           D(y) ~ θ*(1-x^2)*y - x + ϕ]

    return System(eqs, t; name=name)
end

@named VP1 = van_der_pol()
@named VP2 = van_der_pol()

eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ VP1.x]

sys = [VP1,VP2]

@named connected = System(eqs,t)
@named coupledVP = compose(System(eqs,t;name=:connected),sys)
coupledVPs = structural_simplify(coupledVP)

prob = SDEProblem(coupledVPs, [], (0, 2.0))
sol = solve(prob)
plot(sol, title="Coupled van der Pol SDEs")