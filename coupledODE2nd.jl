using ModelingToolkit
using OrdinaryDiffEq
using Plots

@variables t
D = Differential(t)

function van_der_pol(;name, θ=1.0,ϕ=0.1)
    params  = @parameters θ=θ ϕ=ϕ
    sts = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*(1-x^2)*y - x]

    return ODESystem(eqs,t,sts,params; name=name)
end

function van_der_pol_coupled(;name)
    @named VP1 = van_der_pol()
    @named VP2 = van_der_pol()
    @variables jcn(t)[1:2]
    eqs = [VP1.jcn ~ VP2.x + jcn[2],
        VP2.jcn ~ jcn[1]]
    sys = [VP1,VP2]
    stVP1 = states.((VP1,), states(VP1))
    stVP2 = states.((VP2,), states(VP2))
    sts = vcat(stVP1,stVP2,[jcn[1],jcn[2]])
    paraVP1 = parameters.((VP1,),parameters(VP1))
    paraVP2 = parameters.((VP2,),parameters(VP2))
    params = vcat(paraVP1,paraVP2)
    return compose(ODESystem(eqs,t,sts,params;name=:connected),sys; name=name)
end

@named VP = van_der_pol_coupled()
@named VPmore = van_der_pol_coupled()

eqs = [ VP.jcn[1] ~ VPmore.VP2₊x, VP.jcn[2] ~ 0, VPmore.jcn[1] ~ 0.5*VP.VP1₊x, VPmore.jcn[2] ~ 0]

@named VPcomp = compose(ODESystem(eqs;name=:connected),[VP, VPmore])
VPcomps = structural_simplify(VPcomp)

prob = ODEProblem(VPcomps,[],(0.0,10),[])
sol = solve(prob, Tsit5())
plot(sol)