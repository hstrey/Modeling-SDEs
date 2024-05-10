using ModelingToolkit

@variables t
D = Differential(t)

function van_der_pol(;name,θ=1.0)
    @show typeof(θ)
    if typeof(θ)==Num
        θ = ParentScope(θ)
        params = [θ]
    else
        params = @parameters θ=θ
    end
    sts = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*(1-x^2)*y - x]

    return ODESystem(eqs,t,sts,params; name=name)
end

function van_der_pol_coupled(;name, θ=1.0)
    @named VP1 = van_der_pol(θ=θ)
    @named VP2 = van_der_pol(θ=θ)
    @variables jcn(t)
    eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ jcn]
    sys = [VP1,VP2]

    return compose(ODESystem(eqs;name=:connected),sys; name=name)
end

@parameters θ=1.0
@named VP = van_der_pol_coupled(θ=θ)

eqs = [ VP.jcn ~ 0]

@named VPcomp = compose(ODESystem(eqs;name=:connected),[VP])
VPcomps = structural_simplify(VPcomp)

parameters.((VPcomps,), parameters(VPcomps))
states.((VPcomps, ), states(VPcomps))
equations(VPcomps)
