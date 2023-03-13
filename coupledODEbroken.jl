using ModelingToolkit

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
    eqs = [VP1.jcn ~ VP2.x]
    sys = [VP1,VP2]

    return compose(ODESystem(eqs;name=:connected),sys; name=name)
end

@named VP = van_der_pol_coupled()

eqs = [ VP.VP2₊jcn ~ 0]

@named VPcomp = compose(ODESystem(eqs;name=:connected),[VP])
VPcomps = structural_simplify(VPcomp)