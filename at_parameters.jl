using ModelingToolkit

@variables t
D = Differential(t)

function van_der_pol(;name,θ=1.0)
    
    params = @parameters θ=θ
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

# If you want to share parameters you define the parameter and pass it into the
# function that creates the composed system.
# This behavior does not currently works, but the way to get this working
# is to change the behavior of @parameters to allow symbolic defaults
# what I suggest is that if the default is a Num, then θ = ParentScope(θ)
# otherwise create a parameter with the default given.
# By doing this, parameters live in the name space at which they are defined.
@parameters θ=1.0
@named VP = van_der_pol_coupled(θ=θ)

eqs = [ VP.jcn ~ 0]

@named VPcomp = compose(ODESystem(eqs;name=:connected),[VP])
VPcomps = structural_simplify(VPcomp)

parameters.((VPcomps,), parameters(VPcomps))
states.((VPcomps, ), states(VPcomps))
equations(VPcomps)
