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

@named VP1 = van_der_pol()
@named VP2 = van_der_pol()

eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ VP1.x]

sys = [VP1,VP2]

@named coupledVP = compose(ODESystem(eqs;name=:connected),sys)
