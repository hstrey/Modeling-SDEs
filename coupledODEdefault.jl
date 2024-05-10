using ModelingToolkit

function progress_scope(params)
    para_list = []
    for p in params
        pp = ModelingToolkit.unwrap(p)
        if ModelingToolkit.hasdefault(pp)
            d = ModelingToolkit.getdefault(pp)
            @show typeof(d)
            if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
                pp = ParentScope(pp)
                @show pp
            end
        end
        push!(para_list,ModelingToolkit.wrap(pp))
    end
    return para_list
end

@variables t
D = Differential(t)

function van_der_pol(;name,theta=1.0)
    
    params = progress_scope(@parameters θ = theta)
    
    sts = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

    θ = params[1]

    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*(1-x^2)*y - x]

    return ODESystem(eqs,t,sts,params; name=name)
end

function van_der_pol_coupled(;name, theta=1.0)
    @named VP1 = van_der_pol(theta = theta)
    @named VP2 = van_der_pol(theta = theta)
    @variables jcn(t)
    eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ jcn]
    sys = [VP1,VP2]
    
    return compose(ODESystem(eqs,t;name=:connected),sys; name=name)
end

@parameters θ=1.0
@named VP = van_der_pol_coupled(theta=θ)

eqs = [ VP.jcn ~ 0]

@named VPcomp = compose(ODESystem(eqs;name=:connected),[VP])
VPcomps = structural_simplify(VPcomp)

parameters.((VPcomps,), parameters(VPcomps))
states.((VPcomps, ), states(VPcomps))
equations(VPcomps)
