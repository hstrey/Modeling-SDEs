using ModelingToolkit

@variables t
D = Differential(t)

function progress_scope(params)
    para_list = []
    for p in params
        pp = ModelingToolkit.unwrap(p)
        if ModelingToolkit.hasdefault(pp)
            d = getdefault(pp)
            if istype(d)==Num
                pp = ParentScope(pp)
            end
        end
        push!(para_list,ModelingToolkit.wrap(pp))
    end
    return para_list
end

function van_der_pol(;name, Θ=0.1)
    @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

    eqs = [D(x) ~ y + jcn,
           D(y) ~ Θ*(1-x^2)*y - x]

    return ODESystem(eqs,t; name=name)
end

function van_der_pol_coupled(;name)
 
    @parameters Θc=1.0
    
    @named VP1 = van_der_pol(;Θ=Θc)
    @named VP2 = van_der_pol(;Θ=Θc)
    @variables jcn(t)
    eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ jcn]
    sys = [VP1,VP2]

    return compose(ODESystem(eqs;name=:connected),sys; name=name)
end

@named VP = van_der_pol_coupled()

eqs = [ VP.jcn ~ 0]

@named VPcomp = compose(ODESystem(eqs;name=:connected),[VP])
VPcomps = structural_simplify(VPcomp)