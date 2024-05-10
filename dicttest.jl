using ModelingToolkit

Para_dict = Dict{Symbol,Union{Real,Num}}

function scope_dict(para_dict::Para_dict)
    for (n,v) in para_dict
        if typeof(v) == Num
            para_dict_copy[n] = ParentScope(v)
        else
            para_dict_copy[n] = (@parameters $n=v)[1]
        end
    end
    return para_dict_copy
end

function scope_dict2!(para_dict::Para_dict)
    Para_dict(typeof(v) == Num ? n => v : n => (@parameters $n=v)[1] for (n,v) in para_dict)
end

function HarmonicOscillatorBlox(;name, ω=25*(2*pi), ζ=1.0, k=625*(2*pi), h=35.0)
    para_dict = scope_dict!(Para_dict(:ω => ω,:ζ => ζ,:k => k,:h => h))
    ω=para_dict[:ω]
    ζ=para_dict[:ζ]
    k=para_dict[:k]
    h=para_dict[:h]
    @show typeof.(values(para_dict))
end

@parameters ω = 1.0
@named ho = HarmonicOscillatorBlox(ω=ω)

function changedict(dict)
    dict[1] = 10
end

dict = Dict(1 => 2, 2 => 3)
changedict(dict)
dict