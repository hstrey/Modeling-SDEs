apply_to_variables(f::F, ex) where {F} = _apply_to_variables(f, ex)
apply_to_variables(f::F, ex::Num) where {F} = wrap(_apply_to_variables(f, unwrap(ex)))
function _apply_to_variables(f::F, ex) where {F}
    if isvariable(ex)
        return f(ex)
    end
    istree(ex) || return ex
    similarterm(ex, _apply_to_variables(f, operation(ex)),
        map(Base.Fix1(_apply_to_variables, f), arguments(ex)),
        metadata = metadata(ex))
end

abstract type SymScope end

struct LocalScope <: SymScope end
function LocalScope(sym::Union{Num, Symbolic})
    # this do syntax is equivalent to: apply_to_variables(f::F, ex)
    # where f = setmetadata(sym, SymScope, LocalScope()) and ex = sym
    # https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments
    apply_to_variables(sym) do sym
        setmetadata(sym, SymScope, LocalScope())
    end
end

struct ParentScope <: SymScope
    parent::SymScope
end
function ParentScope(sym::Union{Num, Symbolic})
    apply_to_variables(sym) do sym
        setmetadata(sym, SymScope,
            ParentScope(getmetadata(value(sym), SymScope, LocalScope())))
    end
end

struct DelayParentScope <: SymScope
    parent::SymScope
    N::Int
end
function DelayParentScope(sym::Union{Num, Symbolic}, N)
    apply_to_variables(sym) do sym
        setmetadata(sym, SymScope,
            DelayParentScope(getmetadata(value(sym), SymScope, LocalScope()), N))
    end
end
DelayParentScope(sym::Union{Num, Symbolic}) = DelayParentScope(sym, 1)

struct GlobalScope <: SymScope end
function GlobalScope(sym::Union{Num, Symbolic})
    apply_to_variables(sym) do sym
        setmetadata(sym, SymScope, GlobalScope())
    end
end
