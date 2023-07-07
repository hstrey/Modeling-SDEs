# code from https://gist.github.com/YingboMa/667c4420cfec9e050188eee3a5285e6f
using ModelingToolkit
@variables t x(..)[1:3]
pars = @parameters p0, q0, v0, d0, p1, q1, v1, d1, d2, beta0, beta1, tau
D = Differential(t)
hist3 = x(t - tau)[3]
x = collect(x(t))
eqs = [D(x[1]) ~ (v0 / (1 + beta0 * (hist3^2))) * (p0 - q0) * x[1] - d0 * x[1]
       D(x[2]) ~ (v0 / (1 + beta0 * (hist3^2))) * (1 - p0 + q0) * x[1] +
                 (v1 / (1 + beta1 * (hist3^2))) * (p1 - q1) * x[2] - d1 * x[2]
       D(x[3]) ~ (v1 / (1 + beta1 * (hist3^2))) * (1 - p1 + q1) * x[2] - d2 * x[3]]
@named sys = ODESystem(eqs, t, x, pars)

using Symbolics: unwrap
using SymbolicUtils
using SymbolicUtils.Code
using ModelingToolkit: isvariable
iv = unwrap(only(independent_variables(sys)))
eqs = equations(sys)
p = parameters(sys)
varss = Set()
for eq in eqs
    ModelingToolkit.vars!(varss, eq)
end
delay_terms = filter(varss) do v
    isvariable(v) || return false
    istree(v) || return false
    if operation(v) === getindex
        v = arguments(v)[1]
    end
    istree(v) || return false
    args = arguments(v)
    length(args) == 1 && !isequal(iv, args[1]) && occursin(iv, args[1])
end |> collect
hh = Sym{Any}(:h)
# HACK
eqs2 = substitute.(eqs,
                  (Dict(delay_terms[1] => term(getindex,
                                               term(hh, p, unwrap(iv - tau), type = Real),
                                               3, type = Real)),))
out = Sym{Any}(:out)
body = SetArray(false, out, getfield.(eqs2, :rhs))
func = Func([out, DestructuredArgs(states(sys)), hh, DestructuredArgs(parameters(sys)), iv],
            [], body)
my_func_expr = toexpr(func)
my_func = eval(my_func_expr)
h(p, t) = ones(3)
tau1 = 1
lags = [tau1]
p0 = 0.2;
q0 = 0.3;
v0 = 1;
d0 = 5;
p1 = 0.2;
q1 = 0.3;
v1 = 1;
d1 = 1;
d2 = 1;
beta0 = 1;
beta1 = 1;
tspan = (0.0, 10.0)
u0 = [1.0, 1.0, 1.0]
p2 = (p0, q0, v0, d0, p1, q1, v1, d1, d2, beta0, beta1, tau1)
using DelayDiffEq
prob = DDEProblem(my_func, u0, h, tspan, p2; constant_lags = lags)
alg = MethodOfSteps(Rodas4())
sol = solve(prob, alg)
