using ModelingToolkit

@variables t
D = Differential(t)

@parameters a b
b = ParentScope(b)
p = [a              # a is a local variable
    b]  # b is a variable that belongs to one level up in the hierarchy
sts = @variables x(t), y(t)

level0 = ODESystem(Equation[D(x) ~ a*x + b*y], t, sts, p; name = :level0)
level1 = ODESystem(Equation[D(y) ~ a*x + b*y, x~level0.x, y~level0.y], t, sts, []; name = :level1) ∘ level0
level1 = structural_simplify(level1)
parameters.([level0, level1])
equations.([level0, level1])