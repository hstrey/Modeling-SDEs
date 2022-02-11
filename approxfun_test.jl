using ForwardDiff, ApproxFun

f = Fun(x->exp(ForwardDiff.Dual(x,1)))
