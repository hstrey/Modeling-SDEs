using ForwardDiff, ApproxFun

f = (x->exp(ForwardDiff.Dual(x,1)),-1..1)
