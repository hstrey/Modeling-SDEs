using ForwardDiff, Interpolations

f2(x1,x2)= x1 / x2
F       = Array{Float64,2}(undef,10,2)
F[:,1]  = f2.(1:10,1)
F[:,2]  = f2.(1:10,2)
Z       = interpolate(F, (BSpline(Linear()), BSpline(Linear())), OnGrid())
Z       = scale(Z, 1:10, 1:2)

x1, x2  = ForwardDiff.Dual(3.,1.5), 1
Z[x1, x2]            # returns Dual(3.0,1.5)
ForwardDiff.gradient(Z, x1, x2)  # throws error