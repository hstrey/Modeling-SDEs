# Define dispatch for eigen to be used with the Dual type
using LinearAlgebra
using LinearAlgebra: Eigen
using ForwardDiff: Dual
using ForwardDiff
using ChainRules
using ChainRulesCore

# attempt to create a dispatch based on Van der Aa et al. 2007
function LinearAlgebra.eigen(A::Matrix{Dual{T, P, N}}) where {T,P,N}
    @show typeof(A)
    # extract value
    Av = (p->p.value).(A)
    dA = (p->p.partials[1]).(A)
    A_fwd, Adot_ad = frule((ZeroTangent(), copy(dA)), eigen!, copy(Av))
    evalues = Dual.(A_fwd.values,(Adot_ad.values))
    evectors = Dual.(A_fwd.vectors,(Adot_ad.vectors))
    return evalues,evectors
end

p = Dual(2.0, 1.0);
A = [p 0.0;
     p 1];

e,v = eigen(A)



