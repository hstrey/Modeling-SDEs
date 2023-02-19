using Plots, Turing
using CSV
using DataFrames
using LsqFit

l1l20 = CSV.read("cr_kep_v8_k2_r3_fitting_history.csv",DataFrame)

l1l2 = filter(:obj => obj -> !isinf(obj), l1l20)

l1 = l1l2[!,:L1]
l2 = l1l2[!,:L2]
z = l1l2[!,:obj]

scatter(l1,l2,marker_z=log.(z))

ll12 = hcat(l1, l2)

function twoD_Gaussian(xy, p)
    amplitude, xo, yo, sigma_x, sigma_y, rho = p

    # creating linear meshgrid from xy
    x = xy[:, 1]
    y = xy[:, 2]
    g = amplitude .* (((x .- xo).^2)/sigma_x^2 - 2 .* rho .* (x .- xo)/sigma_x .* (y .- yo)/sigma_y + ((y .- yo).^2)/sigma_y^2)
    return g[:]
end

function twoDG(x,y,p)
    amplitude, xo, yo, sigma_x, sigma_y, rho = p

    # creating linear meshgrid from xy
    return amplitude * (((x - xo)^2)/sigma_x^2 - 2 * rho * (x - xo)/sigma_x * (y - yo)/sigma_y + ((y - yo)^2)/sigma_y^2)
end

p0 = Float64.([3, 0.5, 0.18, 0.01, 0.01, -0.6])

fit = LsqFit.curve_fit(twoD_Gaussian, ll12, z, p0)
confidence_inter = confidence_interval(fit,0.05)
p = coef(fit)
twoDGp(x,y) = exp(-twoDG(x,y,p))

xr = -0.2:0.01:0.5
yr = -0.2:0.01:0.5

contour(xr,yr,twoDGp,xlabel="L1",ylabel="L2")
savefig("twoDGauss.png")

contour(xr,yr,twoDGp,xlabel="L1",ylabel="L2",linewidth=3,label=nothing)
scatter!(l1,l2,marker_z=log.(z),xlim=(-0.3,0.6),ylim=(-0.3,0.6),label=nothing)