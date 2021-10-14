using DifferentialEquations, DiffEqFlux, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

function lotka_volterra_noise!(du, u, p, t)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end

u0 = [1.0,1.0]
tspan = (0.0, 10.0)
p = [2.2, 1.0, 2.0, 0.4]
prob_sde = SDEProblem(lotka_volterra!, lotka_volterra_noise!, u0, tspan)


function predict_sde(p)
  return Array(solve(prob_sde, SOSRI(), p=p,
               sensealg = ForwardDiffSensitivity(), saveat = 0.1))
end

loss_sde(p) = sum(abs2, x-1 for x in predict_sde(p))

callback = function (p, l)
    display(l)
    remade_solution = solve(remake(prob_sde, p = p), SOSRI(), saveat = 0.1)
    plt = plot(remade_solution, ylim = (0, 6))
    display(plt)
    return false
end

result_sde = DiffEqFlux.sciml_train(loss_sde, p, ADAM(0.1),
    cb = callback, maxiters = 500)
