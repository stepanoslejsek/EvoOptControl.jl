using EvoOptControl
using Evolutionary
using CairoMakie

# Pendulum dynamics
function dynamics(t, x, u)
  theta, omega = x
  return [omega; -9.81/1*sin(theta) + u[1]]
end

running_cost(t, x, u) = u[1]^2
terminal_cost(t, x) = 0
tspan = (0.0, .1)

x0 = [0.0, 0.0]
xf = [pi, 0.0]

state_dim = 2
control_dim = 1
method = TrapezoidalCollocation(30)

state_constraints = [StateConstraint(identity, [-2pi, -10.0], [2pi, 10.0])]
control_constraints = [ControlConstraint(identity, [-5.0], [5.0])]

prob = OCProblem(running_cost, terminal_cost, tspan, dynamics, x0, xf, state_dim, control_dim, method, state_constraints, control_constraints)

pop_size = 100
iterations = 100
ea_method = CMAES(mu=100, lambda=200)
# ea_method = ES(initStrategy = IsotropicStrategy(90), srecombination=average, recombination=average, mutation=gaussian, smutation=gaussian, mu=100, rho=40, lambda=200, selection=:comma)

N = 100
history = Vector{OCPSolution}(undef, N)
for i = 1:N
  sol = solve_ocp(prob, ea_method, pop_size, iterations)
  history[i] = create_OCPSolution(prob, sol)
end


