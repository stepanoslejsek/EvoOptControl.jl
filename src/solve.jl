function solve_ocp(prob::OCProblem, ea_method::Evolutionary.AbstractOptimizer,
                   population_size::Int=100, iterations::Int=1000)

  t, states, controls, weights = discretize_problem(prob)

  N = prob.collocation.N
  nx = prob.state_dim
  nu = prob.control_dim

  n_vars = N*(nx+nu)

  lb = Float64[]
  ub = Float64[]

  for constraint in prob.state_constraints
    append!(lb, repeat(constraint.lb, N))
    append!(ub, repeat(constraint.ub, N))
  end

  for constraint in prob.control_constraints
    append!(lb, repeat(constraint.lb, N))
    append!(ub, repeat(constraint.ub, N))
  end

  if isempty(lb)
    lb = fill(-Inf, n_vars)
  end
  if isempty(ub)
    ub = fill(Inf, n_vars)
  end

  function objective(x)
    t, states, controls, weights = discretize_problem(prob, x)
    cost = eval_cost(prob, x)
    return cost
  end

  ea_options = Evolutionary.Options(iterations=iterations, show_trace=true, store_trace=true)

  c(x) = [eval_dynamics(prob, x), eval_initial_state(prob, x), eval_final_state(prob, x)]
  lc = uc = zeros(3)
  con = WorstFitnessConstraints(lb, ub, lc, uc, c)

  init_pop = zeros(n_vars)
  result = Evolutionary.optimize(objective, con, init_pop, ea_method, ea_options)
  return result
end