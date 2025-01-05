function solve_ocp(prob::OCProblem, ea_method::Evolutionary.AbstractOptimizer,
                   population_size::Int=100, iterations::Int=1000)

  t, states, controls, weights = discretize_problem(prob)

  N = prob.collocation.N
  nx = prob.state_dim
  nu = prob.control_dim

  n_vars = N*(nx+nu)

  lb = Float64[]
  ub = Float64[]

  state_constraint_count = 0
  control_constraint_count = 0

  for constraint in prob.state_constraints
    if constraint.fun == identity
      append!(lb, repeat(constraint.lb, N))
      append!(ub, repeat(constraint.ub, N))
    else
      state_constraint_count += 1
    end
  end

  for constraint in prob.control_constraints
    if constraint.fun == identity
      append!(lb, repeat(constraint.lb, N))
      append!(ub, repeat(constraint.ub, N))
    else
      control_constraint_flag += 1
    end
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

  if state_constraint_count == 0 && control_constraint_count == 0
    c(x) = [eval_dynamics(prob, x), eval_initial_state(prob, x), eval_final_state(prob, x)]
    lc = zeros(3) .- 0.0001
    uc = zeros(3) .+ 0.0001
    con = WorstFitnessConstraints(lb, ub, lc, uc, c)
  elseif state_constraint_count != 0 && control_constraint_count == 0
    c(x) = [eval_dynamics(prob, x), eval_initial_state(prob, x), eval_final_state(prob, x), eval_state_constr(prob, x)]
    lc = zeros(4) .- 0.0001
    uc = zeros(4) .+ 0.0001
    con = WorstFitnessConstraints(lb, ub, lc, uc, c)
  elseif state_constraint_count == 0 && control_constraint_count != 0
    c(x) = [eval_dynamics(prob, x), eval_initial_state(prob, x), eval_final_state(prob, x), eval_control_constr(prob, x)]
    lc = zeros(4) .- 0.0001
    uc = zeros(4) .+ 0.0001
    con = WorstFitnessConstraints(lb, ub, lc, uc, c)
  else
    c(x) = [eval_dynamics(prob, x), eval_initial_state(prob, x), eval_final_state(prob, x), eval_state_constr(prob, x), eval_control_constr(prob, x)]
    lc = zeros(5) .- 0.0001
    uc = zeros(5) .+ 0.0001
    con = WorstFitnessConstraints(lb, ub, lc, uc, c)
  end

  # *Clever* inicialization (states set to zero except x0 and xf and control to uniform value between boundaries)
  init_pop = vcat(prob.x0, 0.02 * rand(nx*(N-2)) .- 0.01, prob.xf, [rand() * (ub[i] - lb[i]) + lb[i] for i in N*nx+1:n_vars])
  result = Evolutionary.optimize(objective, con, init_pop,  ea_method, ea_options)
  return result
end