function get_collocation_points(method::BackwardEulerCollocation)
  nodes = range(-1, 1, length=method.N)
  return collect(nodes), nothing
end

function get_collocation_points(method::TrapezoidalCollocation)
  nodes = range(-1, 1, length=method.N)
  return collect(nodes), nothing
end

function discretize_problem(prob::OCProblem)
  nodes, weights = get_collocation_points(prob.collocation)
  t0, tf = prob.tspan
  t = @. (tf - t0)/2 * nodes + (tf + t0)/2

  N = prob.collocation.N
  nx = prob.state_dim
  nu = prob.control_dim
  
  x = zeros(N*(nx+nu))

  state_vars = x[1:(N*nx)]
  control_vars = x[(N*nx+1):end]

  states = reshape(state_vars, nx, N)
  controls = reshape(control_vars, nu, N)

  return t, states, controls, weights
end

function discretize_problem(prob::OCProblem, x::Vector{Float64})
  nodes, weights = get_collocation_points(prob.collocation)
  t0, tf = prob.tspan
  t = @. (tf - t0)/2 * nodes + (tf + t0)/2

  N = prob.collocation.N
  nx = prob.state_dim
  nu = prob.control_dim
  
  state_vars = x[1:(N*nx)]
  control_vars = x[(N*nx+1):end]

  states = reshape(state_vars, nx, N)
  controls = reshape(control_vars, nu, N)

  return t, states, controls, weights
end