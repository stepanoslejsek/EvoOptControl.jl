function extract_trajectory(prob::OCProblem, x::Vector{Float64})
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

  return t, states, controls
end

function eval_dynamics_backward_euler(prob::OCProblem{BackwardEulerCollocation}, x::Vector{Float64})
  t, states, controls = extract_trajectory(prob, x)

  N = prob.collocation.N
  h = t[2] - t[1]
  diff_derivative = similar(states)

  for i in 1:N-1
    xᵢ = states[:, i]
    xᵢ₊₁ = states[:, i+1]
    uᵢ₊₁ = controls[:, i+1]

    fᵢ₊₁ = prob.dynamics(t[i+1], xᵢ₊₁, uᵢ₊₁)

    diff_derivative[:, i] = xᵢ₊₁ - xᵢ - h*fᵢ₊₁
  end
  return norm(diff_derivative)^2
end

function eval_dynamics_trapezoidal(prob::OCProblem{TrapezoidalCollocation}, x::Vector{Float64})
  t, states, controls = extract_trajectory(prob, x)

  N = prob.collocation.N
  h = t[2] - t[1]
  diff_derivative = similar(states)

  for i in 1:N-1
    xᵢ = states[:, i]
    uᵢ = controls[:, i]
    xᵢ₊₁ = states[:, i+1]
    uᵢ₊₁ = controls[:, i+1]

    fᵢ = prob.dynamics(t[i], xᵢ, uᵢ)
    fᵢ₊₁ = prob.dynamics(t[i+1], xᵢ₊₁, uᵢ₊₁)

    diff_derivative[:, i] = xᵢ₊₁ - xᵢ - (h/2)*(fᵢ + fᵢ₊₁)
  end
  return norm(diff_derivative).^2
end

function eval_cost_backward_euler(prob::OCProblem{BackwardEulerCollocation}, x::Vector{Float64})
  t, states, controls = extract_trajectory(prob, x)
        
  N = prob.collocation.N
  h = t[2] - t[1]

  terminal_cost = prob.terminal_cost(t[end], states[:, end])
  running_cost = 0
  for i in 1:N
    running_cost += h*prob.running_cost(t[i], states[:, i], controls[:, i])
  end
  return running_cost + terminal_cost
end

function eval_cost_trapezoidal(prob::OCProblem{TrapezoidalCollocation}, x::Vector{Float64})
  t, states, controls = extract_trajectory(prob, x)

  N = prob.collocation.N
  h = t[2] - t[1]

  terminal_cost = prob.terminal_cost(t[end], states[:, end])
  running_cost = 0
  for i in 1:N-1
    xᵢ = states[:, i]
    uᵢ = controls[:, i]
    xᵢ₊₁ = states[:, i+1]
    uᵢ₊₁ = controls[:, i+1]
    running_cost += (h/2)*(prob.running_cost(t[i], xᵢ, uᵢ) + prob.running_cost(t[i+1], xᵢ₊₁, uᵢ₊₁))
  end
  return running_cost + terminal_cost
end

function eval_dynamics(prob::OCProblem{T}, x::Vector{Float64}) where T <: FiniteDifferenceCollocationMethod
  if prob.collocation isa BackwardEulerCollocation
    return eval_dynamics_backward_euler(prob, x)
  else
    return eval_dynamics_trapezoidal(prob, x)
  end
end

function eval_cost(prob::OCProblem{T}, x::Vector{Float64}) where T <: FiniteDifferenceCollocationMethod
  if prob.collocation isa BackwardEulerCollocation
    return eval_cost_backward_euler(prob, x)
  else
    return eval_cost_trapezoidal(prob, x)
  end
end

function eval_initial_state(prob::OCProblem, x::Vector{Float64})
  t, states, controls = extract_trajectory(prob, x)
  return norm(prob.x0 - states[:, 1])^2
end

function eval_final_state(prob::OCProblem, x::Vector{Float64})
  t, states, controls = extract_trajectory(prob, x)
  return norm(prob.xf - states[:, end])^2
end
