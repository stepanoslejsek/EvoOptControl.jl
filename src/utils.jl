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

function create_OCPSolution(prob::OCProblem, sol::Evolutionary.EvolutionaryOptimizationResults)
  t, states, controls = extract_trajectory(prob, sol.minimizer)
  fitness_values = [sol.trace[i].value for i in 1:length(sol.trace)]
  return OCPSolution(fitness_values, sol.minimum, t, states, controls, sol.f_calls)
end

function plot_3d_erd(histories::Vector{OCPSolution})
  min_fitness = minimum(minimum(h.fitness_values[2:end]) for h in histories)
  max_fitness = maximum(maximum(h.fitness_values[2:end]) for h in histories)
  
  n_targets = 100  # Reduced from 2000 for better performance
  targets = range(min_fitness, max_fitness, length=n_targets)
  
  eval_points, _ = compute_erd(histories, targets[1])
  n_evals = length(eval_points)
  probs = zeros(n_evals, length(targets))
  
  # Compute ERD for each target value
  for (i, target) in enumerate(targets)
    _, probabilities = compute_erd(histories, target)
    probs[:, i] = probabilities
  end
  
  fig = Figure()
  ax = Axis3(fig[1, 1],
    title="3D Expected Runtime Distribution",
    xlabel="Number of Function Evaluations",
    ylabel="Target Fitness Value",
    zlabel="Probability",
    elevation=0.3,
    azimuth=0.8
  )
  
  surface!(ax, eval_points, collect(targets), probs,
    colormap=:viridis,
    transparency=false,
  )
  
  Colorbar(fig[1, 2], limits=(0, 1), colormap=:viridis,
    label="Probability of reaching target")
  
  return fig
end

function compute_erd(histories::Vector{OCPSolution}, target::Float64)
  n_runs = length(histories)
  max_evals = maximum(h.f_calls for h in histories)
  n_points = 200
  step = max_evals / n_points
  eval_points = 0:step:max_evals
  
  probabilities = Float64[]
  
  for eval_point in eval_points
    successful_runs = 0
    for history in histories
      # Get fitness values excluding the initial point
      fitness_values = history.fitness_values[2:end]
      eval_indices = range(0, history.f_calls, length=length(fitness_values))
      
      # Check if target was reached by this evaluation point
      for (eval_idx, fitness) in zip(eval_indices, fitness_values)
        if eval_idx > eval_point
          break
        end
        if fitness <= target
          successful_runs += 1
          break
        end
      end
    end
    push!(probabilities, successful_runs / n_runs)
  end
  
  return collect(eval_points), probabilities
end

function simulate_system(dynamics::Function, prob::OCProblem, sol::OCPSolution)
  diffeq_prob = ODEProblem(dynamics, prob.x0, prob.tspan, sol.controls[:,1])
  callback_times = sol.t
  u = sol.controls
  condition(x, t, integrator) = t ∈ callback_times
  affect!(integrator) = integrator.p = u[:, findall(x->x == integrator.t, callback_times)[1]]
  cb = DiscreteCallback(condition, affect!)
  diffeq_sol = solve(diffeq_prob, Rosenbrock23(), callback = cb, tstops = callback_times)
  return diffeq_sol
end

Base.minimum(v::Vector{OCPSolution}) = v[argmin(s.min_value for s in v)]
