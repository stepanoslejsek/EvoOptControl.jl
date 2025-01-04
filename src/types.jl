abstract type CollocationMethod end
abstract type Constraint end
abstract type FiniteDifferenceCollocationMethod <: CollocationMethod end
abstract type PseudoSpectralCollocationMethod <: CollocationMethod end

struct RadauCollocation <: PseudoSpectralCollocationMethod
  Nᵏ::Int
end

struct BackwardEulerCollocation <: FiniteDifferenceCollocationMethod
  N::Int
end

struct TrapezoidalCollocation <: FiniteDifferenceCollocationMethod
  N::Int
end

struct StateConstraint <: Constraint
  fun::Function
  lb::Vector{Float64}
  ub::Vector{Float64}
end

struct ControlConstraint <: Constraint
  fun::Function
  lb::Vector{Float64}
  ub::Vector{Float64}
end

struct OCProblem{T<:CollocationMethod}
  running_cost::Function
  terminal_cost::Function
  tspan::Tuple{Float64, Float64}
  dynamics::Function
  x0::Vector{Float64}
  xf::Vector{Float64}
  state_dim::Int
  control_dim::Int
  collocation::T
  state_constraints::Vector{StateConstraint}
  control_constraints::Vector{ControlConstraint}
end

struct OCPSolution
  fitness_values::Vector{Float64}
  min_value::Float64
  t::Vector{Float64}
  states::Matrix{Float64}
  controls::Matrix{Float64}
  f_calls::Int
end



