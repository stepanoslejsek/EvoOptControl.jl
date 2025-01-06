module EvoOptControl

using Evolutionary
using FastGaussQuadrature
using LinearAlgebra
using Distributions
using DifferentialEquations

include("types.jl")
include("collocation.jl")
include("solve.jl")
include("utils.jl")

export TrapezoidalCollocation, BackwardEulerCollocation, RadauCollocation, StateConstraint, ControlConstraint, OCProblem, OCPSolution
export plot_3d_erd, create_OCPSolution, solve_ocp, plot_results

end # module EvoOptControl
