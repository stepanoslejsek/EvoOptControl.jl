using Pkg

Pkg.activate("../../EvoOptControl")
Pkg.instantiate()

using IJulia
notebook(dir=".")