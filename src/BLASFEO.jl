module BLASFEO
using LinearAlgebra

include("lib/LibBlasfeo.jl")
using .LibBlasfeo

include("utils.jl")
include("vecs.jl")
include("mats.jl")
include("level1.jl")
include("level2.jl")

# Vectors
export BlasfeoDvec, BlasfeoSvec
# Matrices
export BlasfeoDmat, BlasfeoSmat

# Level 1
export axpy!, axpby!, vecmul!, vecmulacc!, vecmuldot!

end # module
