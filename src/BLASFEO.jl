module BLASFEO
using blasfeo_jll
include("vecs.jl")
include("mats.jl")
include("show.jl")

# Vectors
export BlasfeoDvec, BlasfeoSvec
# Matrices
export BlasfeoDmat, BlasfeoSmat

end # module
