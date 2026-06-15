using Clang.Generators
using blasfeo_jll

cd(@__DIR__)

include_dir = normpath(blasfeo_jll.artifact_dir, "blasfeo/include")

# wrapper generator options
options = load_options(joinpath(@__DIR__, "generator.toml"))

# add compiler flags
args = get_default_args()
#push!(args, "-I$include_dir")
push!(args, "-DMF_PANELMAJ")
push!(args, "-DLA_HIGH_PERFORMANCE")
push!(args, "-DEXT_DEP_MALLOC")
push!(args, "-DEXT_DEP")
push!(args, "-fparse-all-comments") # Parse all comments!

# Only include headers defining the blasfeo API
# TODO(@anton) it may be useful to expose the kernels in the future, per conversation with @perrutquist.
headers = [
    joinpath(include_dir, "blasfeo_d_blasfeo_api.h"), joinpath(include_dir, "blasfeo_s_blasfeo_api.h"),
    joinpath(include_dir, "blasfeo_d_aux.h"), joinpath(include_dir, "blasfeo_s_aux.h"),
    joinpath(include_dir, "blasfeo_d_aux_ext_dep.h"), joinpath(include_dir, "blasfeo_s_aux_ext_dep.h"),

]

# create context
ctx = create_context(headers, args, options)

# run generator
build!(ctx)
