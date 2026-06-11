# TODO(@anton) mxn or nxm

function Base.show(io::IO, mat::BlasfeoDmat)
    println("$(mat.m)x$(mat.n) BlasfeoDmat:")
    @ccall blasfeo.blasfeo_print_dmat(mat.m::Cint, mat.n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoDmat}, 0::Cint, 0::Cint)::Cvoid
end

function Base.show(io::IO, mat::BlasfeoSmat)
    println("$(mat.m)x$(mat.n) BlasfeoSmat:")
    @ccall blasfeo.blasfeo_print_smat(mat.m::Cint, mat.n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoSmat}, 0::Cint, 0::Cint)::Cvoid
end

function Base.show(io::IO, vec::BlasfeoDvec)
    println("$(vec.m)-element BlasfeoDvec:")
    @ccall blasfeo.blasfeo_print_dvec(vec.m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoDvec}, 0::Cint)::Cvoid
end

function Base.show(io::IO, vec::BlasfeoSvec)
    println("$(vec.m)-element BlasfeoSvec:")
    @ccall blasfeo.blasfeo_print_svec(vec.m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoSvec}, 0::Cint)::Cvoid
end
