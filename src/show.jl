# TODO(@anton) mxn or nxm

function Base.show(io, mat::BlasfeoDmat)
    println("$(mat.m)x$(mat.n) BlasfeoDmat:")
    @ccall blasfeo.blasfeo_print_dmat(mat.m::Cint, mat.n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoDmat}, 0::Cint, 0::Cint)::Cvoid
end

function Base.show(io, mat::BlasfeoSmat)
    println("$(mat.m)x$(mat.n) BlasfeoSmat:")
    @ccall blasfeo.blasfeo_print_smat(mat.m::Cint, mat.n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoSmat}, 0::Cint, 0::Cint)::Cvoid
end

function Base.show(io, vec::BlasfeoDvec)
    println("$(vec.m)-element BlasfeoDvec:")
    @ccall blasfeo.blasfeo_print_dvec(vec.m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoDvec}, 0::Cint)::Cvoid
end

function Base.show(io, vec::BlasfeoSvec)
    println("$(vec.m)-element BlasfeoSvec:")
    @ccall blasfeo.blasfeo_print_svec(vec.m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoSvec}, 0::Cint)::Cvoid
end

# TODO(@anton) This is probably not needed, we just need `print_matrix`? this is horribly documented.
for (type,shortname) in [
    (:BlasfeoDvec, :dvec),
    (:BlasfeoSvec, :svec),
    ]
    printer = Symbol(:blasfeo_print_,shortname)
    @eval begin
        function Base.show(io::IO, ::MIME"text/plain", vec::$type)
            println("$(vec.m)-element $($type):")
            @ccall blasfeo.$printer(vec.m::Cint, pointer_from_objref(vec)::Ptr{$type}, 0::Cint)::Cvoid
        end

        function Base.show(io::IO, vec::$type)
            println("$(vec.m)-element $($type):")
            @ccall blasfeo.$printer(vec.m::Cint, pointer_from_objref(vec)::Ptr{$type}, 0::Cint)::Cvoid
        end
    end
end

for (type,shortname) in [
    (:BlasfeoDmat, "dmat"),
    (:BlasfeoSmat, "smat"),
    ]
    printer = Symbol(:blasfeo_print_,shortname)
    @eval begin
        function Base.show(io::IO, ::MIME"text/plain", mat::$type)
            println("$(mat.m)x$(mat.n) $($type):")
            @ccall blasfeo.$printer(mat.m::Cint, mat.n::Cint, pointer_from_objref(mat)::Ptr{$type}, 0::Cint, 0::Cint)::Cvoid
        end

        function Base.show(io::IO, mat::$type)
            println("$(mat.m)x$(mat.n) $($type):")
            @ccall blasfeo.$printer(mat.m::Cint, mat.n::Cint, pointer_from_objref(mat)::Ptr{$type}, 0::Cint, 0::Cint)::Cvoid
        end
    end
end
