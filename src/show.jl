# TODO(@anton) mxn or nxm

function Base.show(io, A::BlasfeoDmat)
    println("$(size(A,1))x$(size(A,2)) BlasfeoDmat:")
    blasfeo_print_dmat(
        size(A,1), size(A,2),
        A.mat, 0, 0,
    )
end

function Base.show(io, A::BlasfeoSmat)
    println("$(size(A,1))x$(size(A,2)) BlasfeoSmat:")
    blasfeo_print_smat(
        size(A,1), size(A,2),
        A.mat, 0, 0,
    )
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
    blasfeo_print_mat = Symbol(:blasfeo_print_,shortname)
    @eval begin
        function Base.show(io::IO, ::MIME"text/plain", A::$type)
            println("$(size(A,1))x$(size(A,2)) $($type):")
            $blasfeo_print_mat(
                size(A,1), size(A,2),
                A.mat, 0, 0,
            )
        end

        function Base.show(io::IO, A::$type)
            println("$(size(A,1))x$(size(A,2)) $($type):")
            $blasfeo_print_mat(
                size(A,1), size(A,2),
                A.mat, 0, 0,
            )
        end
    end
end
