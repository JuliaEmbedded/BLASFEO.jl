# NOTE(@anton): Both of these structs are treated as mutable and Julia manages their
#               lifetimes itself. Blasfeo also offers a way to preallocate memory but
#               we still need the bitstype struct and to manage that memory so I think
#               may as well use `blasfeo_allocate_*` and `blasfeo_free_*`

# bits clone of panel major `blasfeo_dmat`
# blasfeo_jll compiles with PANELMAJ so we only need this one and not the column major version
mutable struct BlasfeoDmat <: AbstractMatrix{Cdouble}
    const mem::Ptr{Cdouble} # pointer to passed chunk of memory
    const pA::Ptr{Cdouble} # pointer to a pm*pn array of doubles, the first is aligned to cache line size
    const dA::Ptr{Cdouble} # pointer to a min(m,n) (or max???) array of doubles
    const m::Cint # rows
    const n::Cint # cols
    const pm::Cint # packed number or rows
    const cn::Cint # packed number or cols
    const use_dA::Cint # flag to tell if dA can be used
    const memsize::Cint # size of needed memory

    function BlasfeoDmat(m::Integer,n::Integer)
        mat = new(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        @ccall blasfeo.blasfeo_allocate_dmat(m::Cint, n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoDmat})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_dmat(pointer_from_objref(this)::Ptr{BlasfeoDmat})::Cvoid
        end
        return finalizer(destructor, mat)
    end

    function BlasfeoDmat(other::Matrix{Cdouble})
        mat = new(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        m,n = size(other)
        @ccall blasfeo.blasfeo_allocate_dmat(m::Cint, n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoDmat})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_dmat(pointer_from_objref(this)::Ptr{BlasfeoDmat})::Cvoid
        end
        @ccall blasfeo.blasfeo_pack_dmat(m::Cint, n::Cint,
                                         other::Ptr{Cdouble}, m::Cint,
                                         pointer_from_objref(mat)::Ptr{BlasfeoDmat},
                                         0::Cint, 0::Cint)::Cvoid
        return finalizer(destructor, mat)
    end
end

# bits clone of panel major `blasfeo_smat`
# blasfeo_jll compiles with PANELMAJ so we only need this one and not the column major version
mutable struct BlasfeoSmat <: AbstractMatrix{Cfloat}
    const mem::Ptr{Cfloat} # pointer to passed chunk of memory
    const pA::Ptr{Cfloat} # pointer to a pm*pn array of doubles, the first is aligned to cache line size
    const dA::Ptr{Cfloat} # pointer to a min(m,n) (or max???) array of doubles
    const m::Cint # rows
    const n::Cint # cols
    const pm::Cint # packed number or rows
    const cn::Cint # packed number or cols
    const use_dA::Cint # flag to tell if dA can be used
    const memsize::Cint # size of needed memory

    function BlasfeoSmat(m::Integer,n::Integer)
        mat = new(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        @ccall blasfeo.blasfeo_allocate_smat(m::Cint, n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoSmat})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_smat(pointer_from_objref(this)::Ptr{BlasfeoSmat})::Cvoid
        end
        return finalizer(destructor, mat)
    end

    function BlasfeoSmat(other::Matrix{Cfloat})
        mat = new(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        m,n = size(other)
        @ccall blasfeo.blasfeo_allocate_smat(m::Cint, n::Cint, pointer_from_objref(mat)::Ptr{BlasfeoSmat})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_smat(pointer_from_objref(this)::Ptr{BlasfeoSmat})::Cvoid
        end
        @ccall blasfeo.blasfeo_pack_smat(m::Cint, n::Cint,
                                         other::Ptr{Cdouble}, m::Cint,
                                         pointer_from_objref(mat)::Ptr{BlasfeoSmat},
                                         0::Cint, 0::Cint)::Cvoid
        return finalizer(destructor, mat)
    end
end

# basic matrix operations
# TODO(@anton) Do we _need_ such tight typing on setindex?
for (type,flag) in [
    (:BlasfeoDmat, :d),
    (:BlasfeoSmat, :s),
    ]
    @eval Base.size(A::$type) = (A.m, A.n)
    blasfeo_geex1 = Symbol(:blasfeo_, flag, :geex1)
    @eval Base.getindex(A::$type, I::Vararg{Int, 2}) = @ccall blasfeo.$blasfeo_geex1(pointer_from_objref(A)::Ptr{$type}, (I[1]-1)::Cint, (I[2]-1)::Cint)::eltype($type)
    @eval Base.similar(A::$type) = $type(A.m, A.n)
    @eval Base.similar(A::$type,dims::Dims{2}) = $type(dims...)

end
