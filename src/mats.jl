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
for (type,flag) in [
    (:BlasfeoDmat, :d),
    (:BlasfeoSmat, :s),
    ]
    # size
    @eval Base.size(A::$type) = (A.m, A.n)

    # getindex
    blasfeo_geex1 = Symbol(:blasfeo_, flag, :geex1)
    @eval Base.getindex(A::$type, I::Vararg{Int, 2}) = @ccall blasfeo.$blasfeo_geex1(pointer_from_objref(A)::Ptr{$type}, (I[1]-1)::Cint, (I[2]-1)::Cint)::eltype($type)

    # setindex!
    blasfeo_gein1 = Symbol(:blasfeo_, flag, :gein1)
    @eval Base.setindex!(A::$type, v::T, I::Vararg{Int, 2}) where {T <: Real} = @ccall blasfeo.$blasfeo_gein1(v::eltype($type),pointer_from_objref(A)::Ptr{$type}, (I[1]-1)::Cint, (I[2]-1)::Cint)::eltype($type)

    # similar
    @eval Base.similar(A::$type) = $type(A.m, A.n)
    @eval Base.similar(A::$type,dims::Dims{2}) = $type(dims...)

    # copy
    blasfeo_gecp = Symbol(:blasfeo_, flag, :gecp)
    @eval function Base.copy(A::$type)
        B = similar(A)
        A_ptr = pointer_from_objref(A)
        B_ptr = pointer_from_objref(B)
        @ccall blasfeo.$blasfeo_gecp(
            A.m::Cint, A.n::Cint,
            A_ptr::Ptr{$type}, 0::Cint, 0::Cint,
            B_ptr::Ptr{$type}, 0::Cint, 0::Cint
        )::Cvoid
        return B
    end

    # fill!
    blasfeo_gese = Symbol(:blasfeo_, flag, :gese)
    @eval function Base.fill!(A::$type, b::T) where {T <: Real}
        A_ptr = pointer_from_objref(A)
        @ccall blasfeo.$blasfeo_gese(
            A.m::Cint, A.n::Cint,
            b::eltype($type),
            A_ptr::Ptr{$type}, 0::Cint, 0::Cint,
        )::Cvoid
        return A
    end

    # convert to Matrix and back
    # TODO(@anton) maybe the performance overhead of this means maybe we want only explicit constructors?
    blasfeo_unpack = Symbol(:blasfeo_unpack_, flag, :mat)
    blasfeo_pack = Symbol(:blasfeo_pack_, flag, :mat)
    @eval function Base.convert(::Type{Matrix{eltype($type)}}, A::$type)
        A_ptr = pointer_from_objref(A)
        B = Matrix{eltype($type)}(undef, size(A))
        @ccall blasfeo.$blasfeo_unpack(
            A.m::Cint, A.n::Cint,
            A_ptr::Ptr{$type},
            0::Cint, 0::Cint,
            B::Ptr{eltype($type)},
            A.m::Cint
        )::Cvoid
        return B
    end
    @eval function Base.convert(::Type{$type}, A::Matrix{eltype($type)})
        B = $type(size(A)...)
        B_ptr = pointer_from_objref(B)
        m,n = size(A)
        @ccall blasfeo.$blasfeo_pack(
            m::Cint, n::Cint,
            A::Ptr{eltype($type)}, m::Cint,
            B_ptr::Ptr{$type},
            0::Cint, 0::Cint
        )::Cvoid
        return B
    end
end
