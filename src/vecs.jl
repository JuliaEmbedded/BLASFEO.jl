# NOTE(@anton): Both of these structs are treated as mutable and Julia manages their
#               lifetimes itself. Blasfeo also offers a way to preallocate memory but
#               we still need the bitstype struct and to manage that memory so I think
#               may as well use `blasfeo_allocate_*` and `blasfeo_free_*`

# bits clone of panel major `blasfeo_dvec`
mutable struct BlasfeoDvec <: AbstractVector{Cdouble}
	  mem::Ptr{Cdouble} # pointer to passed chunk of memory
	  pa::Ptr{Cdouble} # pointer to a pm array of doubles, the first is aligned to cache line size
	  m::Cint # size
	  pm::Cint # packed size
	  memsize::Cint # size of needed memory

    function BlasfeoDvec(m::T) where {T <: Integer}
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_dvec(m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoDvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_dvec(pointer_from_objref(this)::Ptr{BlasfeoDvec})::Cvoid
        end
        return finalizer(destructor, vec)
    end

    function BlasfeoDvec(other::Vector{Cdouble})
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_dvec(length(other)::Cint, pointer_from_objref(vec)::Ptr{BlasfeoDvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_dvec(pointer_from_objref(this)::Ptr{BlasfeoDvec})::Cvoid
        end

        @ccall blasfeo.blasfeo_pack_dvec(vec.m::Cint,
                                         other::Ptr{Cdouble}, 1::Cint,
                                         pointer_from_objref(vec)::Ptr{BlasfeoDvec},
                                         0::Cint)::Cvoid

        return finalizer(destructor, vec)
    end
end

# bits clone of panel major `blasfeo_svec`
mutable struct BlasfeoSvec <: AbstractVector{Cfloat}
	  mem::Ptr{Cfloat} # pointer to passed chunk of memory
	  pa::Ptr{Cfloat} # pointer to a pm array of floats, the first is aligned to cache line size
	  m::Cint # size
	  pm::Cint # packed size
	  memsize::Cint # size of needed memory

    function BlasfeoSvec(m::T) where {T <: Integer}
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_svec(m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoSvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_svec(pointer_from_objref(this)::Ptr{BlasfeoSvec})::Cvoid
        end
        return finalizer(destructor, vec)
    end

    function BlasfeoSvec(other::Vector{Cfloat})
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_svec(length(other)::Cint, pointer_from_objref(vec)::Ptr{BlasfeoSvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_svec(pointer_from_objref(this)::Ptr{BlasfeoSvec})::Cvoid
        end

        @ccall blasfeo.blasfeo_pack_svec(vec.m::Cint,
                                         other::Ptr{Cfloat}, 1::Cint,
                                         pointer_from_objref(vec)::Ptr{BlasfeoSvec},
                                         0::Cint)::Cvoid

        return finalizer(destructor, vec)
    end
end

# basic vector operations
for (type,flag) in [
    (:BlasfeoDvec, :d),
    (:BlasfeoSvec, :s),
    ]
    # size
    @eval Base.size(A::$type) = (A.m,)

    # getindex
    blasfeo_vecex1 = Symbol(:blasfeo_, flag, :vecex1)
    @eval Base.getindex(A::$type, I::Vararg{Int, 1}) = @ccall blasfeo.$blasfeo_vecex1(pointer_from_objref(A)::Ptr{$type}, (I[1]-1)::Cint)::eltype($type)

    # setindex!
    blasfeo_vecin1 = Symbol(:blasfeo_, flag, :vecin1)
    @eval Base.setindex!(A::$type, v::T, I::Vararg{Int, 1}) where {T <: Real} = @ccall blasfeo.$blasfeo_vecin1(v::eltype($type),pointer_from_objref(A)::Ptr{$type}, (I[1]-1)::Cint)::eltype($type)

    # similar
    @eval Base.similar(A::$type) = $type(A.m)
    @eval Base.similar(A::$type,dims::Dims{1}) = $type(dims...)

    # copy
    blasfeo_veccp = Symbol(:blasfeo_, flag, :veccp)
    @eval function Base.copy(A::$type)
        B = similar(A)
        A_ptr = pointer_from_objref(A)
        B_ptr = pointer_from_objref(B)
        @ccall blasfeo.$blasfeo_veccp(
            A.m::Cint,
            A_ptr::Ptr{$type}, 0::Cint,
            B_ptr::Ptr{$type}, 0::Cint,
        )::Cvoid
        return B
    end

    # fill!
    blasfeo_vecse = Symbol(:blasfeo_, flag, :vecse)
    @eval function Base.fill!(A::$type, b::T) where {T <: Real}
        A_ptr = pointer_from_objref(A)
        @ccall blasfeo.$blasfeo_vecse(
            A.m::Cint,
            b::eltype($type),
            A_ptr::Ptr{$type}, 0::Cint,
        )::Cvoid
        return A
    end
end
