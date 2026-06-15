# NOTE(@anton): Both of these structs are treated as mutable and Julia manages their
#               lifetimes itself. Blasfeo also offers a way to preallocate memory but
#               we still need the bitstype struct and to manage that memory so I think
#               may as well use `blasfeo_allocate_*` and `blasfeo_free_*`

# bits clone of panel major `blasfeo_dvec`
mutable struct BlasfeoDvec <: AbstractVector{Cdouble}
    vec::Base.RefValue{blasfeo_dvec}

    function BlasfeoDvec(m::T) where {T <: Integer}
        blasfeo_vec = blasfeo_dvec(C_NULL,C_NULL,0,0,0)
        ref_blasfeo_vec = Ref(blasfeo_vec)
        blasfeo_allocate_dvec(m, ref_blasfeo_vec)
        vec = new(ref_blasfeo_vec)
        function destructor(this)
            blasfeo_free_dvec(this)
        end
        return finalizer(destructor, vec)
    end

    function BlasfeoDvec(other::Vector{Cdouble})
        blasfeo_vec = blasfeo_dvec(C_NULL,C_NULL,0,0,0)
        ref_blasfeo_vec = Ref(blasfeo_vec)
        m = length(other)
        blasfeo_allocate_dvec(m, ref_blasfeo_vec)
        vec = new(ref_blasfeo_vec)
        function destructor(this)
            blasfeo_free_dvec(this)
        end
        blasfeo_pack_dvec(m, other, 1, vec, 0)::Cvoid
        return finalizer(destructor, vec)
    end
end

# bits clone of panel major `blasfeo_svec`
mutable struct BlasfeoSvec <: AbstractVector{Cfloat}
    vec::Base.RefValue{blasfeo_svec}

    function BlasfeoSvec(m::T) where {T <: Integer}
        blasfeo_vec = blasfeo_svec(C_NULL,C_NULL,0,0,0)
        ref_blasfeo_vec = Ref(blasfeo_vec)
        blasfeo_allocate_svec(m, ref_blasfeo_vec)
        vec = new(ref_blasfeo_vec)
        function destructor(this)
            blasfeo_free_svec(this)
        end
        return finalizer(destructor, vec)
    end

    function BlasfeoSvec(other::Vector{Cfloat})
        blasfeo_vec = blasfeo_svec(C_NULL,C_NULL,0,0,0)
        ref_blasfeo_vec = Ref(blasfeo_vec)
        m = length(other)
        blasfeo_allocate_svec(m, ref_blasfeo_vec)
        vec = new(ref_blasfeo_vec)
        function destructor(this)
            blasfeo_free_svec(this)
        end
        blasfeo_pack_svec(m, other, 1, vec, 0)
        return finalizer(destructor, vec)
    end
end

# basic vector operations
for (type,flag) in [
    (:BlasfeoDvec, :d),
    (:BlasfeoSvec, :s),
    ]
    # size
    @eval Base.size(A::$type) = (A.vec[].m,)

    # getindex
    blasfeo_vecex1 = Symbol(:blasfeo_, flag, :vecex1)
    @eval Base.getindex(A::$type, I::Vararg{Int, 1}) = $blasfeo_vecex1(A, I[1]-1)

    # setindex!
    blasfeo_vecin1 = Symbol(:blasfeo_, flag, :vecin1)
    @eval Base.setindex!(A::$type, v::T, I::Vararg{Int, 1}) where {T <: Real} = $blasfeo_vecin1(v, A, I[1]-1)

    # similar
    @eval Base.similar(A::$type) = $type(length(A))
    @eval Base.similar(A::$type,dims::Dims{1}) = $type(dims...)

    # copy
    blasfeo_veccp = Symbol(:blasfeo_, flag, :veccp)
    @eval function Base.copy(A::$type)
        B = similar(A)
        $blasfeo_veccp(length(A), A, 0, B, 0)
        return B
    end

    # fill!
    blasfeo_vecse = Symbol(:blasfeo_, flag, :vecse)
    @eval function Base.fill!(A::$type, b::T) where {T <: Real}
        $blasfeo_vecse(length(A), b, A, 0)
        return A
    end

    # convert to Vector and back
    # TODO(@anton) maybe the performance overhead of this means maybe we want only explicit constructors?
    blasfeo_unpack = Symbol(:blasfeo_unpack_, flag, :vec)
    blasfeo_pack = Symbol(:blasfeo_pack_, flag, :vec)
    @eval function Base.convert(::Type{Vector{eltype($type)}}, A::$type)
        B = Vector{eltype($type)}(undef, length(A))
        $blasfeo_unpack(length(A), A, 0, B, 1)
        return B
    end
    @eval function Base.convert(::Type{$type}, A::Vector{eltype($type)})
        B = $type(length(A))
        $blasfeo_pack(length(A), A, 1, B, 0)
        return B
    end

    # unsafe_convert to Ptr{blasfeo_[ds]mat} so you can directly pass to the low level c_interface
    blasfeo_vec =  Symbol(:blasfeo_, flag, :vec)
    @eval Base.unsafe_convert(::Type{Ptr{$blasfeo_vec}}, A::$type) = Base.unsafe_convert(Ptr{$blasfeo_vec}, A.vec)
end
