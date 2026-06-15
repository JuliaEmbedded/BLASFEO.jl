# NOTE(@anton): Both of these structs are treated as mutable and Julia manages their
#               lifetimes itself. Blasfeo also offers a way to preallocate memory but
#               we still need the bitstype struct and to manage that memory so I think
#               may as well use `blasfeo_allocate_*` and `blasfeo_free_*`

# bits clone of panel major `blasfeo_dmat`
# blasfeo_jll compiles with PANELMAJ so we only need this one and not the column major version
mutable struct BlasfeoDmat <: AbstractMatrix{Cdouble}
    mat::Base.RefValue{blasfeo_dmat}

    function BlasfeoDmat(m::Integer,n::Integer)
        blasfeo_mat = blasfeo_dmat(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        ref_blasfeo_mat = Ref(blasfeo_mat)
        blasfeo_allocate_dmat(m, n, ref_blasfeo_mat)
        mat = new(ref_blasfeo_mat)
        function destructor(this)
            blasfeo_free_dmat(this)
        end
        return finalizer(destructor, mat)
    end

    function BlasfeoDmat(other::Matrix{Cdouble})
        blasfeo_mat = blasfeo_dmat(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        ref_blasfeo_mat = Ref(blasfeo_mat)
        m,n = size(other)
        blasfeo_allocate_dmat(m, n, ref_blasfeo_mat)
        mat = new(ref_blasfeo_mat)
        function destructor(this)
            blasfeo_free_dmat(this)
        end
        blasfeo_pack_dmat(m, n, other, m, mat, 0, 0)
        return finalizer(destructor, mat)
    end
end

# bits clone of panel major `blasfeo_smat`
# blasfeo_jll compiles with PANELMAJ so we only need this one and not the column major version
mutable struct BlasfeoSmat <: AbstractMatrix{Cfloat}
    mat::Base.RefValue{blasfeo_smat}

    function BlasfeoSmat(m::Integer,n::Integer)
        blasfeo_mat = blasfeo_smat(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        ref_blasfeo_mat = Ref(blasfeo_mat)
        blasfeo_allocate_smat(m, n, ref_blasfeo_mat)
        mat = new(ref_blasfeo_mat)
        function destructor(this)
            blasfeo_free_smat(this)
        end
        return finalizer(destructor, mat)
    end

    function BlasfeoSmat(other::Matrix{Cfloat})
        blasfeo_mat = blasfeo_smat(C_NULL,C_NULL,C_NULL,0,0,0,0,0,0)
        ref_blasfeo_mat = Ref(blasfeo_mat)
        m,n = size(other)
        blasfeo_allocate_smat(m, n, ref_blasfeo_mat)
        mat = new(ref_blasfeo_mat)
        function destructor(this)
            blasfeo_free_smat(this)
        end
        blasfeo_pack_smat(m, n, other, m, mat, 0, 0)
        return finalizer(destructor, mat)
    end
end

# basic matrix operations
for (type,flag) in [
    (:BlasfeoDmat, :d),
    (:BlasfeoSmat, :s),
    ]
    # size
    @eval Base.size(A::$type) = (A.mat[].m, A.mat[].n)

    # getindex
    blasfeo_geex1 = Symbol(:blasfeo_, flag, :geex1)
    @eval Base.getindex(A::$type, I::Vararg{Int, 2}) = $blasfeo_geex1(A, I[1]-1, I[2]-1)

    # setindex!
    blasfeo_gein1 = Symbol(:blasfeo_, flag, :gein1)
    @eval Base.setindex!(A::$type, v::T, I::Vararg{Int, 2}) where {T <: Real} = $blasfeo_gein1(v, A, I[1]-1, I[2]-1)

    # similar
    @eval Base.similar(A::$type) = $type(size(A)...)
    @eval Base.similar(A::$type,dims::Dims{2}) = $type(dims...)

    # copy
    blasfeo_gecp = Symbol(:blasfeo_, flag, :gecp)
    @eval function Base.copy(A::$type)
        B = similar(A)
        $blasfeo_gecp(
            size(A,1), size(A,2),
            A, 0, 0,
            B, 0, 0,
        )
        return B
    end

    # fill!
    blasfeo_gese = Symbol(:blasfeo_, flag, :gese)
    @eval function Base.fill!(A::$type, b::T) where {T <: Real}
        $blasfeo_gese(
            size(A,1), size(A,2),
            b,
            A, 0, 0,
        )
        return A
    end

    # convert to Matrix and back
    # TODO(@anton) maybe the performance overhead of this means maybe we want only explicit constructors?
    blasfeo_unpack = Symbol(:blasfeo_unpack_, flag, :mat)
    blasfeo_pack = Symbol(:blasfeo_pack_, flag, :mat)
    @eval function Base.convert(::Type{Matrix{eltype($type)}}, A::$type)
        B = Matrix{eltype($type)}(undef, size(A))
        $blasfeo_unpack(
            size(A,1), size(A,2),
            A, 0, 0,
            B, stride(B),
        )::Cvoid
        return B
    end
    @eval function Base.convert(::Type{$type}, A::Matrix{eltype($type)})
        B = $type(size(A)...)
        $blasfeo_pack(
            size(A,1), size(A,2),
            A, stride(A),
            B, 0, 0,
        )::Cvoid
        return B
    end

    # unsafe_convert to Ptr{blasfeo_[ds]mat} so you can directly pass to the low level c interface
    blasfeo_mat =  Symbol(:blasfeo_, flag, :mat)
    @eval Base.unsafe_convert(::Type{Ptr{$blasfeo_mat}}, A::$type) = Base.unsafe_convert(Ptr{$blasfeo_mat}, A.mat)
end
