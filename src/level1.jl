for (type,flag) in [
    (:BlasfeoDvec, :d),
    (:BlasfeoSvec, :s),
    ]
    # dot
    blasfeo_dot = Symbol(:blasfeo_, flag, :dot)
    @eval function LinearAlgebra.dot(a::$type, b::$type)
        a_ptr=pointer_from_objref(a)
        b_ptr=pointer_from_objref(b)
        @ccall blasfeo.$blasfeo_dot(
            a.m::Cint, a_ptr::Ptr{$type}, 0::Cint,
            b_ptr::Ptr{$type}, 0::Cint,
        )::eltype($type)
    end

    # axpy!
    blasfeo_axpy = Symbol(:blasfeo_, flag, :axpy)
    @eval function LinearAlgebra.axpy!(α::T, x::$type, y::$type) where {T <: Real}
        x_ptr=pointer_from_objref(x)
        y_ptr=pointer_from_objref(y)
        @ccall blasfeo.$blasfeo_axpy(
            x.m::Cint,
            α::Cdouble, x_ptr::Ptr{$type}, 0::Cint,
            y_ptr::Ptr{$type}, 0::Cint,
            y_ptr::Ptr{$type}, 0::Cint,
        )::Cvoid
        return y
    end

    # 3 arg axpby
    @eval function axpy(α::T, x::$type, y::$type, z::$type) where {T <: Real}
        x_ptr=pointer_from_objref(x)
        y_ptr=pointer_from_objref(y)
        z_ptr=pointer_from_objref(z)
        @ccall blasfeo.$blasfeo_axpy(
            x.m::Cint,
            α::Cdouble, x_ptr::Ptr{$type}, 0::Cint,
            y_ptr::Ptr{$type}, 0::Cint,
            z_ptr::Ptr{$type}, 0::Cint,
        )::Cvoid
        return z
    end

    # axpby
    blasfeo_axpby = Symbol(:blasfeo_, flag, :axpby)
    @eval function LinearAlgebra.axpby!(α::T1, x::$type, β::T2, y::$type) where {T1 <: Real, T2 <: Real}
        x_ptr=pointer_from_objref(x)
        y_ptr=pointer_from_objref(y)
        @ccall blasfeo.$blasfeo_axpby(
            a.m::Cint,
            α::Cdouble, x_ptr::Ptr{$type}, 0::Cint,
            β::Cdouble, y_ptr::Ptr{$type}, 0::Cint,
            y_ptr::Ptr{$type}, 0::Cint,
        )::Cvoid
        return y
    end

    # 3 arg axpby
    blasfeo_axpby = Symbol(:blasfeo_, flag, :axpby)
    @eval function axpby(α::T1, x::$type, β::T2, y::$type, z::$type) where {T1 <: Real, T2 <: Real}
        x_ptr=pointer_from_objref(x)
        y_ptr=pointer_from_objref(y)
        z_ptr=pointer_from_objref(z)
        @ccall blasfeo.$blasfeo_axpby(
            x.m::Cint,
            α::Cdouble, x_ptr::Ptr{$type}, 0::Cint,
            β::Cdouble, y_ptr::Ptr{$type}, 0::Cint,
            z_ptr::Ptr{$type}, 0::Cint,
        )::Cvoid
        return z
    end

    # :*
    #@eval function Base.:*
end
