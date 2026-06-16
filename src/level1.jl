for (type,flag) in [
    (:BlasfeoDvec, :d),
    (:BlasfeoSvec, :s),
    ]
    # dot
    blasfeo_dot = Symbol(:blasfeo_, flag, :dot)
    @eval LinearAlgebra.dot(a::$type, b::$type) = $blasfeo_dot(length(a), a, 0, b, 0)

    # axpy!
    blasfeo_axpy = Symbol(:blasfeo_, flag, :axpy)
    @eval function LinearAlgebra.axpy!(α::T, x::$type, y::$type) where {T <: Real}
        $blasfeo_axpy(
            length(x),
            α, x, 0,
            y, 0,
            y, 0,
        )
        return y
    end

    # 3 arg axpy
    @eval function LinearAlgebra.axpy!(α::T, x::$type, y::$type, z::$type) where {T <: Real}
        $blasfeo_axpy(
            length(x),
            α, x, 0,
            y, 0,
            z, 0,
        )
        return z
    end

    # axpby!
    blasfeo_axpby = Symbol(:blasfeo_, flag, :axpby)
    @eval function LinearAlgebra.axpby!(α::T1, x::$type, β::T2, y::$type) where {T1 <: Real, T2 <: Real}
        $blasfeo_axpby(
            length(x),
            α, x, 0,
            β, y, 0,
            y, 0,
        )
        return y
    end

    # 3 arg axpby
    blasfeo_axpby = Symbol(:blasfeo_, flag, :axpby)
    @eval function LinearAlgebra.axpby!(α::T1, x::$type, β::T2, y::$type, z::$type) where {T1 <: Real, T2 <: Real}
        $blasfeo_axpby(
            length(x),
            α, x, 0,
            β, y, 0,
            z, 0,
        )
        return z
    end

    # vecmul
    blasfeo_vecmul = Symbol(:blasfeo_, flag, :vecmul)
    @eval function vecmul!(x::$type, y::$type, z::$type)
        $blasfeo_vecmul(
            length(x),
            x, 0,
            y, 0,
            z, 0,
        )
        return z
    end

    # vecmulacc
    blasfeo_vecmulacc = Symbol(:blasfeo_, flag, :vecmulacc)
    @eval function vecmulacc!(x::$type, y::$type, z::$type)
        $blasfeo_vecmulacc(
            length(x),
            x, 0,
            y, 0,
            z, 0,
        )
        return z
    end

    # vecmuldot
    blasfeo_vecmuldot = Symbol(:blasfeo_, flag, :vecmuldot)
    @eval vecmuldot!(x::$type, y::$type, z::$type) = $blasfeo_vecmuldot(length(x), x, 0, y, 0, z, 0)

    # veccpsc
    blasfeo_veccpsc = Symbol(:blasfeo_, flag, :veccpsc)
    @eval function veccpsc!(α::T, x::$type, y::$type) where {T<:Real}
        $blasfeo_veccpsc(
            length(x),
            α,
            x, 0,
            y, 0,
        )
        return y
    end

    #### unary arithmetic:
    # TODO(@apozharski) all of this is to avoid wierd promotion rules where
    # *(Real, BlasfeoDvec) -> Vector{Float64} !?
    # But is there a better way?

    # unary plus
    @eval Base.:+(x::$type) = x

    # unary minus
    @eval Base.:-(x::$type) = veccpsc!(-1, x. similar(x))

    #### binary arithmetic
    # binary plus
    @eval Base.:+(x::$type, y::$type) = axpy!(1.0, x, y, similar(x))
    @eval Base.:-(x::$type, y::$type) = axpy!(-1.0, y, x, similar(x))
    @eval Base.:*(x::$type, y::$type) = vecmul!(x, y, similar(x))
    @eval Base.:*(x::$type, y::T) where {T<:Real} = veccpsc!(y, x, similar(x))
    @eval Base.:*(x::T, y::$type) where {T<:Real} = veccpsc!(x, y, similar(y))
    @eval Base.:\(x::$type, y::T) where {T<:Real} = veccpsc!(inv(y), x, similar(x))
    @eval Base.:\(x::T, y::$type) where {T<:Real} = veccpsc!(inv(x), y, similar(y))
    @eval Base.:/(x::$type, y::T) where {T<:Real} = veccpsc!(inv(y), x, similar(x))
    @eval Base.:/(x::T, y::$type) where {T<:Real} = veccpsc!(inv(x), y, similar(y))

    # TODO(@apozharski) givens plane rotations
end
