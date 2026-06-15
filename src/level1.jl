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

    # TODO(@apozharski) givens plane rotations
end
