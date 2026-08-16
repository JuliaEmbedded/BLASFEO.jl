@doc """
    axpy!(α, x, y, z)

Along with the standard 2 argument `axpy!(α, x, y)`, `BLASFEO.jl` uses the 3 argument style of the `BLASFEO` api.

This allows a separate output vector `z` rather than overwriting `y`. This results in `z = α * x + y`.
"""
LinearAlgebra.axpy!

@doc """
    axpy!(α, x, β, y, z)

Along with the standard 2 argument `axpby!(α, x, β, y)`, `BLASFEO.jl` uses the 3 argument style of the `BLASFEO` api.

This allows a separate output vector `z` rather than overwriting `y`. This results in `z = α * x + β * y`.
"""
LinearAlgebra.axpby!

@doc """
    axpy!(α, x, y, z)

Along with the standard 2 argument `axpy!(α, x, y)`, `BLASFEO.jl` uses the 3 argument style of the `BLASFEO` api.

This allows a separate output vector `z` rather than overwriting `y`. This results in `z = α * x + y`.
"""
LinearAlgebra.axpy!

@doc """
    vecmul!(x, y, z)

Performs indexwise multiplication of `x` and `y` and stores it in `z`. It produces equivalent results to `z .= x .* y`.
"""
vecmul!

@doc """
    vecmulacc!(x, y, z)

Performs indexwise multiplication of `x` and `y` and adds it elementwise to `z`. It produces equivalent results to `z .+= x .* y`.
"""
vecmulacc!

@doc """
    vecmuldot!(x, y, z)

Performs indexwise multiplication of `x` and `y` and stores it to `z`. It produces equivalent results to `z .+= x .* y`.
Additionally this returns the dot product of `x` and `y`: `x^T * y`.
"""
vecmuldot!

@doc """
    veccpsc!(α, x, y)

Preforms a copy and scale operation: `y .= α * x`.
"""
veccpsc!


for (type,flag) in [
    (:BlasfeoDvec, :d),
    (:BlasfeoSvec, :s),
    ]
    # dot
    blasfeo_dot = Symbol(:blasfeo_, flag, :dot)
    @eval function LinearAlgebra.dot(a::$type, b::$type)
        @checkvvdims a b
        $blasfeo_dot(length(a), a, 0, b, 0)
    end

    # axpy!
    blasfeo_axpy = Symbol(:blasfeo_, flag, :axpy)
    @eval function LinearAlgebra.axpy!(α::T, x::$type, y::$type) where {T <: Real}
        @checkvvdims x y
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
        @checkvvdims x y z
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
        @checkvvdims x y
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
        @checkvvdims x y z
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
        @checkvvdims x y z
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
        @checkvvdims x y z
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
    @eval function vecmuldot!(x::$type, y::$type, z::$type)
        @checkvvdims x y z
        $blasfeo_vecmuldot(length(x), x, 0, y, 0, z, 0)
    end

    # veccpsc
    blasfeo_veccpsc = Symbol(:blasfeo_, flag, :veccpsc)
    @eval function veccpsc!(α::T, x::$type, y::$type) where {T<:Real}
        @checkvvdims x y
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
    @eval Base.:-(x::$type) = veccpsc!(-1, x, similar(x))

    #### binary arithmetic
    # binary plus
    @eval function Base.:+(x::$type, y::$type)
        @checkvvdims x y
        return axpy!(1.0, x, y, similar(x))
    end
    @eval function Base.:-(x::$type, y::$type)
        @checkvvdims x y
        return axpy!(-1.0, y, x, similar(x))
    end
    @eval function Base.:*(x::$type, y::$type)
        @checkvvdims x y
        return vecmul!(x, y, similar(x))
    end
    @eval Base.:*(x::$type, y::T) where {T<:Real} = veccpsc!(y, x, similar(x))
    @eval Base.:*(x::T, y::$type) where {T<:Real} = veccpsc!(x, y, similar(y))
    @eval Base.:\(x::$type, y::T) where {T<:Real} = veccpsc!(inv(y), x, similar(x))
    @eval Base.:\(x::T, y::$type) where {T<:Real} = veccpsc!(inv(x), y, similar(y))
    @eval Base.:/(x::$type, y::T) where {T<:Real} = veccpsc!(inv(y), x, similar(x))
    @eval Base.:/(x::T, y::$type) where {T<:Real} = veccpsc!(inv(x), y, similar(y))

    # TODO(@apozharski) givens plane rotations
end
