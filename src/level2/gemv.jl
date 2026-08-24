for (El, Mat, Vec, flag) in [
    (:Cdouble, :BlasfeoDmat, :BlasfeoDvec, :d),
    (:Cfloat, :BlasfeoSmat, :BlasfeoSvec, :s),
    ]
    blasfeo_gemv_n  = Symbol(:blasfeo_, flag, :gemv_n)
    blasfeo_gemv_t  = Symbol(:blasfeo_, flag, :gemv_t)

    @eval function Base.:*(A::$Mat, x::$Vec)
        z = similar(x, size(A,1))
        return mul!(z,A,x)
    end

    @eval function Base.:*(A::Transpose{$El, $Mat}, x::$Vec)
        z = similar(x, size(A,1))
        return mul!(z,A,x) # A^T*x
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::$Mat, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end
        $blasfeo_gemv_n(
            size(A,1), size(A,2),
            1.0, A, 0, 0,
            B, 0,
            0.0, B, 0,
            Y, 0,
        )
        return Y
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::Transpose{$El, $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end
        $blasfeo_gemv_t(
            size(A,2), size(A,1),
            1.0, A, 0, 0,
            B, 0,
            0.0, B, 0,
            Y, 0,
        )
        return Y
    end

end
