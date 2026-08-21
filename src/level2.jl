
for (Mat, Vec, flag) in [
    (:BlasfeoDmat, :BlasfeoDvec, :d),
    (:BlasfeoSmat, :BlasfeoSvec, :s),
    ]
    # Overload matrix-vector multiplication

    blasfeo_gemv_n  = Symbol(:blasfeo_, flag, :gemv_n)
    blasfeo_gemv_t  = Symbol(:blasfeo_, flag, :gemv_t)
    blasfeo_gemm_nn = Symbol(:blasfeo_, flag, :gemm_nn)
    blasfeo_gemm_nt = Symbol(:blasfeo_, flag, :gemm_nt)
    blasfeo_gemm_tn = Symbol(:blasfeo_, flag, :gemm_tn)
    blasfeo_gemm_tt = Symbol(:blasfeo_, flag, :gemm_tt)
    blasfeo_symv_l = Symbol(:blasfeo_, flag, :symv_l)
    blasfeo_symv_u = Symbol(:blasfeo_, flag, :symv_u)

    @eval function Base.:*(A::$Mat, x::$Vec)
        @boundscheck begin
            size(A)[2] == length(x) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
        end
        z = similar(x, size(A,1))
        $blasfeo_gemv_n(
            size(A,1), size(A,2),
            1.0, A, 0, 0,
            x, 0,
            0.0, x, 0,
            z, 0,
        )
        return z # A*x
    end

    @eval function Base.:*(A::Transpose{eltype($Mat), $Mat}, x::$Vec)
        @boundscheck begin
            size(A,2) == length(x) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
        end

        z = similar(x,size(A,1))
        $blasfeo_gemv_t(
            size(A,2), size(A,1),
            1.0, A, 0, 0,
            x, 0,
            0.0, x, 0,
            z, 0,
        )
        return z # A^T*x
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

    @eval function LinearAlgebra.mul!(Y::$Vec, A::Transpose{eltype($Mat), $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_gemv_t(
            size(A,1), size(A,2),
            1.0, A, 0, 0,
            B, 0,
            0.0, B, 0,
            Y, 0,
        )
        return Y
    end


    @eval function Base.:*(A::Symmetric{eltype($Mat), $Mat}, x::$Vec)
        @boundscheck begin
            size(A,1) == length(x) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
        end

        z = similar(x)
        if A.uplo == 'L'
            $blasfeo_symv_l(
                size(A,1),
                1.0, A.data, 0, 0,
                x, 0,
                0.0, x, 0,
                z, 0,
            )
        else
            $blasfeo_symv_u(
                size(A,1),
                1.0, A.data, 0, 0,
                x, 0,
                0.0, x, 0,
                z, 0,
            )
        end
        return z # A^T*x
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::Symmetric{eltype($Mat), $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        if A.uplo == 'L'
            $blasfeo_symv_l(
                size(A,1),
                1.0, A.data, 0, 0,
                B, 0,
                0.0, B, 0,
                Y, 0,
            )
        else
            $blasfeo_symv_u(
                size(A,1),
                1.0, A.data, 0, 0,
                B, 0,
                0.0, B, 0,
                Y, 0,
            )
        end
        return Y
    end

    
end
