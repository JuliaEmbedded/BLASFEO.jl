
for (El, Mat, Vec, flag) in [
    (:Cdouble, :BlasfeoDmat, :BlasfeoDvec, :d),
    (:Cfloat, :BlasfeoSmat, :BlasfeoSvec, :s),
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
    blasfeo_trmv_lnn = Symbol(:blasfeo_, flag, :trmv_lnn)
    blasfeo_trmv_ltn = Symbol(:blasfeo_, flag, :trmv_ltn)
    blasfeo_trmv_lnu = Symbol(:blasfeo_, flag, :trmv_lnu)
    blasfeo_trmv_ltu = Symbol(:blasfeo_, flag, :trmv_ltu)
    blasfeo_trmv_unn = Symbol(:blasfeo_, flag, :trmv_unn)
    blasfeo_trmv_utn = Symbol(:blasfeo_, flag, :trmv_utn)
    blasfeo_trmv_unu = Symbol(:blasfeo_, flag, :trmv_unu)
    blasfeo_trmv_utu = Symbol(:blasfeo_, flag, :trmv_utu)

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

    @eval function Base.:*(A::Symmetric{$El, $Mat}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::Symmetric{$El, $Mat}, B::$Vec)
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


    @eval function Base.:*(A::LowerTriangular{$El, $Mat}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::LowerTriangular{$El, $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trmv_lnn(
            size(A,1),
            A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    @eval function Base.:*(A::LowerTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::LowerTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trmv_ltn(
            size(A,1),
            A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    @eval function Base.:*(A::UpperTriangular{$El, $Mat}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::UpperTriangular{$El, $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trmv_unn(
            size(A,1),
            A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    @eval function Base.:*(A::UpperTriangular{$El, Transpose{$El, $Mat}}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::UpperTriangular{$El, Transpose{$El, $Mat}}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trmv_utn(
            size(A,1),
            A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    @eval function Base.:*(A::UnitLowerTriangular{$El, $Mat}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::UnitLowerTriangular{$El, $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trmv_lnu(
            size(A,1),A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    # need to implement trmv_unu
    @eval function Base.:*(A::UnitUpperTriangular{$El, $Mat}, x::$Vec)
        z = similar(x)
        return mul!(z,A,x)
    end

    @eval function LinearAlgebra.mul!(Y::$Vec, A::UnitUpperTriangular{$El, $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trmv_unu(
            size(A,1),
            A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end
end
