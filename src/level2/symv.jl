for (El, Mat, Vec, flag) in [
    (:Cdouble, :BlasfeoDmat, :BlasfeoDvec, :d),
    (:Cfloat, :BlasfeoSmat, :BlasfeoSvec, :s),
    ]
    # Overload matrix-vector multiplication
    blasfeo_symv_l = Symbol(:blasfeo_, flag, :symv_l)
    blasfeo_symv_u = Symbol(:blasfeo_, flag, :symv_u)

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
end
