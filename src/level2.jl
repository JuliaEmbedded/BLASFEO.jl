
for (Mat, Vec, flag) in [
    (:BlasfeoDmat, :BlasfeoDvec, :d),
    (:BlasfeoSmat, :BlasfeoSvec, :s),
    ]
    # Overload matrix-vector multiplication

    blasfeo_gemv_n = Symbol(:blasfeo_, flag, :gemv_n)
    @eval function Base.:*(A::$Mat, x::$Vec)
        @boundscheck begin
            size(A)[2] == length(x) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
        end
        z = similar(x)
        $blasfeo_gemv_n(
            size(A,1), size(A,2),
            1.0, A, 0, 0,
            x, 0,
            0.0, x, 0,
            z, 0,
        )
        return z # A*x
    end

    blasfeo_gemv_t = Symbol(:blasfeo_, flag, :gemv_t)
    @eval function Base.:*(A::Transpose{eltype($Mat), $Mat}, x::$Vec)
        @boundscheck begin
            size(A.parent)[1] == length(x) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
        end

        z = similar(x)
        $blasfeo_gemv_t(
            size(A,1), size(A,2),
            1.0, A, 0, 0,
            x, 0,
            0.0, x, 0,
            z, 0,
        )
        return z # A^T*x
    end
end
