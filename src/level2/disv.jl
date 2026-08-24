for (El, Mat, Vec, flag) in [
    (:Cdouble, :BlasfeoDmat, :BlasfeoDvec, :d),
    (:Cfloat, :BlasfeoSmat, :BlasfeoSvec, :s),
    ]

    blasfeo_vecmul = Symbol(:blasfeo_, flag, :vecmul)
    
    @eval function Base.:\(A::Diagonal{$El, $Vec}, x::$Vec)
        z = similar(x)
        return ldiv!(z,A,x)
    end

    @eval function LinearAlgebra.ldiv!(Y::$Vec, A::Diagonal{$El, $Vec}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        #TODO(@anton) make a blasfeo kernel for this!
        Y .= $El(1.0)./A.diag
        
        $blasfeo_vecmul(
            size(A,1),
            B, 0,
            Y, 0,
            Y, 0,
        )
        return Y
    end
end
