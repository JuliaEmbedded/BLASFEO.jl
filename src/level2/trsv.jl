for (El, Mat, Vec, flag) in [
    (:Cdouble, :BlasfeoDmat, :BlasfeoDvec, :d),
    (:Cfloat, :BlasfeoSmat, :BlasfeoSvec, :s),
    ]

    blasfeo_trsv_lnn = Symbol(:blasfeo_, flag, :trsv_lnn)
    blasfeo_trsv_ltn = Symbol(:blasfeo_, flag, :trsv_ltn)
    blasfeo_trsv_lnu = Symbol(:blasfeo_, flag, :trsv_lnu)
    blasfeo_trsv_ltu = Symbol(:blasfeo_, flag, :trsv_ltu)
    blasfeo_trsv_unn = Symbol(:blasfeo_, flag, :trsv_unn)
    blasfeo_trsv_utn = Symbol(:blasfeo_, flag, :trsv_utn)
    #blasfeo_trsv_unu = Symbol(:blasfeo_, flag, :trsv_unu)
    #blasfeo_trsv_utu = Symbol(:blasfeo_, flag, :trsv_utu)

    #-------------------------------------------------------------------------------------------------------------------------------#
    @eval function Base.:\(A::LowerTriangular{$El, $Mat}, x::$Vec)
        z = similar(x)
        return ldiv!(z,A,x)
    end

    @eval function LinearAlgebra.ldiv!(Y::$Vec, A::LowerTriangular{$El, $Mat}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end

        $blasfeo_trsv_lnn(
            size(A,1),
            A.data, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end
    #-------------------------------------------------------------------------------------------------------------------------------#

    #-------------------------------------------------------------------------------------------------------------------------------#
    @eval function Base.:\(A::LowerTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
        z = similar(x)
        return ldiv!(z,A,x)
    end

    @eval function LinearAlgebra.ldiv!(Y::$Vec, A::LowerTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end
        $blasfeo_trsv_utn(
            size(A,1),
            A.data.parent, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    #-------------------------------------------------------------------------------------------------------------------------------#
    @eval function Base.:\(A::UpperTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
        z = similar(x)
        return ldiv!(z,A,x)
    end

    @eval function LinearAlgebra.ldiv!(Y::$Vec, A::UpperTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
        @boundscheck begin
            size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
            size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
        end
        $blasfeo_trsv_ltn(
            size(A,1),
            A.data.parent, 0, 0,
            B, 0,
            Y, 0,
        )
        return Y
    end

    if El == :Cdouble

        #-------------------------------------------------------------------------------------------------------------------------------#
        @eval function Base.:\(A::UpperTriangular{$El, $Mat}, x::$Vec)
            z = similar(x)
            return ldiv!(z,A,x)
        end
        @eval function LinearAlgebra.ldiv!(Y::$Vec, A::UpperTriangular{$El, $Mat}, B::$Vec)
            @boundscheck begin
                size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
                size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
            end

            $blasfeo_trsv_unn(
                size(A,1),
                A.data, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end
        #-------------------------------------------------------------------------------------------------------------------------------#

        #-------------------------------------------------------------------------------------------------------------------------------#
        @eval function Base.:\(A::UnitLowerTriangular{$El, $Mat}, x::$Vec)
            z = similar(x)
            return ldiv!(z,A,x)
        end

        @eval function LinearAlgebra.ldiv!(Y::$Vec, A::UnitLowerTriangular{$El, $Mat}, B::$Vec)
            @boundscheck begin
                size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
                size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
            end

            $blasfeo_trsv_lnu(
                size(A,1),
                A.data, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end

        #-------------------------------------------------------------------------------------------------------------------------------#
        @eval function Base.:\(A::UnitUpperTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
            z = similar(x)
            return ldiv!(z,A,x)
        end

        @eval function LinearAlgebra.ldiv!(Y::$Vec, A::UnitUpperTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
            @boundscheck begin
                size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
                size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
            end
            $blasfeo_trsv_ltu(
                size(A,1),
                A.data.parent, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end
        #-------------------------------------------------------------------------------------------------------------------------------#
    end
    # TODO(@anton) need to implement trsv_utu and trsv_unu upstream
end
