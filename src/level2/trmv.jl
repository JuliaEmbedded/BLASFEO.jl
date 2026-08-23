for (El, Mat, Vec, flag) in [
    (:Cdouble, :BlasfeoDmat, :BlasfeoDvec, :d),
    (:Cfloat, :BlasfeoSmat, :BlasfeoSvec, :s),
    ]

    blasfeo_trmv_lnn = Symbol(:blasfeo_, flag, :trmv_lnn)
    blasfeo_trmv_ltn = Symbol(:blasfeo_, flag, :trmv_ltn)
    blasfeo_trmv_lnu = Symbol(:blasfeo_, flag, :trmv_lnu)
    blasfeo_trmv_ltu = Symbol(:blasfeo_, flag, :trmv_ltu)
    blasfeo_trmv_unn = Symbol(:blasfeo_, flag, :trmv_unn)
    blasfeo_trmv_utn = Symbol(:blasfeo_, flag, :trmv_utn)
    # blasfeo_trmv_unu = Symbol(:blasfeo_, flag, :trmv_unu)
    # blasfeo_trmv_utu = Symbol(:blasfeo_, flag, :trmv_utu)

    #-------------------------------------------------------------------------------------------------------------------------------#
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
    #-------------------------------------------------------------------------------------------------------------------------------#

    if El == :Cdouble # TODO(@anton) blasfeo_strmv_unn and blasfeo_strmv_lnu are unimplemented :(
        #-------------------------------------------------------------------------------------------------------------------------------#
        @eval function Base.:*(A::LowerTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
            z = similar(x)
            return mul!(z,A,x)
        end

        @eval function LinearAlgebra.mul!(Y::$Vec, A::LowerTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
            @boundscheck begin
                size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
                size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
            end
            $blasfeo_trmv_utn(
                size(A,1),
                A.data.parent, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end
        #-------------------------------------------------------------------------------------------------------------------------------#

        #-------------------------------------------------------------------------------------------------------------------------------#
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
        #-------------------------------------------------------------------------------------------------------------------------------#

        #-------------------------------------------------------------------------------------------------------------------------------#
        @eval function Base.:*(A::UpperTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
            z = similar(x)
            return mul!(z,A,x)
        end

        @eval function LinearAlgebra.mul!(Y::$Vec, A::UpperTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
            @boundscheck begin
                size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
                size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
            end
            $blasfeo_trmv_ltn(
                size(A,1),
                A.data.parent, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end
        #-------------------------------------------------------------------------------------------------------------------------------#

        #-------------------------------------------------------------------------------------------------------------------------------#
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
                size(A,1),
                A.data, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end
        #-------------------------------------------------------------------------------------------------------------------------------#

        #-------------------------------------------------------------------------------------------------------------------------------#
        @eval function Base.:*(A::UnitUpperTriangular{$El, Transpose{$El,$Mat}}, x::$Vec)
            z = similar(x)
            return mul!(z,A,x)
        end

        @eval function LinearAlgebra.mul!(Y::$Vec, A::UnitUpperTriangular{$El, Transpose{$El,$Mat}}, B::$Vec)
            @boundscheck begin
                size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
                size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
            end
            $blasfeo_trmv_ltu(
                size(A,1),
                A.data.parent, 0, 0,
                B, 0,
                Y, 0,
            )
            return Y
        end
        #-------------------------------------------------------------------------------------------------------------------------------#
    end

    # need to implement trmv_unu
    # @eval function Base.:*(A::UnitUpperTriangular{$El, $Mat}, x::$Vec)
    #     z = similar(x)
    #     return mul!(z,A,x)
    # end

    # @eval function LinearAlgebra.mul!(Y::$Vec, A::UnitUpperTriangular{$El, $Mat}, B::$Vec)
    #     @boundscheck begin
    #         size(A,2) == length(B) || throw(DimensionMismatch("Matrix second dimension doesn't match vector dimension"))
    #         size(A,1) == length(Y) || throw(DimensionMismatch("Matrix first dimension doesn't match output vector dimension"))
    #     end

    #     $blasfeo_trmv_unu(
    #         size(A,1),
    #         1.0, A.data, 0, 0,
    #         B, 0,
    #         0.0, B, 0,
    #         Y, 0,
    #     )
    #     return Y
    # end
end
