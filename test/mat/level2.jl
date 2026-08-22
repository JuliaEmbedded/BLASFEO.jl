@testset "Level 2 BLAS Operations" begin
    @testset for (MAT,VEC) in ((BlasfeoDmat,BlasfeoDvec), (BlasfeoSmat,BlasfeoSvec))
        A_orig = rand(eltype(VEC), 100, 100)
        B_orig = rand(eltype(VEC), 95, 95)
        C_orig = rand(eltype(VEC), 100, 50)
        a_orig = rand(eltype(VEC), 100)
        y_orig = rand(eltype(VEC), 100)
        b_orig = rand(eltype(VEC), 90)
        c_orig = rand(eltype(VEC), 50)
        A = copy(A_orig)
        B = copy(B_orig)
        C = copy(C_orig)
        a = copy(a_orig)
        y = copy(y_orig)
        b = copy(b_orig)
        c = copy(c_orig)
        A_blasfeo = MAT(A)
        B_blasfeo = MAT(B)
        C_blasfeo = MAT(C)
        a_blasfeo = VEC(a)
        y_blasfeo = VEC(y)
        b_blasfeo = VEC(b)
        c_blasfeo = VEC(c)

        @test_throws DimensionMismatch A_blasfeo*b_blasfeo
        @test_throws DimensionMismatch B_blasfeo*a_blasfeo
        @test_throws DimensionMismatch mul!(y_blasfeo, B_blasfeo,a_blasfeo)
        @test_throws DimensionMismatch mul!(y_blasfeo, A_blasfeo,b_blasfeo)
        @test_throws DimensionMismatch mul!(b_blasfeo, A_blasfeo,a_blasfeo)

        # Test nondestructively
        @test A*a ≈ A_blasfeo*a_blasfeo
        @test isa(A_blasfeo*a_blasfeo, VEC)
        @test transpose(A)*a ≈ transpose(A_blasfeo)*a_blasfeo
        @test isa(transpose(A_blasfeo)*a_blasfeo, VEC)

        # test mul!
        @test mul!(y,A,a) ≈ mul!(y_blasfeo, A_blasfeo, a_blasfeo)
        @test mul!(y,transpose(A),a) ≈ mul!(y_blasfeo, transpose(A_blasfeo), a_blasfeo)

        # test asymmetric
        @test C*c ≈ C_blasfeo*c_blasfeo
        @test isa(C_blasfeo*c_blasfeo, VEC)
        @test transpose(C)*y ≈ transpose(C_blasfeo)*y_blasfeo
        @test isa(transpose(C_blasfeo)*y_blasfeo, VEC)

        # test symmetric
        A_sym_L = Symmetric(A,:L)
        A_sym_L_blasfeo = Symmetric(A_blasfeo,:L)
        A_sym_U = Symmetric(A,:U)
        A_sym_U_blasfeo = Symmetric(A_blasfeo,:U)

        @test A_sym_L*a ≈ A_sym_L_blasfeo*a_blasfeo
        @test isa(A_sym_L_blasfeo*a_blasfeo, VEC)
        @test A_sym_U*a ≈ A_sym_U_blasfeo*a_blasfeo
        @test isa(A_sym_U_blasfeo*a_blasfeo, VEC)

        # test triangular
        A_lnn = LowerTriangular(A)
        A_lnn_blasfeo = LowerTriangular(A_blasfeo)
        A_unn = UpperTriangular(A)
        A_unn_blasfeo = UpperTriangular(A_blasfeo)
        A_lnu = UnitLowerTriangular(A)
        A_lnu_blasfeo = UnitLowerTriangular(A_blasfeo)
        A_unu = UnitUpperTriangular(A)
        A_unu_blasfeo = UnitUpperTriangular(A_blasfeo)

        # test lnn
        @test A_lnn*a ≈ A_lnn_blasfeo*a_blasfeo
        @test isa(A_lnn_blasfeo*a_blasfeo, VEC)
        # test unn
        @test A_unn*a ≈ A_unn_blasfeo*a_blasfeo
        @test isa(A_unn_blasfeo*a_blasfeo, VEC)
        # test lnu
        @test A_lnu*a ≈ A_lnu_blasfeo*a_blasfeo
        @test isa(A_lnu_blasfeo*a_blasfeo, VEC)
        # test unu
        # TODO(@anton) not implemented upstream
        @test A_unu*a ≈ A_unu_blasfeo*a_blasfeo
        @test isa(A_unu_blasfeo*a_blasfeo, VEC)
    end
end
