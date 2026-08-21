@testset "Level 2 BLAS Operations" begin
    @testset for (MAT,VEC) in ((BlasfeoDmat,BlasfeoDvec), (BlasfeoSmat,BlasfeoSvec))
        A = rand(eltype(VEC), 100, 100)
        B = rand(eltype(VEC), 95, 95)
        a = rand(eltype(VEC), 100)
        y = rand(eltype(VEC), 100)
        b = rand(eltype(VEC), 90)
        A_blasfeo = MAT(A)
        B_blasfeo = MAT(B)
        a_blasfeo = VEC(a)
        y_blasfeo = VEC(y)
        b_blasfeo = VEC(b)

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
        @test A*a ≈ mul!(y_blasfeo, A_blasfeo, a_blasfeo)
        @test transpose(A)*a ≈ mul!(y_blasfeo, transpose(A_blasfeo), a_blasfeo)

        # test asymmetric
        A = rand(eltype(VEC), 100, 50)
        a = rand(eltype(VEC), 50)
        y = rand(eltype(VEC), 100)
        A_blasfeo = MAT(A)
        a_blasfeo = VEC(a)
        y_blasfeo = VEC(y)

        @test A*a ≈ A_blasfeo*a_blasfeo
        @test isa(A_blasfeo*a_blasfeo, VEC)
        @test transpose(A)*y ≈ transpose(A_blasfeo)*y_blasfeo
        @test isa(transpose(A_blasfeo)*y_blasfeo, VEC)

        # test symmetric
        A = rand(eltype(VEC), 100, 100)
        a = rand(eltype(VEC), 100)
        y = rand(eltype(VEC), 100)
        A_blasfeo = MAT(A)
        a_blasfeo = VEC(a)
        y_blasfeo = VEC(y)
        A_L = Symmetric(A,:L)
        A_L_blasfeo = Symmetric(A_blasfeo,:L)
        A_U = Symmetric(A,:U)
        A_U_blasfeo = Symmetric(A_blasfeo,:U)

        @test A_L*a ≈ A_L_blasfeo*a_blasfeo
        @test isa(A_L_blasfeo*a_blasfeo, VEC)
        @test A_U*a ≈ A_U_blasfeo*a_blasfeo
        @test isa(A_U_blasfeo*a_blasfeo, VEC)
    end
end
