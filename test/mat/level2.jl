@testset "Level 2 BLAS Operations" begin
    @testset for (MAT,VEC) in ((BlasfeoDmat,BlasfeoDvec), (BlasfeoSmat,BlasfeoSvec))
        A = rand(eltype(VEC), 100, 100)
        B = rand(eltype(VEC), 95, 95)
        a = rand(eltype(VEC), 100)
        b = rand(eltype(VEC), 90)
        A_blasfeo = MAT(A)
        B_blasfeo = MAT(B)
        a_blasfeo = VEC(a)
        b_blasfeo = VEC(b)

        @test_throws DimensionMismatch A_blasfeo*b_blasfeo
        @test_throws DimensionMismatch B_blasfeo*a_blasfeo

        # Test nondestructively
        @test A*a ≈ A_blasfeo*a_blasfeo
        @test isa(A_blasfeo*a_blasfeo, VEC)
        @test transpose(A)*a ≈ transpose(A_blasfeo)*a_blasfeo
        @test isa(transpose(A_blasfeo)*a_blasfeo, VEC)
    end
end
