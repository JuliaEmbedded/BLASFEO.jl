@testset "Level 2 BLAS Operations" begin
    @testset for (MAT,VEC) in ((BlasfeoDmat,BlasfeoDvec), (BlasfeoSmat,BlasfeoSvec))
        A = rand(eltype(VEC), 100, 100)
        a = rand(eltype(VEC), 100)
        A_blasfeo = MAT(A)
        a_blasfeo = VEC(a)

        # Test nondestructively
        @test A*a ≈ A_blasfeo*a_blasfeo
        @test isa(A_blasfeo*a_blasfeo, VEC)
        @test transpose(A)*a ≈ transpose(A_blasfeo)*a_blasfeo
        @test isa(transpose(A_blasfeo)*a_blasfeo, VEC)
    end
end
