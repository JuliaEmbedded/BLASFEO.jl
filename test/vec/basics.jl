@testset "Vector Basic Operations" begin
    @testset for VEC in (BlasfeoDvec, BlasfeoSvec)
        A = rand(eltype(VEC), 100)
        A_blasfeo = VEC(A)
        @test A == A_blasfeo
        @test A[10] == A_blasfeo[10]

        A[10] = 10.0; A_blasfeo[10] = 10.0
        A[20] = 20; A_blasfeo[20] = 20
        @test A == A_blasfeo

        B_blasfeo = similar(A_blasfeo)
        @test size(A_blasfeo) == size(B_blasfeo)
        # Note(@anton) Blasfeo _always_ clears memory.
        @test A_blasfeo != B_blasfeo

        C_blasfeo = copy(A_blasfeo)
        @test A_blasfeo == C_blasfeo

        C_blasfeo[15] = 15.0
        @test C_blasfeo[15] == 15.0
        @test A_blasfeo != C_blasfeo

        fill!(C_blasfeo, 100.0)
        @test all(C_blasfeo .== 100.0)
    end
end
