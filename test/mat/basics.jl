@testset "Matrix Basic Operations" begin
    @testset for MAT in (BlasfeoDmat, BlasfeoSmat)
        A = rand(eltype(MAT), 100, 100)
        A_blasfeo = MAT(A)
        @test A == A_blasfeo
        @test A[10,20] == A_blasfeo[10,20]

        A[10,20] = 30.0; A_blasfeo[10,20] = 30.0
        A[20,30] = 50; A_blasfeo[20,30] = 50
        @test A == A_blasfeo

        B_blasfeo = similar(A_blasfeo)
        @test size(A_blasfeo) == size(B_blasfeo)
        # Note(@anton) Blasfeo _always_ clears memory.
        @test A_blasfeo != B_blasfeo

        C_blasfeo = copy(A_blasfeo)
        @test A_blasfeo == C_blasfeo

        C_blasfeo[15,15] = 30.0
        @test C_blasfeo[15,15] == 30.0
        @test A_blasfeo != C_blasfeo

        fill!(C_blasfeo, 100.0)
        @test all(C_blasfeo .== 100.0)
    end
end
