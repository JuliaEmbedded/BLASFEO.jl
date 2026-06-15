@testset "Level 1 BLAS Operations" begin
    @testset for VEC in (BlasfeoDvec, BlasfeoSvec)
        a = rand(eltype(VEC), 100)
        b = rand(eltype(VEC), 100)
        c = zeros(eltype(VEC), 100)
        a_blasfeo = VEC(a)
        b_blasfeo = VEC(b)
        c_blasfeo = VEC(c)

        α = rand()
        β = rand()

        # Test nondestructively
        @test dot(a,b) ≈ dot(a_blasfeo, b_blasfeo)
        @test (α.*a .+ b) ≈ axpy(α,a_blasfeo,b_blasfeo,c_blasfeo)
        @test ((α.*a) .+ (β.*b)) ≈ axpby(α,a_blasfeo,β,b_blasfeo,c_blasfeo)

        # Test destructively
        axpy!(α,a,b);axpy!(α,a_blasfeo,b_blasfeo)
        @test b ≈ b_blasfeo
        axpby!(α,a,β,b);axpby!(α,a_blasfeo,β,b_blasfeo)
        @test b ≈ b_blasfeo
    end
end
