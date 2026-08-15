@testset "Level 1 BLAS Operations" begin
    @testset for VEC in (BlasfeoDvec, BlasfeoSvec)
        a = rand(eltype(VEC), 100)
        b = rand(eltype(VEC), 100)
        c = zeros(eltype(VEC), 100)
        d = zeros(eltype(VEC), 99)
        a_blasfeo = VEC(a)
        b_blasfeo = VEC(b)
        c_blasfeo = VEC(c)
        d_blasfeo = VEC(d)

        α = rand(eltype(VEC))
        β = rand(eltype(VEC))

        # Test dimensions mismatch throws
        @test_throws DimensionMismatch a_blasfeo+d_blasfeo
        @test_throws DimensionMismatch a_blasfeo-d_blasfeo
        @test_throws DimensionMismatch a_blasfeo*d_blasfeo

        # Test basic operations
        @test +a ≈ +a_blasfeo
        @test isa(+a_blasfeo, VEC)
        @test -a ≈ -a_blasfeo
        @test isa(-a_blasfeo, VEC)
        @test a+b ≈ a_blasfeo + b_blasfeo

        # Test nondestructively
        @test dot(a,b) ≈ dot(a_blasfeo, b_blasfeo)
        @test (α.*a .+ b) ≈ axpy!(α,a_blasfeo,b_blasfeo,c_blasfeo)
        @test ((α.*a) .+ (β.*b)) ≈ axpby!(α,a_blasfeo,β,b_blasfeo,c_blasfeo)
        @test (a .* b) ≈ vecmul!(a_blasfeo, b_blasfeo, c_blasfeo)
        fill!(c_blasfeo, 1.0)
        @test (1 .+ a .* b) ≈ vecmulacc!(a_blasfeo, b_blasfeo, c_blasfeo)
        fill!(c_blasfeo, 0.0)
        @test sum(a .* b) ≈ vecmuldot!(a_blasfeo, b_blasfeo, c_blasfeo)
        @test (a .* b) ≈ c_blasfeo

        # Test destructively
        axpy!(α,a,b);axpy!(α,a_blasfeo,b_blasfeo)
        @test b ≈ b_blasfeo
        axpby!(α,a,β,b);axpby!(α,a_blasfeo,β,b_blasfeo)
        @test b ≈ b_blasfeo
    end
end
