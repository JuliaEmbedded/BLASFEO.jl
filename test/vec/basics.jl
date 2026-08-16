@testset "Vector Basic Operations" begin
    @testset for VEC in (BlasfeoDvec, BlasfeoSvec)
        a = rand(eltype(VEC), 100)
        a_blasfeo = VEC(a)
        @test a == a_blasfeo
        @test a[10] == a_blasfeo[10]

        a[10] = 10.0; a_blasfeo[10] = 10.0
        a[20] = 20; a_blasfeo[20] = 20
        @test a == a_blasfeo

        b_blasfeo = similar(a_blasfeo)
        @test size(a_blasfeo) == size(b_blasfeo)
        # Note(@anton) Blasfeo _always_ clears memory.
        @test a_blasfeo != b_blasfeo

        c_blasfeo = copy(a_blasfeo)
        @test a_blasfeo == c_blasfeo

        c_blasfeo[15] = 15.0
        @test c_blasfeo[15] == 15.0
        @test a_blasfeo != c_blasfeo

        fill!(c_blasfeo, 100.0)
        @test all(c_blasfeo .== 100.0)
    end
end
