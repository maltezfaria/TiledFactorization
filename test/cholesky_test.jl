using Test
using LinearAlgebra
using TiledFactorization

A = TiledFactorization.spd_matrix(100)
for tile_size in (10:15)
    for tturbo in (Val(false),Val(true))
        F = TiledFactorization.cholesky(A,tile_size,tturbo)
        @test F isa LinearAlgebra.Cholesky
        @test F.L*F.U ≈ A
    end
end
