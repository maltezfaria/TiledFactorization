using Test
using LinearAlgebra
using TiledFactorization

A = TiledFactorization.spd_matrix(100)
for tile_size in (10:15)
    for tturbo in (Val(false),Val(true))
        for copy in (Val(true),Val(false))
            F = TiledFactorization.cholesky(A,tile_size,tturbo;copy)
            @test F isa LinearAlgebra.Cholesky
            @test F.L*F.U ≈ A
        end
    end
end
