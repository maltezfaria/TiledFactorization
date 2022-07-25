using Test
using LinearAlgebra
using TiledFactorization
import TiledFactorization as TF

# A = TiledFactorization.spd_matrix(100)
# for tile_size in (10:15)
#     for tturbo in (Val(false),Val(true))
#         F = TiledFactorization.cholesky(A,tile_size,tturbo)
#         @test F isa LinearAlgebra.Cholesky
#         @test F.L*F.U ≈ A
#     end
# end

@testset "Dagger" begin
    # create an SPD matrix
    m = 1280
    A = rand(m,m)
    A = (A + adjoint(A))/2
    A = A + m*I

    # Compute Dagger version
    F = TF.cholesky_dagger!(copy(A))
    er = norm(F.L*F.U-A,Inf)/max(norm(A),norm(F.L*F.U))
    @test er < 10^(-10)
    @info "Dagger error" er
end
