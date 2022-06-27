using Test
using LinearAlgebra
using TiledFactorization
using DataFlowTasks

DataFlowTasks.force_sequential(false)

sch = DataFlowTasks.JuliaScheduler()
DataFlowTasks.setscheduler!(sch)

function spd_matrix(m)
    # create an SPD matrix
    A = rand(m,m)
    A = (A + adjoint(A))/2
    A = A + m*I
end

A = spd_matrix(100)
tile_size = 50
F = TiledFactorization.cholesky(A,tile_size,Val(false))
@info norm(F.L*F.U - A,Inf)

for tile_size in (10:15)
    for tturbo in (Val(false),Val(true))
        F = TiledFactorization.cholesky(A,tile_size,tturbo)
        @test F isa LinearAlgebra.Cholesky
        @test F.L*F.U ≈ A
    end
end
