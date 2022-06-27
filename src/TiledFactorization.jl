"""
    module TiledFactorization

Tiled algorithms for factoring dense matrices.
"""
module TiledFactorization

using LinearAlgebra
using LoopVectorization
using RecursiveFactorization
using TriangularSolve
using Octavian

using DataFlowTasks
using DataFlowTasks: R,W,RW

import DataFlowTasks: memory_overlap

memory_overlap(L::UnitLowerTriangular,A) = memory_overlap(L.data,A)
memory_overlap(A,L::UnitLowerTriangular) = memory_overlap(L,A)
memory_overlap(U::UpperTriangular,A) = memory_overlap(U.data,A)
memory_overlap(U,L::UpperTriangular) = memory_overlap(L,U)
memory_overlap(U::Adjoint,A) = memory_overlap(U.parent,A)
memory_overlap(U,L::Adjoint) = memory_overlap(L,U)

function schur_complement!(C,A,B,threads::Val{T}) where {T}
    if T
        Octavian.matmul!(C,A,B,-1,1)
    else
        Octavian.matmul_serial!(C,A,B,-1,1)
    end
end

const TILESIZE = Ref(256)

include("tiledmatrix.jl")
include("cholesky.jl")
include("lu.jl")

end
