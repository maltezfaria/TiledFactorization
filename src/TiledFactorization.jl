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
