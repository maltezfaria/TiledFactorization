"""
    module TiledFactorization

Tiled algorithms for factoring dense matrices.
"""
module TiledFactorization

const PROJECT_ROOT =  pkgdir(TiledFactorization)

using LinearAlgebra
using LoopVectorization
using RecursiveFactorization
using TriangularSolve

using DataFlowTasks

using DataFlowTasks: @spawn

const TILESIZE = Ref(256)

settilesize!(n::Int) = (TILESIZE[]=n)

include("utils.jl")
include("tiledmatrix.jl")
include("cholesky.jl")
include("lu.jl")

end
