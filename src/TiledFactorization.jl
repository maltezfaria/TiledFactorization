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
using Octavian
using BenchmarkTools

using DataFlowTasks

using DataFlowTasks: R,W,RW

function schur_complement!(C,A,B,threads::Val{T}=Val(false)) where {T}
    if T
        Octavian.matmul!(C,A,B,-1,1)
    else
        Octavian.matmul_serial!(C,A,B,-1,1)
    end
end

function pick_tile_size()
    @info "Timing schur complement for different tile sizes..."
    dict = Dict{Int,Float64}()
    for n in [64,128,256,512,1024]
        C,A,B = rand(n,n),rand(n,n),rand(n,n)
        flops = 2*n^3 + 2*n^2
        t = @belapsed schur_complement!($C,$A,$B)
        gflops = flops / (t) / 1e9
        println("\tn=$n, \tgflops=$gflops")
        push!(dict,n=>gflops)
    end
    @info "Done."
    return dict
end

const TILESIZE = Ref(256)

settilesize!(n::Int) = (TILESIZE[]=n)

include("utils.jl")
include("tiledmatrix.jl")
include("cholesky.jl")
include("lu.jl")

end
