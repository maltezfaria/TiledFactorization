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
using Dagger
using Requires
using BenchmarkTools
using FileIO
using Match
using DataFrames, CSV

using DataFlowTasks

using DataFlowTasks: R,W,RW

# FIXME: this probably belongs upstream at the DataFlowTasks package
import DataFlowTasks: memory_overlap
memory_overlap(L::UnitLowerTriangular,A) = memory_overlap(L.data,A)
memory_overlap(A,L::UnitLowerTriangular) = memory_overlap(L,A)
memory_overlap(U::UpperTriangular,A)     = memory_overlap(U.data,A)
memory_overlap(U,L::UpperTriangular)     = memory_overlap(L,U)
memory_overlap(U::Adjoint,A)             = memory_overlap(U.parent,A)
memory_overlap(U,L::Adjoint)             = memory_overlap(L,U)

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
include("benchmark.jl")

function __init__()
    # CairoMakie conditionnal loading
    @require CairoMakie="13f3f980-e62b-5c42-98c6-ff1f3baf88f0" include("plots.jl")
end

end
