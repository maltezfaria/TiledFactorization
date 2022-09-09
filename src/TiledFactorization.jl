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
using Serialization

using DataFlowTasks

using DataFlowTasks: R,W,RW

const TILESIZE = Ref(256)

settilesize!(n::Int) = (TILESIZE[]=n)

include("utils.jl")
include("tiledmatrix.jl")
include("cholesky.jl")
include("lu.jl")

function benchmark(methods,sizes;overwrite=false)
    hostname = gethostname()
    nt = Threads.nthreads()
    BLAS.set_num_threads(nt)
    # full path where serialized data will be stored
    fname = joinpath(PROJECT_ROOT,"benchmarks","data.jls")
    # if file already exists data will be appended to it
    if isfile(fname)
        @info "Appending benchmark data to existing file"
        data = deserialize(fname)
    else
        @info "Creating new data file for benchmarks"
        data = Dict()
    end
    # loop over sizes and methods and store the execution time
    for sz in sizes
        A = spd_matrix(sz)
        for f in methods
            key = (;hostname,threads=nt,method=repr(f),size=sz)
            if haskey(data,key)
                overwrite || (@show "skipping entry (key already found)"; continue)
            end
            # Benchmark
            b = @benchmark ($f)(B) setup=(B=copy($A)) evals=1
            t = median(b).time * 10^(-9)
            data[key] = t
            println("$key \t --> \t $t")
        end
    end
    serialize(fname,data)
    return data
end

end
