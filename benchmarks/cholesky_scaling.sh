#!/bin/bash

# launch julia, activate the enrionment in the parent's folder, and instantiate
# it
julia --project -e '
    using Pkg
    Pkg.instantiate()
    Pkg.status()
'

# used to generate scalability for cholesky factorization. Should be run from
# the project root of TiledFactorization. Assumes that MKL, BenchmarkTools and
# Plots packages are available on your environment

for t in 1 2 4 8 16
do
echo $t threads
julia --project --threads=$t -e '
    using MKL
    using TiledFactorization
    include(joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks","utils.jl"))
    methods = [TiledFactorization.cholesky!,LinearAlgebra.cholesky!]
    sizes = [5000]
    benchmark(methods, sizes; overwrite=true)
'
done

# plot
julia --project -e '
    using TiledFactorization
    include(joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks","utils.jl"))
    methods = ["TiledFactorization.cholesky!","LinearAlgebra.cholesky!"]
    sz = 5000
    fig = plot_scalability_chol(methods,sz)
    savefig(fig,joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks","cholesky_scaling.png"))
'

