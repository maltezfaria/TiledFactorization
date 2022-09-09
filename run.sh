#!/bin/bash

# generate the data if needed
for t in 1 2 4 8 16
do
echo $t threads
julia --project --threads=$t -e '
    using TiledFactorization, LinearAlgebra;
    names = [TiledFactorization.cholesky_forkjoin!,LinearAlgebra.cholesky!]
    sizes = [5000]
    TiledFactorization.benchmark(names, sizes; overwrite=true)
'
done