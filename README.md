# TiledFactorization

TiledFactorization is package that implements a cholesky tiled factorization using DataFlowTasks.jl package.

# Benchmarking

## Run

Benchmarks can be ran with :
```
using TiledFactorization
import TiledFactorization as TF

names = ["openblas", "dft"]
sizes = 500:500:3000 |> collect
TF.benchmark(names, sizes)
```

# Plots
*Figure to be added*

You can visualize benchmarks with :

```
names = ["openblas", "dft"]
machine = gethostname()
nthreads = [1, 2, 4, 8]
size = 5000

TF.plot_scalability(names, nthreads, machine, size)
```
where the number of threads varies but size stays fixed, or with :

```
names = ["openblas", "dft"]
machine = gethostname()
nthread = 4
size = 500:500:3000 |> collect

TF.plot_sizes(names, sizes, machine, nthread)
```

where the matrix size varies and the number of threads is fixed.

Current cholesky versions to be compared (and there aliases) :
* TiledFactorization (dft)
* OpenBLAS (openblas)
* Dagger (dagger)
* Forkjoin (forkjoin)

Current machine names :
* LEGION: personnal laptop with 4 cores *description*
* maury: *description*