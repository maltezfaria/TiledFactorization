using TiledFactorization
import TiledFactorization as TF

# Parameters
names = ["openblas", "dft", "mkl"]
sizes = 500:500:3000 |> collect
nthreads = 4
machine = "LEGION"

TF.plot_sizes(names, sizes, machine, nthreads)
