using TiledFactorization
import TiledFactorization as TF

# Parameters
names = ["openblas", "dft"]
nthreads = [1, 2, 4]
machine = "LEGION"
size = 3000

# Plot
f = TF.plot_scalability(names, nthreads, machine, size)
