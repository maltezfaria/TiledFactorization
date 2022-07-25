using TiledFactorization
using DataFlowTasks
using LinearAlgebra

import TiledFactorization as TF

# DataFlowTask environnement variables
capacity = 50
sch = DataFlowTasks.JuliaScheduler(capacity)
DataFlowTasks.setscheduler!(sch)

# Arguments
names = ["openblas", "dft", "mkl", "dagger", "forkjoin"]
sizes = 500:500:5000 |> collect

# Sizes
TF.benchmark(names, sizes)