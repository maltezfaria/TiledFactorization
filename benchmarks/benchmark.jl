using TiledFactorization
using DataFlowTasks
using LinearAlgebra

import TiledFactorization as TF

# DataFlowTask environnement variables
capacity = 50
sch = DataFlowTasks.JuliaScheduler(capacity)
DataFlowTasks.setscheduler!(sch)

# Arguments
names = ["openblas", "dft"]
sizes = [500, 1000]

# Sizes
TF.benchmark(names, sizes)