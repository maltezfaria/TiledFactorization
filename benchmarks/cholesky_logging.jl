using Test
using DataFlowTasks
using LinearAlgebra
using BenchmarkTools
using Plots

import TiledFactorization as TF

capacity = 100
sch = DataFlowTasks.JuliaScheduler(capacity)
DataFlowTasks.setscheduler!(sch)

cp = Val(true)

m         = 5_000
tile_size = 256
# create an SPD matrix
A = TF.spd_matrix(m)
nt = Threads.nthreads()
# go once to compile
B = copy(A)
F = TF.cholesky!(B,tile_size;copy=cp)
@info "error: $(norm(F.L*F.U-A,Inf))"
GC.gc()
te = @elapsed TF.cholesky!(B,tile_size;copy=cp)
@info "elapsed: $te"

GC.gc()
# reset logger and run it again
logger = DataFlowTasks.getlogger()
DataFlowTasks.resetcounter!()
DataFlowTasks.resetlogger!()
F = TF.cholesky!(B,tile_size;copy=cp)
@info "Number of tasks = $(DataFlowTasks.TASKCOUNTER[])"
@info "Number of threads = $(Threads.nthreads())"
plot(logger;categories=["chol","ldiv","schur","copyin","copyout","Cholesky"])

# using GraphViz: Graph
# Graph(DataFlowTasks.logger_to_dot(logger))
