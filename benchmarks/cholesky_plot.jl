PLOT    = true
SAVEFIG = true
USEMKL  = true

if USEMKL
    using MKL
end

# for speed disable debug and logging
using DataFlowTasks
DataFlowTasks.enable_debug(false)
DataFlowTasks.enable_log(false)
# DataFlowTasks.force_sequential(false; static =true)

using Test
using TiledFactorization
using LinearAlgebra
using BenchmarkTools


import TiledFactorization as TF

nt = Threads.nthreads()
BLAS.set_num_threads(nt)

@info "BLAS config" BLAS.get_config()
@info "BLAS threads  = $(BLAS.get_num_threads())"
@info "Julia threads = $nt"

capacity = 50
sch = DataFlowTasks.JuliaScheduler(capacity)
DataFlowTasks.setscheduler!(sch)

tilesize = 256
TF.TILESIZE[] = tilesize

nn    = 1000:500:5000 |> collect
t_blas = Float64[]
t_forkjoin = Float64[]
t_dataflow = Float64[]

for m in nn
    # create an SPD matrix
    A = rand(m,m)
    A = (A + adjoint(A))/2
    A = A + m*I

    # test blas
    @info "BLAS cholesky"
    b = @benchmark cholesky!(B) setup=(B=copy($A)) evals=1
    push!(t_blas,median(b).time)

    # test DataFlowTasks parallelism
    @info "Dataflow pseudotiled cholesky"
    b = @benchmark TiledFactorization.cholesky!(B) setup=(B=copy($A)) evals=1
    push!(t_dataflow,median(b).time)

    # test fork-join parallelism
    @info "Forkjoin cholesky"
    b = @benchmark TiledFactorization._cholesky_forkjoin!(B) setup=(B=TF.PseudoTiledMatrix(copy($A),TF.TILESIZE[]))
    push!(t_forkjoin,median(b).time)

    # compute the error
    DataFlowTasks.TASKCOUNTER[] = 1 # reset task counter to display how many tasks were created for
    F = TiledFactorization.cholesky(A)
    er_dft  = norm(F.L*F.U-A,Inf)/max(norm(A),norm(F.L*F.U))
    F       = cholesky!(copy(A))
    er_blas = norm(F.L*F.U-A,Inf)/max(norm(A),norm(F.L*F.U))

    # for m×m tiled matrix, there should be 1/6*(m^3-m) + 1/2*(m^2-m) + m tasks
    # created
    println("="^80)
    @info "m                 = $(m)"
    @info "Number of tasks   = $(DataFlowTasks.TASKCOUNTER[])"
    @info "er_dft            = $er_dft"
    @info "er_blas           = $er_blas"
    @info "t_blas            = $(t_blas[end])"
    @info "t_dataflow        = $(t_dataflow[end])"
    @info "t_forkjoin        = $(t_forkjoin[end])"
    println("="^80)
end

if PLOT
    using Plots
    flops = @. 1/3*nn^3 + 1/2*nn^2 # I think this is correct (up to O(n)), but double check
    plot(nn,flops./(t_blas),label=USEMKL ? "MKL" : "OpenBLAS",xlabel="n",ylabel="GFlops/second",m=:x,title="Cholesky factorization",legend=:bottomright)
    plot!(nn,flops./(t_forkjoin),label="TiledFactorization (forkjoin)",m=:x)
    plot!(nn,flops./(t_dataflow),label="TiledFactorization (dataflow)",m=:x)
    SAVEFIG && savefig(joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks/choleskyperf_capacity_$(capacity)_tilesize_$(tilesize).png"))
    # peakflops vary, not sure how to measure it. Maybe use cpuinfo?
    # peak = LinearAlgebra.peakflops()
    # plot!(nn,peak/1e9*ones(length(nn)))
    nothing
end
