using Plots
using TiledFactorization
using BenchmarkTools
using Serialization
using LinearAlgebra

function benchmark(methods,sizes;overwrite=false)
    hostname = gethostname()
    nt = Threads.nthreads()
    BLAS.set_num_threads(nt)
    # full path where serialized data will be stored
    fname = joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks","data.jls")
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
        A = TiledFactorization.spd_matrix(sz)
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

function load_bench_data()
    fname = joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks","data.jls")
    if isfile(fname)
        @info "Loading benchmark data"
        data = deserialize(fname)
    else
        @info "Creating new benchmark data file"
        data = Dict()
    end
end

function clear_bench_data()
    fname = joinpath(TiledFactorization.PROJECT_ROOT,"benchmarks","data.jls")
    rm(fname)
end

function plot_scalability_chol(methods, sz::Int, nthreads=1:typemax(Int), host=gethostname())
    # extract data
    data = load_bench_data()
    threads = Dict(m=>[] for m in methods)
    gflops = Dict(m=>[] for m in methods)
    for (k,v) in data
        @show k
        n = k.size
        if n == sz && k.threads ∈ nthreads
            g = (1/3*n^3 + 1/2*n^2) / v / 1e9# flops per second
            m = k.method
            if m ∈ methods
                push!(threads[m],k.threads)
                push!(gflops[m],g)
            end
        end
    end
    fig = plot(;xlabel="threads",ylabel="gigaflops",legend=:topleft)
    for m in methods
        # sort by threads
        p = sortperm(threads[m])
        permute!(threads[m],p)
        permute!(gflops[m],p)
        # plot
        plot!(fig,threads[m],gflops[m];label=m,m=:o,xticks=threads[m])
    end
    return fig
end
