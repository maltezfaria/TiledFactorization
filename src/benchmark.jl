"""
    benchmark(names::Vector{String}, sizes::Vector{Int})  
Runs the benchmarks for all functions corresponding to `names`,
and for all `sizes`. Writes the result in a file in the data subfolder
wich would be of type :
`name_machine_nthreads.dat`
"""
function benchmark(names::Vector{String}, sizes::Vector{Int})
    # Parameters
    machine = gethostname()
    nthreads = Threads.nthreads()

    # BLAS parameters
    BLAS.set_num_threads(nthreads)

    for name ∈ names
        func = getfuncname(name)
        for size ∈ sizes
            # Clean memory
            GC.gc()
            
            # Matrix
            A = rand(size, size)
            A = (A + adjoint(A))/2
            A = A + size*I

            # Benchmark
            b = @benchmark $func(B) setup=(B=copy($A)) evals=1
            t = median(b).time * 10^(-9)

            # Filename
            file = getfilepath(name, machine, nthreads)

            # Write Result
            writeresult!(file, size, t)
        end
    end
end

"""
    writeresult!(file, size, result)  
Write the result of a measure in file at line starting with size.
If this line doesn't exists we create it.
If the file doesn't exists, we create it.
The line would be of type : `size result\n`
"""
function writeresult!(file, size, result)
    str = string(size)
    str *= " "
    str *= string(result)
    str *= '\n' 

    # Check if data subfolder exists, create it if not
    path = joinpath(PROJECT_ROOT, "data")
    !isdir(path) && mkdir(path)

    io = try
        open(file, "r+")
    catch
        open(file, "w")
    end
    
    buf = IOBuffer(UInt8[], read=true, write=true)

    sizeline_exists = false
    for line ∈ eachline(io)
        if first(split(line)) != string(size) || isempty(line)
            write(buf, line*'\n')
        else
            sizeline_exists = true
            write(buf, str)
        end
    end

    !sizeline_exists && write(buf, str)

    seekstart(buf)
    write(file, buf)

    close(io)
end

"""
    getfuncname(name::String)  
Give correspondant function for name
"""
function getfuncname(name::String)
    @match name begin
        "openblas" => LinearAlgebra.cholesky!
        "dft"      => cholesky!
        "dagger"   => cholesky_dagger!
        "forkjoin" => cholesky_forkjoin!
        "mkl"      => MKL.cholesky!
    end
end

# Utility function
function getfilepath(name, machine, nthreads)
    filename = name * '_'
    filename *= machine * '_'
    filename *= string(nthreads)
    filename *= ".csv"

    # Filepath
    filepath = joinpath("data", filename)
    filepath = joinpath(PROJECT_ROOT, filepath)
end


# ============================================================
#                       POST PROCESSING
# ============================================================

struct Parameters
    name::String
    machine::String
    nthreads::Int64
end
function loaddataframes(names::Vector{String}, machine::String, nthreads::Vector{Int})
    dtfs = Dict{Parameters, DataFrame}()

    datafolder = joinpath(PROJECT_ROOT, "data")
    for file ∈ readdir(datafolder)
        # Load DataFrame
        filepath = joinpath(datafolder, file)
        dtf = CSV.read(filepath, DataFrame, header=0)

        # Get Parameters
        p = getparameters(file)

        if p.name ∈ names && p.machine == machine && p.nthreads ∈ nthreads
            push!(dtfs, p => dtf)
        end
    end

    dtfs
end

# Gives the nb of flops for a matrix of size n
flops(n) = 1/3*n^3 + 1/2*n^2

"""
    getparameters(filename::String)  
Gives parameters of a benchmark according to filename
"""
function getparameters(filename::String)
    # Parameters alone but in strings
    str_param = split(filename, "_")
    str_param[end] = split(str_param[end], ".")[1]

    # Get parameters from string
    name = str_param[1]
    machine = str_param[2]
    nthreads = parse(Int64, str_param[3])
    
    Parameters(name, machine, nthreads)
end