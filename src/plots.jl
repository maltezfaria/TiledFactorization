"""
    plot_scalability(names::Vector{String}, nthreads::Vector{Int}, machine::String, size::Int)  
Plot variation of performance depending on `nthreads` for all `names` benchmarks ran on `machine`
at `size`
"""
function plot_scalability(names::Vector{String}, nthreads::Vector{Int}, machine::String, size::Int)
    # Extract data
    dtfs = loaddataframes(names, machine, nthreads)

    # Number of set of parameters 
    nsetparam = length(names) * length(nthreads) 
    length(dtfs) != nsetparam && error("We didn't find all the data you were looking for")

    f = Figure()
    ax = Axis(
        f[1,1],
        title="Scalability on a $(size)x$(size) matrix\nOn $machine",
        xlabel="Nthreads", ylabel="GFlops",
        xticks = nthreads
    )

    # X axis
    x = nthreads

    # Vetor of data for each name
    ys = Dict{String, Vector{Float64}}()
    for name ∈ names
        push!(ys, name => Vector{Float64}(undef, length(nthreads)))
    end

    # Extract data
    fl = flops(size)
    for (param, dtf) ∈ dtfs
        i = findfirst(x->x==size, dtf[:, 1])

        # Error : didn't find data
        if i === nothing
            params = [param.name, param.machine, param.nthreads]
            params = string.(params) .* " "
            error("Didn't find [$size] entry in data file for [$(params...)]")
        end

        # Y data : Performance in GFlops
        t = fl/(dtf[i, 2] * 10^9)

        # Insert in Y data storage (ys) at the right index
        idx = findfirst(x->x==param.nthreads, nthreads)
        ys[param.name][idx] = t
    end

    # Plot
    for name ∈ names
        lines!(
            ax,
            x, ys[name],
            label = name,
            linewidth = 3
        )
        scatter!(
            ax,
            x, ys[name],
            color = :white,
            strokewidth = 2,
            strokecolor = :black
        )
    end

    # Legend
    axislegend(position = :lt)

    # Save image
    save(f, "scala", names, machine)

    
    f
end

"""
    plot_sizes(names::Vector{String}, sizes::Vector{Int}, machine, nthreads)  
Plot variation of performance depending on `size` for all `names` benchmarks ran on `machine`
at `nthreads`    
"""
function plot_sizes(names::Vector{String}, sizes::Vector{Int}, machine, nthreads)
    dtfs = loaddataframes(names, machine, [nthreads])

    f = Figure()
    ax = Axis(
        f[1,1],
        title="Performance with matrix size",
        xlabel="Matrix Size", ylabel="GFlops",
        xticks = sizes
    )
    x = sizes

    # Init ys : Vetor of data for each name
    ys = Dict{String, Vector{Float64}}()
    for name ∈ names
        push!(ys, name => Vector{Float64}())
    end

    # Extract data
    for (param, dtf) ∈ dtfs
        for i ∈ 1:length(dtf[:,1])
            size = dtf[i, 1]
            if size ∈ sizes
                t = flops(size) / (dtf[i, 2] * 10^9)
                push!(ys[param.name], t)
            end
        end
    end

    # Plot
    for name ∈ names
        lines!(
            ax,
            x, ys[name],
            label = name,
            linestyle = :solid,
            linewidth = 3
        )
        scatter!(
            ax,
            x, ys[name],
            color = :white,
            strokewidth = 2,
            strokecolor = :black
        )
    end

    # Legend
    axislegend(position = :lt)

    # Save image
    save(f, "sizes", names, machine)

    # Display
    f
end


function save(f::Figure, type::String, names::Vector{String}, machine::String)
    # File name -> type_n1_n2.svg
    filename = "$(type)_"
    for name ∈ names
        filename *= "$(name[1:2])_"
    end

    # Filepath
    subfolder = joinpath(PROJECT_ROOT, "fig")
    filename = joinpath(subfolder, filename)
    filename *= "_$(machine).svg"

    CairoMakie.save(filename, f)
end
