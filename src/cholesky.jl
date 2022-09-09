#=
    Tiled Cholesky factorization in pure Julia. The serial performance
    essentially comes from `LoopVectorization`. The parallelization is handled
    by `DataFlowTask`s.
=#

cholesky(A::Matrix,args...) = cholesky!(deepcopy(A),args...)

# Wrappers
function cholesky!(A::Matrix,s=TILESIZE[],tturbo::Val{T}=Val(false)) where {T}
    _cholesky!(PseudoTiledMatrix(A,s),tturbo)
end
function cholesky_forkjoin!(A::Matrix,s=TILESIZE[],tturbo::Val{T}=Val(false)) where {T}
    _cholesky_forkjoin!(PseudoTiledMatrix(A,s),tturbo)
end
function cholesky_dagger!(A::Matrix, s=TILESIZE[], tturbo::Val{T}=Val(false)) where {T}
    _cholesky_dagger!(PseudoTiledMatrix(A,s), tturbo)
end

# tiled cholesky factorization
function _cholesky!(A::PseudoTiledMatrix,tturbo::Val{T}=Val(false)) where {T}
    m,n = size(A) # number of blocks
    for i in 1:m
        Aii = A[i,i]
        @dspawn _chol!(@RW(Aii),UpperTriangular,tturbo) label="chol[$i,$i]"
        U = UpperTriangular(Aii)
        L = adjoint(U)
        for j in i+1:n
            Aij = A[i,j]
            @dspawn begin
                @R Aii
                @RW Aij
                TriangularSolve.ldiv!(L,Aij,tturbo)
            end label="ldiv(L[$i,$i],A[$i,$j])"
        end
        for j in i+1:m
            Aij = A[i,j]
            for k in j:n
                # TODO: for k = j, only the upper part needs to be updated,
                # dividing the cost of that operation by two
                Ajk = A[j,k]
                Aji = adjoint(Aij)
                Aik = A[i,k]
                @dspawn begin
                    @RW Ajk
                    @R  Aij Aik
                    schur_complement!(Ajk,Aji,Aik,tturbo)
                end label="schur!(A[$j,$k],A[$j,$i],A[$i,$k])"
            end
        end
    end
    DataFlowTasks.sync()
    return Cholesky(A.data,'U',zero(LinearAlgebra.BlasInt))
end

# a fork-join approach for comparison with the data-flow parallelism
function _cholesky_forkjoin!(A::PseudoTiledMatrix,tturbo::Val{T}=Val(false)) where {T}
    m,n = size(A) # number of blocks
    for i in 1:m
        _chol!(A[i,i],UpperTriangular,tturbo)
        Aii = A[i,i]
        U = UpperTriangular(Aii)
        L = adjoint(U)
        Threads.@threads for j in i+1:n
            Aij = A[i,j]
            TriangularSolve.ldiv!(L,Aij,tturbo)
        end
        # spawn m*(m+1)/2 tasks and sync them at the end
        @sync for j in i+1:m
            Aij = A[i,j]
            for k in j:n
                Ajk = A[j,k]
                Aji = adjoint(Aij)
                Aik = A[i,k]
                Threads.@spawn schur_complement!(Ajk,Aji,Aik,tturbo)
            end
        end
    end
    return Cholesky(A.data,'U',zero(LinearAlgebra.BlasInt))
end

# Implementation
function _cholesky_dagger!(A::PseudoTiledMatrix, tturbo::Val{T}=Val(false)) where {T}
    # Number of blocks
    m,n = size(A)

    # Thunks init
    thunks = Matrix{Dagger.EagerThunk}(undef, m, n)
    for i ∈ 1:m, j ∈ 1:n
        thunks[i, j] = Dagger.@spawn A[i, j] * 1.0
    end

    for i in 1:m
        # Diagonal block
        thunks[i, i] = Dagger.@spawn chol_task(thunks[i, i], tturbo)      
        
        L = adjoint(UpperTriangular(fetch(thunks[i, i])))

        # Forward substitutions
        for j in i+1:n
            thunks[i, j] = Dagger.@spawn TriangularSolve.ldiv!(L,thunks[i, j], tturbo)
        end

        # Partial submatrix update
        for j in i+1:m
            Aji = adjoint(fetch(thunks[i, j]))
            for k in j:n
                thunks[j, k] = Dagger.@spawn Octavian.matmul_serial!(thunks[j, k], Aji, thunks[i, k],-1,1)
            end
        end
    end

    for i ∈ 1:m, j ∈ i:n
        A[i, j] .= fetch(thunks[i, j])
    end

    return Cholesky(A.data,'U',zero(LinearAlgebra.BlasInt))
end

# Utility spawn thunk
function chol_task(Aii, tturbo)
    _chol!(Aii,UpperTriangular,tturbo)
    Aii
end


# Modified from the generic version from LinearAlgebra (MIT license).
function _chol!(A::AbstractMatrix{<:Real}, ::Type{UpperTriangular},tturbo::Val{T}=Val(false)) where {T}
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    @inbounds begin
        for k = 1:n
            Akk = A[k,k]
            for i = 1:k - 1
                Akk -= A[i,k]*A[i,k]
            end
            A[k,k] = Akk
            Akk, info = _chol!(Akk, UpperTriangular)
            if info != 0
                return UpperTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(Akk')
            if T
                @tturbo for j = k + 1:n
                    for i = 1:k - 1
                        A[k,j] -= A[i,k]*A[i,j]
                    end
                end
                @tturbo for j in k+1:n
                    A[k,j] = AkkInv*A[k,j]
                end
            else
                @turbo for j = k + 1:n
                    for i = 1:k - 1
                        A[k,j] -= A[i,k]*A[i,j]
                    end
                end
                @turbo for j in k+1:n
                    A[k,j] = AkkInv*A[k,j]
                end
            end
        end
    end
    return UpperTriangular(A), convert(Int32, 0)
end


## Numbers
function _chol!(x::Number, uplo)
    rx = real(x)
    rxr = sqrt(abs(rx))
    rval =  convert(promote_type(typeof(x), typeof(rxr)), rxr)
    rx == abs(x) ? (rval, convert(Int32, 0)) : (rval, convert(Int32, 1))
end
