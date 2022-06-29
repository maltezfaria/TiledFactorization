#=
    Tiled Cholesky factorization in pure Julia. The serial performance
    essentially comes from `LoopVectorization`. The parallelization is handled
    by `DataFlowTask`s.
=#

cholesky(A::Matrix,args...;kwargs...) = cholesky!(deepcopy(A),args...;kwargs...)

function cholesky!(A::Matrix,s=TILESIZE[],tturbo::Val{T}=Val(false);copy=Val(false)) where {T}
    _cholesky!(PseudoTiledMatrix(A,s),tturbo,copy)
end

# tiled cholesky factorization
function _cholesky!(A::PseudoTiledMatrix{S},tturbo::Val{T},copy::Val{C}) where {S,T,C}
    m,n = size(A) # number of blocks
    if C
        # copy A into contiguous blocks if copy is set to true
        A_  = Matrix{Matrix{S}}(undef,m,n)
        for i in 1:m
            for j in i:n
                A_[i,j] = Matrix{S}(undef,size(A[i,j]))
                @dspawn copy!(@W(A_[i,j]),@R(A[i,j])) label="copyin($i,$j)"
            end
        end
    else
        A_ = A
    end
    # main loop
    for i in 1:m
        Aii     = A_[i,i]
        @dspawn _chol!(@RW(Aii),UpperTriangular,tturbo) label="chol[$i,$i]"
        U = UpperTriangular(Aii)
        L = adjoint(U)
        for j in i+1:n
            Aij = A_[i,j]
            @dspawn begin
                @R  Aii # data in L
                @RW Aij
                TriangularSolve.ldiv!(L,Aij,tturbo)
            end label="ldiv(L[$i,$i],A[$i,$j])"
        end
        for j in i+1:m
            Aij = A_[i,j]
            for k in j:n
                # TODO: for k = j, only the upper part needs to be updated,
                # dividing the cost of that operation by two
                Ajk = A_[j,k]
                Aji = adjoint(Aij)
                Aik = A_[i,k]
                @dspawn begin
                    @RW Ajk
                    @R Aij # data in Aij
                    @R Aik
                    schur_complement!(Ajk,Aji,Aik,tturbo)
                end label="schur!(A[$j,$k],A[$j,$i],A[$i,$k])"
            end
        end
    end
    if C
        # copy back to standard format if needed
        for i in 1:m
            for j in i:n
                @dspawn copy!(@W(A[i,j]),@R(A_[i,j])) label="copyout($i,$j)"
            end
        end
    end
    # create the factorization object. Note that fetching this will force to
    # wait on all previous tasks
    res = @dspawn Cholesky(@R(A.data),'U',zero(LinearAlgebra.BlasInt)) label="Cholesky"
    return fetch(res)
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
