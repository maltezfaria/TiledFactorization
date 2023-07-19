#=
    Tiled LU factorization in pure Julia. The serial performance essentially
    comes from the `TriangularSolve` and `LoopVectorization`, and
    `RecursiveFactorization` packages. The parallelization is handled by
    `DataFlowTask`s.
=#

# compatibility with <=1.6
const NOPIVOT = VERSION >= v"1.7" ? NoPivot : Val{false}

lu(A::Matrix,args...) = lu!(deepcopy(A),args...)

function lu!(A::Matrix,s=TILESIZE[],tturbo::Val{T}=Val(false)) where {T}
    _lu!(PseudoTiledMatrix(A,s),tturbo)
end

function _lu!(A::PseudoTiledMatrix,tturbo::Val{T}=Val(false)) where {T}
    m,n = size(A)
    for i in 1:m
        Aii = A[i,i]
        # TODO: for simplicity, no pivot is allowed. Pivoting the diagonal
        # blocks requires permuting the corresponding row/columns before continuining
        @spawn RecursiveFactorization.lu!(@RW(Aii),NOPIVOT(),tturbo)
        # @spawn LinearAlgebra.lu!(Aii) (Aii,) (RW,)
        for j in i+1:n
            Aij = A[i,j]
            Aji = A[j,i]
            @spawn begin
                @R Aii
                @RW Aij Aji
                TriangularSolve.ldiv!(UnitLowerTriangular(Aii),Aij,tturbo)
                TriangularSolve.rdiv!(Aji,UpperTriangular(Aii),tturbo)
            end
            # TriangularSolve.ldiv!(UnitLowerTriangular(Aii),Aij,tturbo)
            # TriangularSolve.rdiv!(Aji,UpperTriangular(Aii),tturbo)
        end
        for j in i+1:m
            for k in i+1:n
                Ajk = A[j,k]
                Aji = A[j,i]
                Aik = A[i,k]
                @spawn schur_complement!(@RW(Ajk),@R(Aji),@R(Aik),tturbo)
                # schur_complement!(Ajk,Aji,Aik,tturbo)
            end
        end
    end
    # create the factorization object. Note that fetching this will force to
    # wait on all previous tasks
    res = @spawn LU(@R(A.data),LinearAlgebra.BlasInt[],zero(LinearAlgebra.BlasInt))
    return fetch(res)
end

# a fork-join approach for comparison with the data-flow parallelism
function _lu_forkjoin!(A::PseudoTiledMatrix,tturbo::Val{T}=Val(false)) where {T}
    m,n = size(A)
    for i in 1:m
        Aii = A[i,i]
        # FIXME: for simplicity, no pivot is allowed. Pivoting the diagonal
        # blocks requires permuting the corresponding row/columns before continuining
        RecursiveFactorization.lu!(Aii,NOPIVOT(),tturbo)
        Threads.@threads for j in i+1:n
            Aij = A[i,j]
            Aji = A[j,i]
            TriangularSolve.ldiv!(UnitLowerTriangular(Aii),Aij,tturbo)
            TriangularSolve.rdiv!(Aji,UpperTriangular(Aii))
        end
        @sync for j in i+1:m
            for k in i+1:n
                Ajk = A[j,k]
                Aji = A[j,i]
                Aik = A[i,k]
                Threads.@spawn schur_complement!(Ajk,Aji,Aik,tturbo)
            end
        end
    end
    return LU(A.data,LinearAlgebra.BlasInt[],zero(LinearAlgebra.BlasInt))
end
