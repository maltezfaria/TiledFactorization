#=
    Tiled LU factorization in pure Julia. The serial performance essentially
    comes from the `TriangularSolve` and `LoopVectorization`, and
    `RecursiveFactorization` packages. The parallelization is handled by
    `DataFlowTask`s.
=#

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
        @dspawn RecursiveFactorization.lu!(@RW(Aii),NoPivot(),tturbo)
        # @dspawn LinearAlgebra.lu!(Aii) (Aii,) (RW,)
        for j in i+1:n
            Aij = A[i,j]
            Aji = A[j,i]
            @dspawn begin
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
                @dspawn schur_complement!(@RW(Ajk),@R(Aji),@R(Aik),tturbo)
                # schur_complement!(Ajk,Aji,Aik,tturbo)
            end
        end
    end
    # wait for all computations before returning
    res = @dspawn LU(@R(A.data),LinearAlgebra.BlasInt[],zero(LinearAlgebra.BlasInt)) label="LU"
    return fetch(res)
end

# a fork-join approach for comparison with the data-flow parallelism
function _lu_forkjoin!(A::PseudoTiledMatrix,tturbo::Val{T}=Val(false)) where {T}
    m,n = size(A)
    for i in 1:m
        Aii = A[i,i]
        # FIXME: for simplicity, no pivot is allowed. Pivoting the diagonal
        # blocks requires permuting the corresponding row/columns before continuining
        RecursiveFactorization.lu!(Aii,NoPivot(),tturbo)
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
    # wait for all computations before returning
    DataFlowTasks.sync()
    return LU(A.data,LinearAlgebra.BlasInt[],zero(LinearAlgebra.BlasInt))
end
