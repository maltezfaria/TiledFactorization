function spd_matrix(m)
    # create an m×m SPD matrix
    A = rand(m,m)
    A = (A + adjoint(A))/2
    A = A + m*I
end

function schur_complement!(C,A,B,threads::Val{T}=Val(false)) where {T}
    RecursiveFactorization.schur_complement!(C,A,B,threads)
    # if T
    #     Octavian.matmul!(C,A,B,-1,1)
    # else
    #     Octavian.matmul_serial!(C,A,B,-1,1)
    # end
end
