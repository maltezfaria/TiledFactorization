function spd_matrix(m)
    # create an SPD matrix
    A = rand(m,m)
    A = (A + adjoint(A))/2
    A = A + m*I
end
