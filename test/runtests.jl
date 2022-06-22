using SafeTestsets

@safetestset "Cholesky" begin include("cholesky_test.jl") end
@safetestset "LU" begin include("lu_test.jl") end
