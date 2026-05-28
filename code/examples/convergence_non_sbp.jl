using divfree_sbp
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Plots

#############################
# Run square convergence test
#############################   

# Test with SBP FD
tspan = (0.0, 1.0)
cfl = 1.0
n_iterations = 6
md = 1

error_Ex_L2 = zeros(Float64, n_iterations, 2)
error_Ey_L2 = zeros(Float64, n_iterations, 2)
error_Bz_L2 = zeros(Float64, n_iterations, 2)

it = 1
for strong in [false, true]
    N = 12
    for i = 1:n_iterations
        println("1D dof = ", md * (N))

        nodes, _, _, D, _, _ = tensor_product_sbp.d1_fd_sbp(3, N)
        nodes = vec(nodes)

        _, W, _, _, _, _ = tensor_product_sbp.d1_fd_sbp(2, N)

        semi = SemiDiscretizationFEECSparse(N - 1, Diagonal(W), sparse(D), nodes, md, -1, 1)
        u = initial_condition_projected(initial_condition_periodic, semi, tspan[1])

        timedisc!(
            u,
            semi,
            tspan,
            cfl,
            dt_analysis = 0.1,
            save_visu = false,
            strong = strong,
            constant = true,
        )

        u_nodal = convert2nodal(semi, u)
        u_exact = initial_condition_nodal(initial_condition_periodic, semi, tspan[2])

        error_Ex_L2[i, it] = l2_norm(semi, u_exact[1] .- u_nodal[1], true, false)
        error_Ey_L2[i, it] = l2_norm(semi, u_exact[2] .- u_nodal[2], false, true)
        error_Bz_L2[i, it] = l2_norm(semi, u_exact[3] .- u_nodal[3], true, true)

        N *= 2
    end
    global it += 1
end

eoc_Ex = zeros(Float64, n_iterations - 1, 2)
eoc_Ey = zeros(Float64, n_iterations - 1, 2)
eoc_Bz = zeros(Float64, n_iterations - 1, 2)

for k = 1:2
    for i = 1:(n_iterations-1)
        eoc_Ex[i, k] = log.(error_Ex_L2[i, k] ./ error_Ex_L2[i+1, k]) ./ log(2)
        eoc_Ey[i, k] = log.(error_Ey_L2[i, k] ./ error_Ey_L2[i+1, k]) ./ log(2)
        eoc_Bz[i, k] = log.(error_Bz_L2[i, k] ./ error_Bz_L2[i+1, k]) ./ log(2)
    end
end

if !isdir("out")
    mkdir("out")
end


writedlm(joinpath("out", "errors_non_sbp.csv"), [error_Ex_L2 error_Ey_L2 error_Bz_L2])
writedlm(
    joinpath("out", "EOC_non_sbp.csv"),
    [round.(eoc_Ex, digits = 2) round.(eoc_Ey, digits = 2) round.(eoc_Bz, digits = 2)],
)

