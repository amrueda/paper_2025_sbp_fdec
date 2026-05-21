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
#n_iterations = 6
#degrees = [2, 3]

n_iterations = 6
degrees = [2, 3]

error_Ex_L2 = zeros(Float64, n_iterations, length(degrees))
error_Ey_L2 = zeros(Float64, n_iterations, length(degrees))
error_Bz_L2 = zeros(Float64, n_iterations, length(degrees))

for p in degrees
    println(" ")
    println("p = ", p)
    println(" ")
    md = 1
    N = 4 * p
    for i = 1:n_iterations
        println("1D dof = ", md * (N))

        nodes, W, Q, D, tL, tR = tensor_product_sbp.d1_fd_sbp(p, N)
        nodes = vec(nodes)

        semi = SemiDiscretizationFEECSparse(N - 1, Diagonal(W), sparse(D), nodes, md, -1, 1)

        u = initial_condition_projected(initial_condition_periodic, semi, tspan[1])

        timedisc!(
            u,
            semi,
            tspan,
            cfl,
            dt_analysis = 0.1,
            save_visu = false,
            strong = false,
            constant = true,
        )
        u_nodal = convert2nodal(semi, u)
        u_exact = initial_condition_nodal(initial_condition_periodic, semi, tspan[2])

        error_Ex_L2[i, p-1] = l2_norm(semi, u_exact[1] .- u_nodal[1], true, false)
        error_Ey_L2[i, p-1] = l2_norm(semi, u_exact[2] .- u_nodal[2], false, true)
        error_Bz_L2[i, p-1] = l2_norm(semi, u_exact[3] .- u_nodal[3], true, true)

        N *= 2
    end
end

eoc_Ex = zeros(Float64, n_iterations - 1, length(degrees))
eoc_Ey = zeros(Float64, n_iterations - 1, length(degrees))
eoc_Bz = zeros(Float64, n_iterations - 1, length(degrees))

for i = 1:(n_iterations-1)
    eoc_Ex[i, :] = log.(error_Ex_L2[i, :] ./ error_Ex_L2[i+1, :]) ./ log(2)
    eoc_Ey[i, :] = log.(error_Ey_L2[i, :] ./ error_Ey_L2[i+1, :]) ./ log(2)
    eoc_Bz[i, :] = log.(error_Bz_L2[i, :] ./ error_Bz_L2[i+1, :]) ./ log(2)
end

if !isdir("out")
    mkdir("out")
end

for j in eachindex(degrees)
    writedlm(
        joinpath("out", "errors_weak_form_p" * string(degrees[j]) * ".csv"),
        [error_Ex_L2[:, j] error_Ey_L2[:, j] error_Bz_L2[:, j]],
    )
    writedlm(
        joinpath("out", "EOC_weak_form_p" * string(degrees[j]) * ".csv"),
        [round.(eoc_Ex[:, j], digits = 2) round.(eoc_Ey[:, j], digits = 2) round.(
            eoc_Bz[:, j],
            digits = 2,
        )],
    )
end
