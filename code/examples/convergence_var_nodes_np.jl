using divfree_sbp
using LinearAlgebra
using DelimitedFiles

#############################
# Run square convergence test
#############################   

#= # ODE problem
using OrdinaryDiffEq
u = initial_condition_projected(semi, 0.0)
tspan = (0.0, 1.0)
prob = ODEProblem(compute_rhs!, u0, tspan, semi)
# solve(prob) # Does not work. It needs that function zero() can be applied on u0....
 =#

# Test with SBP FD
tspan = (0.0, 1.0)
cfl = 1.0
#n_iterations = 6
#n_number_nodes = 3
#degrees = [2, 3]

n_iterations = 6
n_number_nodes = 2
degrees = [3]

error_Ex_L2 = zeros(Float64, n_iterations, n_number_nodes, length(degrees))
error_Ey_L2 = zeros(Float64, n_iterations, n_number_nodes, length(degrees))
error_Bz_L2 = zeros(Float64, n_iterations, n_number_nodes, length(degrees))

it = 1

for p in degrees
    for N = (4*p):(4*p+n_number_nodes-1)
        println(" ")
        println("p = ", p)
        println(" ")
        for i = 1:n_iterations
            md = 2^(i - 1)
            println("1D dof = ", md * N)

            nodes, W, Q, D, tL, tR = tensor_product_sbp.d1_fd_sbp(p, N)
            nodes = vec(nodes)

            semi = SemiDiscretizationFEECSparse(
                N - 1,
                W,
                D,
                nodes,
                md,
                0,
                1,
                periodic = (false, false),
                essential = (false, true),
            )

            u = initial_condition_projected(initial_condition_non_periodic, semi, tspan[1])

            ener0 = compute_energy(semi, u)
            timedisc!(
                u,
                semi,
                tspan,
                cfl,
                dt_analysis = 0.1,
                save_visu = false,
                constant = true,
            )
            u_nodal = convert2nodal(semi, u)
            u_exact = initial_condition_nodal(initial_condition_non_periodic, semi, tspan[2])

            error_Ex_L2[i, N-4*p+1, it] =
                l2_norm(semi, u_exact[1] .- u_nodal[1], true, false)
            error_Ey_L2[i, N-4*p+1, it] =
                l2_norm(semi, u_exact[2] .- u_nodal[2], false, true)
            error_Bz_L2[i, N-4*p+1, it] =
                l2_norm(semi, u_exact[3] .- u_nodal[3], true, true)
        end
    end
    global it += 1
end

eoc_Ex = zeros(Float64, n_iterations - 1, n_number_nodes, length(degrees))
eoc_Ey = zeros(Float64, n_iterations - 1, n_number_nodes, length(degrees))
eoc_Bz = zeros(Float64, n_iterations - 1, n_number_nodes, length(degrees))
for p = 1:length(degrees)
    for i = 1:(n_iterations-1)
        eoc_Ex[i, :, p] = log.(error_Ex_L2[i, :, p] ./ error_Ex_L2[i+1, :, p]) ./ log(2)
        eoc_Ey[i, :, p] = log.(error_Ey_L2[i, :, p] ./ error_Ey_L2[i+1, :, p]) ./ log(2)
        eoc_Bz[i, :, p] = log.(error_Bz_L2[i, :, p] ./ error_Bz_L2[i+1, :, p]) ./ log(2)
    end
end

if !isdir("out")
    mkdir("out")
end

for j in eachindex(degrees)
    for i = 1:n_number_nodes
        node_amount = 4 * degrees[j] + i - 1
        writedlm(
            joinpath(
                "out",
                "EOC_constant_nodes_p" *
                string(degrees[j]) *
                "_" *
                string(node_amount) *
                "nodes_np.csv",
            ),
            [round.(eoc_Ex[:, i, j], digits = 2) round.(eoc_Ey[:, i, j], digits = 2) round.(
                eoc_Bz[:, i, j],
                digits = 2,
            )],
        )
        writedlm(
            joinpath(
                "out",
                "errors_constant_nodes_p" *
                string(degrees[j]) *
                "_" *
                string(node_amount) *
                "nodes_np.csv",
            ),
            [error_Ex_L2[:, i, j] error_Ey_L2[:, i, j] error_Bz_L2[:, i, j]],
        )
    end
end
