using divfree_sbp
using LinearAlgebra
using Plots

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
n_iterations = 6
n_number_nodes = 3
degrees = [2, 3]

error_Ex_L2 = zeros(Float64, n_iterations, n_number_nodes, 2)
error_Ey_L2 = zeros(Float64, n_iterations, n_number_nodes, 2)
error_Bz_L2 = zeros(Float64, n_iterations, n_number_nodes, 2)

for p in degrees
    for N in (4 * p):(4 * p + n_number_nodes - 1)
        println(" ")
        println("p = ", p)
        println(" ")
        for i in 1:n_iterations
            md = 2^(i - 1)
            println("1D dof = ", md * N)

            nodes, W, Q, D, tL, tR = tensor_product_sbp.d1_fd_sbp(p, N)
            nodes = vec(nodes)

            semi = SemiDiscretizationFEECSparse(N - 1, W, D, nodes, md, -1, 1)

            u = initial_condition_projected(semi, tspan[1])

            ener0 = compute_energy(semi, u)
            timedisc!(u, semi, tspan, cfl, dt_analysis = 0.1, save_visu = false,
                      constant = true)

            u_nodal = convert2nodal(semi, u)
            u_exact = initial_condition_nodal(semi, tspan[2])

            error_Ex_L2[i, N - 4 * p + 1, p - 1] = l2_norm(semi, u_exact[1] .- u_nodal[1],
                                                           true, false)
            error_Ey_L2[i, N - 4 * p + 1, p - 1] = l2_norm(semi, u_exact[2] .- u_nodal[2],
                                                           false, true)
            error_Bz_L2[i, N - 4 * p + 1, p - 1] = l2_norm(semi, u_exact[3] .- u_nodal[3],
                                                           true, true)
        end
    end
end

eoc_Ex = zeros(Float64, n_iterations - 1, n_number_nodes, 2)
eoc_Ey = zeros(Float64, n_iterations - 1, n_number_nodes, 2)
eoc_Bz = zeros(Float64, n_iterations - 1, n_number_nodes, 2)
for p in 1:2
    for i in 1:(n_iterations - 1)
        eoc_Ex[i, :, p] = log.(error_Ex_L2[i, :, p] ./ error_Ex_L2[i + 1, :, p]) ./ log(2)
        eoc_Ey[i, :, p] = log.(error_Ey_L2[i, :, p] ./ error_Ey_L2[i + 1, :, p]) ./ log(2)
        eoc_Bz[i, :, p] = log.(error_Bz_L2[i, :, p] ./ error_Bz_L2[i + 1, :, p]) ./ log(2)
    end
end
#plot(plot_Ex_L2, plot_Ey_L2, plot_Bz_L2, plot_energy, layout = (2, 2))
