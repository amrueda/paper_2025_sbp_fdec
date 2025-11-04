using divfree_sbp
using Plots

# Test with the FEEC Sparse semidiscretization

md = 2
N = 12

tspan = (0.0, 10000.0)
cfl = 1.0
degrees = [2, 3]
analysis_timestep = 5e-2
analysis_points = Int(ceil(tspan[2] / analysis_timestep)) + 2
div_rk = zeros(analysis_points, length(degrees))
div_implicit = zeros(analysis_points, length(degrees))
energy_rk = zeros(analysis_points, length(degrees))
energy_implicit = zeros(analysis_points, length(degrees))

i = 1

for p in degrees
    nodes, W, Q, D, tL, tR = tensor_product_sbp.d1_fd_sbp(p, N)
    nodes = vec(nodes)

    semi = SemiDiscretizationFEECSparse(N - 1, W, D, nodes, md, -1, 1)

    u = initial_condition_projected(semi, tspan[1])
    div_rk[:, i], energy_rk[:, i] = timedisc!(u, semi, tspan, cfl,
                                              dt_analysis = analysis_timestep,
                                              save_visu = false, implicit = false)

    u = initial_condition_projected(semi, tspan[1])
    div_implicit[:, i], energy_implicit[:, i] = timedisc!(u, semi, tspan, cfl,
                                                          dt_analysis = analysis_timestep,
                                                          save_visu = false,
                                                          implicit = true)

    global i += 1
end

if !isdir("out")
    mkdir("out")
end

div1 = plot(tspan[1]:analysis_timestep:tspan[2], div_rk[1:(analysis_points - 1), 1],
            label = "p = 2, Runge-Kutta", xlabel = "time", ylabel = "divergence",
            linestyle = :dash)
plot!(div1, tspan[1]:analysis_timestep:tspan[2], div_implicit[1:(analysis_points - 1), 1],
      label = "p = 2, Crank-Nicolson", linestyle = :dot)
plot!(div1, tspan[1]:analysis_timestep:tspan[2], div_rk[1:(analysis_points - 1), 2],
      label = "p = 3, Runge-Kutta", linestyle = :dashdot)
plot!(div1, tspan[1]:analysis_timestep:tspan[2], div_implicit[1:(analysis_points - 1), 2],
      label = "p = 3, Crank-Nicolson", linestyle = :dashdotdot)
savefig(div1, joinpath("out", "divergence_test_coarse_T10000.pdf"))

en1 = plot(tspan[1]:analysis_timestep:tspan[2],
           energy_rk[1:(analysis_points - 1), 1] .- energy_rk[1, 1],
           label = "p = 2, Runge-Kutta", xlabel = "time", ylabel = "energy error",
           legend = :right, linestyle = :dash)
plot!(en1, tspan[1]:analysis_timestep:tspan[2],
      energy_implicit[1:(analysis_points - 1), 1] .- energy_implicit[1, 1],
      label = "p = 2, Crank-Nicolson", linestyle = :dot)
plot!(en1, tspan[1]:analysis_timestep:tspan[2],
      energy_rk[1:(analysis_points - 1), 2] .- energy_rk[1, 2],
      label = "p = 3, Runge-Kutta", linestyle = :dashdot)
plot!(en1, tspan[1]:analysis_timestep:tspan[2],
      energy_implicit[1:(analysis_points - 1), 2] .- energy_implicit[1, 2],
      label = "p = 3, Crank-Nicolson", linestyle = :dashdotdot)
savefig(en1, joinpath("out", "energy_test_coarse_T10000.pdf"))
