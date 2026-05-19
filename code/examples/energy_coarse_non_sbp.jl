using divfree_sbp
using Plots

# Test with the FEEC Sparse semidiscretization

md = 2
N = 12

tspan = (0.0, 1.0)
cfl = 1.0
degrees = [3]
analysis_timestep = 5e-2
analysis_points = Int(ceil(tspan[2] / analysis_timestep)) + 2
energy = zeros(analysis_points, 2)

for p in degrees
    nodes, _, _, D, _, _ = tensor_product_sbp.d1_fd_sbp(3, N)
    nodes = vec(nodes)

    _, W, _, _, _, _ = tensor_product_sbp.d1_fd_sbp(2, N)

    semi = SemiDiscretizationFEECSparse(N - 1, W, D, nodes, md, -1, 1)
    u = initial_condition_projected(initial_condition_periodic, semi, tspan[1])
    _, energy[:, 1] = timedisc!(
        u,
        semi,
        tspan,
        cfl,
        dt_analysis = analysis_timestep,
        strong = true,
        save_visu = false,
        implicit = false,
        constant = true,
    )

    u = initial_condition_projected(semi, tspan[1])
    _, energy[:, 2] = timedisc!(
        u,
        semi,
        tspan,
        cfl,
        dt_analysis = analysis_timestep,
        strong = false,
        save_visu = false,
        implicit = false,
        constant = true,
    )
end

if !isdir("out")
    mkdir("out")
end

en1 = plot(
    tspan[1]:analysis_timestep:tspan[2],
    energy[1:(analysis_points-1), 1] .- energy[1, 1],
    label = "Strong form",
    xlabel = "time",
    ylabel = "energy error",
    linestyle = :dash,
    legend = :bottomleft,
)
plot!(
    en1,
    tspan[1]:analysis_timestep:tspan[2],
    energy[1:(analysis_points-1), 2] .- energy[1, 2],
    label = "Weak form",
    linestyle = :dashdot,
)
savefig(en1, joinpath("out", "energy_test_non_sbp.pdf"))
