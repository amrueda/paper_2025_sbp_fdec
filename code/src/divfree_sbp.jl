module divfree_sbp

include("tensor_product_sbp.jl")
using SparseArrays
using LinearAlgebra
using Plots
using Measures
using Printf
using Krylov

export tensor_product_sbp
export SemiDiscretizationFEEC, SemiDiscretizationFEECSparse, SemiDiscretizationSEM
export compute_curl, compute_div
export compute_rhs!
export plot_variables
export initial_condition_projected, initial_condition_nodal
export ssprk33!, convert2nodal, compute_energy, timedisc!, l2_norm

"""
Mimetic semi-discretization
"""
struct SemiDiscretizationFEEC
    N::Int
    W::Matrix{Float64}
    D::Matrix{Float64}
    Winv::Matrix{Float64}
    delta::Matrix{Float64}
    G1::Matrix{Float64}
    V::Matrix{Float64}
    K::Matrix{Float64}
    nodes::Vector{Float64}
end

struct SemiDiscretizationFEECSparse
    N::Int
    n_elements_direction::Int
    W_local::Diagonal{Float64, Vector{Float64}}
    D_local::Matrix{Float64}
    W::Diagonal{Float64, Vector{Float64}}
    W_x::Diagonal{Float64, Vector{Float64}}
    W_y::Diagonal{Float64, Vector{Float64}}
    W_hat::Diagonal{Float64, Vector{Float64}}
    W_hat_inv::Diagonal{Float64, Vector{Float64}}
    V_x::SparseMatrixCSC{Float64, Int}
    V_y::SparseMatrixCSC{Float64, Int}
    V2_x::SparseMatrixCSC{Float64, Int}
    delta_x::SparseMatrixCSC{Float64, Int}
    delta_y::SparseMatrixCSC{Float64, Int}
    Wd_x::SparseMatrixCSC{Float64, Int}
    Wd_y::SparseMatrixCSC{Float64, Int}
    element_nodes::Vector{Float64}
end

"""
Compute operators
"""
function SemiDiscretizationFEEC(N, W, D, nodes)

    # Mass matrix inverse
    Winv = diagm(1 ./ diag(W))

    # Delta matrix
    delta = zeros(Float64, N, N + 1)
    for i in 1:N
        delta[i, i] = -1
        delta[i, i + 1] = 1
    end

    # Gluing operator
    G1 = zeros(Float64, N + 1, N + 1)
    G1[1, 1] = 0.5
    G1[1, N + 1] = 0.5
    G1[N + 1, 1] = 0.5
    G1[N + 1, N + 1] = 0.5
    for i in 2:N
        G1[i, i] = 1
    end

    # Edge basis functions
    V = zeros(Float64, N + 1, N)
    for j in 1:N
        for i in 1:(N + 1)
            for k in 1:j
                V[i, j] -= D[i, k]
            end
        end
    end

    K = transpose(D) * W * V
    return SemiDiscretizationFEEC(N, W, D, Winv, delta, G1, V, K, nodes)
end

"""
Compute global operators based on element-local operators.
We normalize the element node vector to the interval [0,dx], 
where dx is the size of an element in one direction.
"""
function SemiDiscretizationFEECSparse(N, W, D, nodes, n_elements_direction, a, b)
    dx = (b - a) / n_elements_direction
    esf = dx / (nodes[end] - nodes[1])
    element_nodes = esf .* (nodes .- nodes[1])
    n_elements = n_elements_direction^2
    I_Np1 = I(N + 1)
    I_N = I(N)
    I_m = I(n_elements)
    W_diag = esf * Diagonal(diag(W))
    D /= esf
    # element neighbour matrices
    P_x = spzeros(Float64, n_elements, n_elements)
    P_y = spzeros(Float64, n_elements, n_elements)

    for i in 1:(n_elements_direction - 1)
        for j in 1:n_elements_direction
            P_y[(i - 1) * n_elements_direction + j, i * n_elements_direction + j] = 1.0
        end
    end

    for j in 1:n_elements_direction
        P_y[(n_elements_direction - 1) * n_elements_direction + j, j] = 1.0
    end

    for i in 1:n_elements_direction
        for j in 1:(n_elements_direction - 1)
            P_x[(i - 1) * n_elements_direction + j, (i - 1) * n_elements_direction + j + 1] = 1.0
        end
        P_x[i * n_elements_direction, (i - 1) * n_elements_direction + 1] = 1.0
    end
    # intertwined mass matrix
    W_hat = Diagonal(view(W_diag, 1:N, 1:N))
    W_hat[1, 1] += W_diag[N + 1, N + 1]
    # Delta matrix
    delta_hat = spzeros(Float64, N, N)
    for i in 1:(N - 1)
        delta_hat[i, i] = -1
        delta_hat[i, i + 1] = 1
    end
    delta_hat[N, N] = -1.0
    delta_tilde = spzeros(Float64, N, N)
    delta_tilde[N, 1] = 1.0
    # Edge basis functions
    V = spzeros(Float64, N + 1, N)
    for j in 1:N
        for i in 1:(N + 1)
            for k in 1:j
                V[i, j] -= D[i, k]
            end
        end
    end

    V_x = kron(I_N, V)
    V_x = kron(I_m, V_x)

    V2_x = kron(I_Np1, V)
    V2_x = kron(I_m, V2_x)

    V_y = kron(V, I_N)
    V_y = kron(I_m, V_y)

    delta_hat_x = kron(I_N, delta_hat)
    delta_hat_x = kron(I_m, delta_hat_x)
    delta_tilde_x = kron(I_N, delta_tilde)
    delta_tilde_x = kron(P_x, delta_tilde_x)
    delta_x = delta_hat_x + delta_tilde_x

    delta_hat_y = kron(delta_hat, I_N)
    delta_hat_y = kron(I_m, delta_hat_y)
    delta_tilde_y = kron(delta_tilde, I_N)
    delta_tilde_y = kron(P_y, delta_tilde_y)
    delta_y = delta_hat_y + delta_tilde_y

    W_2D = kron(W_diag, W_diag)
    W_2D = kron(I_m, W_2D)

    W_hat_2D = kron(W_hat, W_hat)
    W_hat_2D = kron(I_m, W_hat_2D)
    W_hat_2D_inv = Diagonal(1 ./ diag(W_hat_2D))

    W_x = kron(W_diag, W_hat)
    W_x = kron(I_m, W_x)
    W_y = kron(W_hat, W_diag)
    W_y = kron(I_m, W_y)
    Wd_x = W_hat_2D_inv * transpose(delta_y) * transpose(V_y) * W_x * V_y
    Wd_y = W_hat_2D_inv * transpose(delta_x) * transpose(V_x) * W_y * V_x

    return SemiDiscretizationFEECSparse(N, n_elements_direction, W_diag, D, W_2D, W_x, W_y,
                                        W_hat_2D, W_hat_2D_inv, V_x, V_y, V2_x, delta_x,
                                        delta_y, Wd_x, Wd_y, element_nodes)
end

"""
Compute div of E = (Ex, Ey)
and return in nodal storage
"""
function compute_div(semi::SemiDiscretizationFEEC, Ex, Ey)
    return semi.V * (semi.delta * Ex + Ey * transpose(semi.delta)) * transpose(semi.V)
end

"""
Compute curl of B (scalar in 2D)
"""
function compute_curl(semi::SemiDiscretizationFEEC, B)
    return semi.G1 * B * transpose(semi.delta), -semi.delta * B * semi.G1
end

"""
Compute div of E = (Ex, Ey)
and return in nodal storage
"""
function compute_div(semi::SemiDiscretizationFEECSparse, Ex, Ey)
    return semi.V2_x * semi.V_y * (semi.delta_x * Ex + semi.delta_y * Ey)
end

"""
Compute curl of B (scalar in 2D)
"""
function compute_curl(semi::SemiDiscretizationFEECSparse, B)
    return semi.delta_y * B, -semi.delta_x * B
end

"""
Compute RHS
"""
function compute_rhs!(du, u, semi::SemiDiscretizationFEEC, t)
    Ex, Ey, Bz = u
    Ex_t, Ey_t = compute_curl(semi, Bz)
    Bz_t = semi.G1 * (semi.Winv * semi.K * Ey - Ex * transpose(semi.K) * semi.Winv) *
           semi.G1
    du[1] = Ex_t
    du[2] = Ey_t
    du[3] = Bz_t
    return du
end

function compute_rhs!(du, u, semi::SemiDiscretizationFEECSparse, t; strong = false)
    if strong
        compute_rhs_strong!(du, u, semi, t)
    else
        compute_rhs_weak!(du, u, semi, t)
    end
    return nothing
end

function compute_rhs_strong!(du, u, semi::SemiDiscretizationFEECSparse, t)
    Ex, Ey, Bz = convert2nodal(semi, u)
    du[1], du[2] = compute_curl(semi, Bz)
    Bz_t = Vector{Float64}(undef, length(Bz))
    m = semi.n_elements_direction
    N = semi.N
    w_0 = semi.W_local[1, 1]
    w_N = semi.W_local[N + 1, N + 1]
    w_inv = 1 / (w_0 + w_N)
    ind = 1
    for i in 1:m
        for j in 1:m
            offset = ((i - 1) * m + (j - 1)) * N * (N + 1)
            offset_neighbor_x = compute_left_neighbor(i, j, m) * N * (N + 1)
            offset_neighbor_y = compute_lower_neighbor(i, j, m) * N * (N + 1)

            for l in 1:N
                offset_local = offset + (l - 1) * (N + 1)
                offset_neighbor_local = offset_neighbor_x + (l - 1) * (N + 1)

                for k in 1:N
                    if k == 1
                        s = Ey[offset_neighbor_local + (N + 1)]
                        s -= Ey[offset_local + 1]
                        s -= w_N *
                             sum(semi.D_local[N + 1, r] * Ey[offset_neighbor_local + r]
                                 for r in 1:(N + 1))
                        s -= w_0 * sum(semi.D_local[1, r] * Ey[offset_local + r]
                                 for r in 1:(N + 1))
                        Bz_t[ind] = w_inv * s
                    else
                        Bz_t[ind] = -sum(semi.D_local[k, r] * Ey[offset_local + r]
                                         for r in 1:(N + 1))
                    end
                    if l == 1
                        s = -Ex[offset_neighbor_y + N^2 + k]
                        s += Ex[offset + k]
                        s += w_N * sum(semi.D_local[N + 1, r] *
                                 Ex[offset_neighbor_y + (r - 1) * N + k] for r in 1:(N + 1))
                        s += w_0 * sum(semi.D_local[1, r] * Ex[offset + (r - 1) * N + k]
                                 for r in 1:(N + 1))
                        Bz_t[ind] += w_inv * s
                    else
                        Bz_t[ind] += sum(semi.D_local[l, r] * Ex[offset + (r - 1) * N + k]
                                         for r in 1:(N + 1))
                    end
                    ind += 1
                end
            end
        end
    end
    du[3] = Bz_t
    return nothing
end

function compute_rhs_weak!(du, u, semi::SemiDiscretizationFEECSparse, t)
    Ex, Ey, Bz = u
    du[3] = semi.Wd_y * Ey - semi.Wd_x * Ex
    du[1], du[2] = compute_curl(semi, Bz)
    return nothing
end

function compute_left_neighbor(i, j, m)
    if j == 1
        return m * i - 1
    else
        return (i - 1) * m + j - 2
    end
end

function compute_lower_neighbor(i, j, m)
    if i == 1
        return m * (m - 1) + j - 1
    else
        return m * (i - 2) + j - 1
    end
end

"""
Initial condition for convergence test at the quadrature nodes. 
Modified from 
    * Ratnani & Sonnendrücker (2012). An Arbitrary High-Order Spline Finite Element Solver
    for the Time Domain Maxwell Equations.
"""
function initial_condition_nodal(semi, t)
    x = y = semi.nodes
    N = semi.N
    # Mag field
    Bz = zeros(Float64, N + 1, N + 1)
    for j in 1:(N + 1)
        for i in 1:(N + 1)
            Bz[i, j] = cos(pi * x[i] + pi) * cos(pi * y[j] + pi) *
                       cos((pi * 2 / sqrt(2)) * t)
        end
    end
    # Electric field
    Ex = zeros(Float64, N + 1, N + 1)
    for j in 1:(N + 1)
        for i in 1:(N + 1)
            Ex[i, j] = -0.5 * sqrt(2) * cos(pi * x[i] + pi) * sin(pi * y[j] + pi) *
                       sin((pi * 2 / sqrt(2)) * t)
        end
    end
    Ey = zeros(Float64, N + 1, N + 1)
    for j in 1:(N + 1)
        for i in 1:(N + 1)
            Ey[i, j] = 0.5 * sqrt(2) * sin(pi * x[i] + pi) * cos(pi * y[j] + pi) *
                       sin((pi * 2 / sqrt(2)) * t)
        end
    end

    return [Ex, Ey, Bz]
end

"""
Initial condition for convergence test at the quadrature nodes. 
Modified from 
    * Ratnani & Sonnendrücker (2012). An Arbitrary High-Order Spline Finite Element Solver
    for the Time Domain Maxwell Equations.
"""
function initial_condition_nodal(semi::SemiDiscretizationFEECSparse, t)
    md = semi.n_elements_direction
    dx = 2 / md
    m = md^2
    N = semi.N
    nodes = semi.element_nodes
    # Mag field
    Bz = zeros(Float64, m * N^2)
    for o in 1:md
        for k in 1:md
            for j in 1:N
                for i in 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Bz[idx] = cos(pi * x + pi) * cos(pi * y + pi) *
                              cos((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    # Electric field
    Ex = zeros(Float64, m * N * (N + 1))
    for o in 1:md
        for k in 1:md
            for j in 1:(N + 1)
                for i in 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx = (o - 1) * md * N * (N + 1) + (k - 1) * N * (N + 1) + (j - 1) * N +
                          i
                    Ex[idx] = -0.5 * sqrt(2) * cos(pi * x + pi) * sin(pi * y + pi) *
                              sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    Ey = zeros(Float64, m * N * (N + 1))
    for o in 1:md
        for k in 1:md
            for j in 1:N
                for i in 1:(N + 1)
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx = (o - 1) * md * N * (N + 1) + (k - 1) * N * (N + 1) +
                          (j - 1) * (N + 1) + i
                    Ey[idx] = 0.5 * sqrt(2) * sin(pi * x + pi) * cos(pi * y + pi) *
                              sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    return [Ex, Ey, Bz]
end

"""
Initial condition for convergence test (projected to polynomial space). 
Modified from 
    * Ratnani & Sonnendrücker (2012). An Arbitrary High-Order Spline Finite Element Solver
    for the Time Domain Maxwell Equations.
"""
function initial_condition_projected(semi::SemiDiscretizationFEEC, t)
    x = y = semi.nodes
    N = semi.N
    # Mag field
    Bz = zeros(Float64, N + 1, N + 1)
    for j in 1:(N + 1)
        for i in 1:(N + 1)
            Bz[i, j] = cos(pi * x[i] + pi) * cos(pi * y[j] + pi) *
                       cos((pi * 2 / sqrt(2)) * t)
        end
    end
    # Electric field
    Ex = zeros(Float64, N + 1, N)
    for j in 1:N
        for i in 1:(N + 1)
            Ex[i, j] = 0.5 * sqrt(2) *
                       (cos(pi * x[i] + pi) * (1 / pi) *
                        (cos(pi * y[j + 1] + pi) -
                         cos(pi * y[j] + pi))) * sin((pi * 2 / sqrt(2)) * t)
        end
    end
    Ey = zeros(Float64, N, N + 1)
    for j in 1:(N + 1)
        for i in 1:N
            Ey[i, j] = 0.5 * sqrt(2) *
                       (cos(pi * y[j] + pi) * (-1 / pi) *
                        (cos(pi * x[i + 1] + pi) -
                         cos(pi * x[i] + pi))) * sin((pi * 2 / sqrt(2)) * t)
        end
    end

    return [Ex, Ey, Bz]
end

"""
Initial condition for convergence test (projected to polynomial space). 
Modified from 
    * Ratnani & Sonnendrücker (2012). An Arbitrary High-Order Spline Finite Element Solver
    for the Time Domain Maxwell Equations.
"""
function initial_condition_projected(semi::SemiDiscretizationFEECSparse, t)
    N = semi.N
    md = semi.n_elements_direction
    m = md^2
    dx = 2 / md
    nodes = semi.element_nodes
    # Mag field
    Bz = zeros(Float64, m * N^2)
    for o in 1:md
        for k in 1:md
            for j in 1:N
                for i in 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Bz[idx] = cos(pi * x + pi) * cos(pi * y + pi) *
                              cos((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    # Electric field
    Ex = zeros(Float64, m * N^2)
    for o in 1:md
        for k in 1:md
            for j in 1:N
                for i in 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    yp1 = (o - 1) * dx + nodes[j + 1] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Ex[idx] = 0.5 * sqrt(2) *
                              (cos(pi * x + pi) * (1 / pi) *
                               (cos(pi * yp1 + pi) - cos(pi * y + pi))) *
                              sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    Ey = zeros(Float64, m * N^2)
    for o in 1:md
        for k in 1:md
            for j in 1:N
                for i in 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    xp1 = (k - 1) * dx + nodes[i + 1] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Ey[idx] = 0.5 * sqrt(2) *
                              (cos(pi * y + pi) * (-1 / pi) *
                               (cos(pi * xp1 + pi) - cos(pi * x + pi))) *
                              sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end

    return [Ex, Ey, Bz]
end

"""
Convert a solution array to nodal representation
"""
function convert2nodal(semi::SemiDiscretizationFEEC, u)
    Ex, Ey, Bz = u

    Ex_nodal = Ex * transpose(semi.V)
    Ey_nodal = semi.V * Ey
    Bz_nodal = Bz
    return Ex_nodal, Ey_nodal, Bz_nodal
end

"""
Convert a solution array to nodal representation
"""
function convert2nodal(semi::SemiDiscretizationFEECSparse, u)
    Ex, Ey, Bz = u

    Ex_nodal = semi.V_y * Ex
    Ey_nodal = semi.V_x * Ey
    Bz_nodal = Bz
    return Ex_nodal, Ey_nodal, Bz_nodal
end

"""
SEM semi-discretization
"""
struct SemiDiscretizationSEM
    N::Int
    W::Matrix{Float64}
    D::Matrix{Float64}
    Winv::Matrix{Float64}
    G1::Matrix{Float64}
    K::Matrix{Float64}
    nodes::Vector{Float64}
end

"""
Compute operators
"""
function SemiDiscretizationSEM(N, W, D, nodes)

    # Mass matrix inverse
    Winv = diagm(1 ./ diag(W))

    # Gluing operator
    G1 = zeros(Float64, N + 1, N + 1)
    G1[1, 1] = 0.5
    G1[1, N + 1] = 0.5
    G1[N + 1, 1] = 0.5
    G1[N + 1, N + 1] = 0.5
    for i in 2:N
        G1[i, i] = 1
    end

    K = transpose(D) * W
    return SemiDiscretizationSEM(N, W, D, Winv, G1, K, nodes)
end

"""
Convert a solution array to nodal representation
"""
function convert2nodal(semi::SemiDiscretizationSEM, u)
    return Ex, Ey, Bz = u
end

"""
Initial condition "projected" for the nodal SEM discretization
"""
function initial_condition_projected(semi::SemiDiscretizationSEM, t)
    return initial_condition_nodal(semi, t)
end

"""
Compute RHS
"""
function compute_rhs!(du, u, semi::SemiDiscretizationSEM, t)
    Ex, Ey, Bz = u
    Ex_t = semi.G1 * (-Bz * transpose(semi.K) * semi.Winv) * semi.G1
    Ey_t = semi.G1 * (semi.Winv * semi.K * Bz) * semi.G1
    Bz_t = semi.G1 * (semi.Winv * semi.K * Ey - Ex * transpose(semi.K) * semi.Winv) *
           semi.G1
    du[1] = Ex_t
    du[2] = Ey_t
    du[3] = Bz_t
    return du
end

"""
Compute div of E = (Ex, Ey) strongly (locally)
"""
function compute_div(semi::SemiDiscretizationSEM, Ex, Ey)
    return semi.D * Ex + Ey * transpose(semi.D)
end

function plot_variables(semi, u; prefix = "")
    Ex, Ey, Bz = u
    # Transform to nodal
    Ex_nodal, Ey_nodal, Bz = convert2nodal(semi, u)
    # Compute DivE
    divE = compute_div(semi, Ex, Ey)
    # We need to transpose the field before plotting them  for contourf 
    p1 = heatmap(vec(semi.nodes), vec(semi.nodes), transpose(Ex_nodal), levels = 20,
                 color = :turbo)
    p2 = heatmap(vec(semi.nodes), vec(semi.nodes), transpose(Ey_nodal), levels = 20,
                 color = :turbo)
    p3 = heatmap(vec(semi.nodes), vec(semi.nodes), transpose(Bz), levels = 20,
                 color = :turbo)
    p4 = heatmap(vec(semi.nodes), vec(semi.nodes), transpose(divE), levels = 20,
                 color = :turbo)
    return plot(p1, p2, p3, p4, layout = (2, 2),
                title = [prefix * "Ex" prefix * "Ey" prefix * "Bz" prefix * "divE"],
                right_margin = 10mm)
end

function plot_variables(semi::SemiDiscretizationFEECSparse, u; prefix = "")
    Ex, Ey, Bz = u
    # Transform to nodal
    md = semi.n_elements_direction
    dx = 2.0 / md
    N = semi.N
    points_dc = zeros(md * (N + 1))
    points_cont = zeros(md * N + 1)
    for i in 0:(md - 1)
        for j in 1:N
            points_cont[i * N + j] = i * dx + semi.element_nodes[j]
        end
        for j in 2:N
            points_dc[i * (N + 1) + j] = i * dx + semi.element_nodes[j]
        end
        points_dc[i * (N + 1) + 1] = i * dx + 1e-10
        points_dc[(i + 1) * (N + 1)] = (i + 1) * dx - 1e-10
    end
    points_dc .-= 1.0
    points_cont .-= 1.0
    points_cont[end] = 1.0
    points_dc[1] = -1.0
    points_dc[end] = 1.0

    Ex_nodal, Ey_nodal, Bz = convert2nodal(semi, u)
    # Compute DivE
    divE = compute_div(semi, Ex, Ey)

    # Convert the vectors of the field and divergence values to matrices usable for plotting
    Ex_matrix = convert_to_matrix(semi, Ex_nodal, true, false)
    Ey_matrix = convert_to_matrix(semi, Ey_nodal, false, true)
    Bz_matrix = convert_to_matrix(semi, Bz, true, true)
    div_matrix = convert_to_matrix(semi, divE, false, false)

    # We need to transpose the field before plotting them  for contourf 

    p1 = heatmap(points_cont, points_dc, Ex_matrix, levels = 20, color = :turbo)
    p2 = heatmap(points_dc, points_cont, Ey_matrix, levels = 20, color = :turbo)
    p3 = heatmap(points_cont, points_cont, Bz_matrix, levels = 20, color = :turbo)
    p4 = heatmap(points_dc, points_dc, div_matrix, levels = 20, color = :turbo)
    return plot(p1, p2, p3, p4, layout = (2, 2),
                title = [prefix * "Ex" prefix * "Ey" prefix * "Bz" prefix * "divE"],
                right_margin = 10mm)
end

function ssprk33!(du, u, semi, t, dt, u0; strong = false)
    u0 .= u

    # first stage
    compute_rhs!(du, u, semi, t, strong = strong)
    u .= u + dt * du

    # second stage
    compute_rhs!(du, u, semi, t, strong = strong)
    u .= u + dt * du
    u .= 0.75 * u0 + 0.25 * u

    # last stage
    compute_rhs!(du, u, semi, t, strong = strong)
    u .= u + dt * du
    u .= u0 / 3 + 2 * u / 3
end

function discrete_gradient!(u, semi, dt)
    Ex, Ey, Bz = u
    b = [Ex + 0.5 * dt * semi.delta_y * Bz; Ey - 0.5 * dt * semi.delta_x * Bz;
         -0.5 * dt * semi.Wd_x * Ex + 0.5 * dt * semi.Wd_y * Ey + Bz]
    n = size(semi.W_hat)
    A = sparse([I(n[1]) spzeros(n) -0.5*dt*semi.delta_y;
                spzeros(n) I(n[1]) 0.5*dt*semi.delta_x;
                0.5*dt*semi.Wd_x -0.5*dt*semi.Wd_y I(n[1])])
    x = reduce(vcat, u)
    x, _ = krylov_solve(:gmres, A, b, x, atol = 1e-12, rtol = 1e-12)
    u[1] = x[1:n[1]]
    u[2] = x[(n[1] + 1):(2 * n[1])]
    u[3] = x[(2 * n[1] + 1):(3 * n[1])]
end

function compute_energy(semi, u)
    Ex_nodal, Ey_nodal, Bz_nodal = convert2nodal(semi, u)

    ener = 0.5 * Ex_nodal .^ 2
    ener .+= 0.5 * Ey_nodal .^ 2
    ener .+= 0.5 * Bz_nodal .^ 2

    return sum(semi.W * ener * semi.W)
end

"""
Alternative energy computation for the sparse implementation.
"""

function compute_energy(semi::SemiDiscretizationFEECSparse, u)
    Ex, Ey, Bz = u

    e = transpose(Ex) * transpose(semi.V_y) * semi.W_x * semi.V_y * Ex
    e += transpose(Ey) * transpose(semi.V_x) * semi.W_y * semi.V_x * Ey
    e += transpose(Bz) * semi.W_hat * Bz

    return 0.5 * e
end

function compute_denergy_dt(semi, u, du)
    Ex, Ey, Bz = convert2nodal(semi, u)
    dEx, dEy, dBz = convert2nodal(semi, du)

    dener = Ex .* dEx
    dener += Ey .* dEy
    dener += Bz .* dBz
    return sum(semi.W * dener * semi.W)
end

function compute_denergy_dt(semi::SemiDiscretizationFEECSparse, u, du)
    Ex, Ey, Bz = u
    dEx, dEy, dBz = du

    de = transpose(dEx) * transpose(semi.V_y) * semi.W_x * semi.V_y * Ex
    de += transpose(dEy) * transpose(semi.V_x) * semi.W_y * semi.V_x * Ey
    de += transpose(dBz) * semi.W_hat * Bz
    return de
end

function determine_timestep(semi, cfl; implicit = false, constant = false)
    if constant
        return 1e-4
    else
        return cfl * minimum(diag(semi.W))
    end
end

function determine_timestep(semi::SemiDiscretizationFEECSparse, cfl; implicit = false,
                            constant = false)
    if constant
        return 1e-4
    else
        return cfl * minimum(diag(semi.W_local))
    end
end

function timedisc!(u, semi, tspan, cfl; dt_analysis = 0.1, save_visu = false,
                   implicit = false, strong = false, constant = false)
    t = tspan[1]
    du = deepcopy(u)
    r0 = deepcopy(u)
    finish = false
    analysis_points = Int(ceil(tspan[2] / dt_analysis)) + 2
    div = zeros(analysis_points)
    energy = zeros(analysis_points)
    counter = 2
    # Advance in time
    #################

    # First analysis
    compute_rhs!(du, u, semi, t, strong = strong)
    energy[1] = compute_energy(semi, u)
    div[1] = maximum(abs.(compute_div(semi, u[1], u[2])))
    println(@sprintf("%5s %12s %12s %12s %12s", "iter", "time", "total_energy",
                     "max|div(E)|", "d(Ener)/dt"))
    println(@sprintf("%5d %12.6e %12.6e %12.6e %12.6e", 0, t, energy[1], div[1],
                     compute_denergy_dt(semi, u, du)))
    t_analysis = 0
    if save_visu
        plot = plot_variables(semi, u)
        savefig(plot, @sprintf("output_%.4f.png", t_analysis))
    end
    for n in 1:100000000
        analyze = false
        dt = determine_timestep(semi, cfl, implicit = implicit, constant = constant)
        if t + dt > tspan[2]
            finish = true
            dt = tspan[2] - t
        elseif t + dt > t_analysis + dt_analysis
            analyze = true
            t_analysis += dt_analysis
            dt = t_analysis - t
        end
        if !implicit
            ssprk33!(du, u, semi, t, dt, r0, strong = strong)
        else
            discrete_gradient!(u, semi, dt)
        end
        t = t + dt

        if analyze || finish
            compute_rhs!(du, u, semi, t, strong = strong)
            energy[counter] = compute_energy(semi, u)
            div[counter] = maximum(abs.(compute_div(semi, u[1], u[2])))
            println(@sprintf("%5d %12.6e %12.6e %12.6e %12.6e", n, t, energy[counter],
                             div[counter], compute_denergy_dt(semi, u, du)))
            if save_visu
                plot = plot_variables(semi, u)
                savefig(plot, @sprintf("output_%.4f.png", t_analysis))
            end
            counter += 1
            finish && break
        end
    end

    return div, energy
end

"""
Takes discrete L2 norm of a field stored in the nodes!
"""
function l2_norm(semi, field)
    return sqrt(sum(semi.W * field .^ 2 * semi.W))
end

"""
Takes discrete L2 norm of a field stored in the nodes!
Since we have different mass matrices for continuous and discontinuous directions, 
we have to tell the function in which directions we are continuous or discontinuous.
"""
function l2_norm(semi::SemiDiscretizationFEECSparse, field, cont_x::Bool, cont_y::Bool)
    if !cont_x && !cont_y
        return sqrt(transpose(field) * semi.W * field)
    elseif cont_x && cont_y
        return sqrt(transpose(field) * semi.W_hat * field)
    elseif cont_x
        return sqrt(transpose(field) * semi.W_x * field)
    else
        return sqrt(transpose(field) * semi.W_y * field)
    end
end

""" 
Takes a vector of nodal field values and converts it to a matrix for plotting purposes.
To determine the proper dimensions of the matrix, we need to specify in which directions we are continuous or discontinuous.
"""
function convert_to_matrix(semi::SemiDiscretizationFEECSparse, field, cont_x::Bool,
                           cont_y::Bool)
    md = semi.n_elements_direction
    N = semi.N
    if cont_x
        n_x = md * N + 1
    else
        n_x = md * (N + 1)
    end

    if cont_y
        n_y = md * N + 1
    else
        n_y = md * (N + 1)
    end

    field_matrix = zeros(n_y, n_x)
    if cont_x && cont_y
        for i in 1:md
            for j in 1:md
                offset = (i - 1) * md * N^2 + (j - 1) * N^2
                range_x = ((i - 1) * N + 1):(i * N)
                range_y = ((j - 1) * N + 1):(j * N)
                field_matrix[range_x, range_y] = transpose(reshape(view(field,
                                                                        (offset + 1):(offset + N^2)),
                                                                   N,
                                                                   N))
            end
        end
        field_matrix[:, end] = view(field_matrix, :, 1)
        field_matrix[end, :] = view(field_matrix, 1, :)
    elseif cont_x
        for i in 1:md
            for j in 1:md
                offset = (i - 1) * md * N * (N + 1) + (j - 1) * N * (N + 1)
                range_x = ((i - 1) * (N + 1) + 1):(i * (N + 1))
                range_y = ((j - 1) * N + 1):(j * N)
                field_matrix[range_x, range_y] = transpose(reshape(view(field,
                                                                        (offset + 1):(offset + N * (N + 1))),
                                                                   N, N + 1))
            end
        end
        field_matrix[:, end] = view(field_matrix, :, 1)
    elseif cont_y
        for i in 1:md
            for j in 1:md
                offset = (i - 1) * md * N * (N + 1) + (j - 1) * N * (N + 1)
                range_x = ((i - 1) * N + 1):(i * N)
                range_y = ((j - 1) * (N + 1) + 1):(j * (N + 1))
                field_matrix[range_x, range_y] = transpose(reshape(view(field,
                                                                        (offset + 1):(offset + N * (N + 1))),
                                                                   N + 1, N))
            end
        end
        field_matrix[end, :] = view(field_matrix, 1, :)
    else
        for i in 1:md
            for j in 1:md
                offset = (i - 1) * md * (N + 1)^2 + (j - 1) * md * (N + 1)
                range_x = ((i - 1) * (N + 1) + 1):(i * (N + 1))
                range_y = ((j - 1) * (N + 1) + 1):(j * (N + 1))
                field_matrix[range_x, range_y] = transpose(reshape(view(field,
                                                                        (offset + 1):(offset + (N + 1)^2)),
                                                                   N + 1, N + 1))
            end
        end
    end
    return field_matrix
end

end
