module divfree_sbp

include("tensor_product_sbp.jl")
using SparseArrays
using LinearAlgebra
using Plots
using DelimitedFiles
using Measures
using Printf
using Krylov
using QuadGK

import LinearAlgebra: size, mul!
import Base: eltype

export tensor_product_sbp
export SemiDiscretizationFEEC, SemiDiscretizationFEECSparse, SemiDiscretizationSEM
export compute_curl, compute_div
export compute_rhs!
export plot_variables
export initial_condition_projected,
    initial_condition_nodal, initial_condition_periodic, initial_condition_non_periodic
export ssprk33!, convert2nodal, compute_energy, timedisc!, l2_norm


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

"""
New variant of the mimetic semi-discretization
"""
struct SemiDiscretizationFEECSparse
    a::Float64
    b::Float64
    periodic::Tuple{Bool,Bool}
    essential::Tuple{Bool,Bool}
    increments_x::Tuple{Bool,Bool,Bool}
    increments_y::Tuple{Bool,Bool,Bool}
    N::Int
    n_elements_direction::Int
    n_elements::Int
    element_nodes::Vector{Float64}
    right_neighbors::Vector{Int}
    upper_neighbors::Vector{Int}
    left_neighbors::Vector{Int}
    lower_neighbors::Vector{Int}
    D::SparseMatrixCSC{Float64,Int}
    W::Diagonal{Float64,Vector{Float64}}
    W_hat::Diagonal{Float64,Vector{Float64}}
    W_hat_right::Diagonal{Float64,Vector{Float64}}
    W_hat_inv::Diagonal{Float64,Vector{Float64}}
    W_hat_inv_left::Diagonal{Float64,Vector{Float64}}
    W_hat_inv_right::Diagonal{Float64,Vector{Float64}}
    V::SparseMatrixCSC{Float64,Int}
    delta::SparseMatrixCSC{Float64,Int}
    delta_boundary::SparseMatrixCSC{Float64,Int}
    delta_right::SparseMatrixCSC{Float64,Int}
    delta_boundary_right::SparseMatrixCSC{Float64,Int}
    Wd::SparseMatrixCSC{Float64,Int}
    Wd_boundary::SparseMatrixCSC{Float64,Int}
    Wd_left::SparseMatrixCSC{Float64,Int}
    Wd_right::SparseMatrixCSC{Float64,Int}
    Wd_boundary_right::SparseMatrixCSC{Float64,Int}
end

struct LinearProblemFEECSparse
    semi::SemiDiscretizationFEECSparse
    dt::Float64
    ranges::Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}}
    lengths::Tuple{Int,Int,Int}
end

function LinearProblemFEECSparse(semi, dt)
    ne = semi.n_elements
    ned = semi.n_elements_direction
    periodic = semi.periodic
    N = semi.N
    length_Ex = ne * N^2 + !periodic[1] * ned * N
    length_Ey = ne * N^2 + !periodic[2] * ned * N
    length_Bz =
        ne * N^2 + (!periodic[1] + !periodic[2]) * ned * N + !periodic[1] * !periodic[2]
    range_Ex = 1:length_Ex
    range_Ey = (length_Ex+1):(length_Ex+length_Ey)
    range_Bz = (length_Ex+length_Ex+1):(length_Ex+length_Ey+length_Bz)
    return LinearProblemFEECSparse(
        semi,
        dt,
        (range_Ex, range_Ey, range_Bz),
        (length_Ex, length_Ey, length_Bz),
    )
end

function eltype(system::LinearProblemFEECSparse)
    return eltype(system.semi.W)
end

function size(system::LinearProblemFEECSparse)
    size = sum(system.lengths)
    return (size, size)
end

function mul!(x, system::LinearProblemFEECSparse, v)
    semi = system.semi
    length_Ex, length_Ey, length_Bz = system.lengths
    range_Ex, range_Ey, range_Bz = system.ranges

    x[range_Ex] .=
        view(v, range_Ex) .-
        0.5 .* system.dt .* product_kronecker_combined(
            semi.delta,
            semi.delta_boundary,
            view(v, range_Bz),
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.upper_neighbors,
            semi.lower_neighbors,
            semi.periodic,
            false,
            semi.increments_x[3],
            semi.increments_y[3],
            true,
            A_r = semi.delta_right,
            A_r_bound = semi.delta_boundary_right,
        )
    x[range_Ey] .=
        view(v, range_Ey) .+
        0.5 .* system.dt .* product_kronecker_combined(
            semi.delta,
            semi.delta_boundary,
            view(v, range_Bz),
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.right_neighbors,
            semi.left_neighbors,
            semi.periodic,
            false,
            semi.increments_x[3],
            semi.increments_y[3],
            false,
            A_r = semi.delta_right,
            A_r_bound = semi.delta_boundary_right,
        )
    x[range_Bz] .=
        0.5 .* system.dt .* product_kronecker_combined(
            semi.Wd,
            semi.Wd_boundary,
            view(v, range_Ex),
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.lower_neighbors,
            semi.upper_neighbors,
            semi.periodic,
            true,
            semi.increments_x[1],
            semi.increments_y[1],
            true,
            A_l = semi.Wd_left,
            A_r = semi.Wd_right,
            A_r_bound = semi.Wd_boundary_right,
        ) .-
        0.5 .* system.dt .* product_kronecker_combined(
            semi.Wd,
            semi.Wd_boundary,
            view(v, range_Ey),
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.left_neighbors,
            semi.right_neighbors,
            semi.periodic,
            true,
            semi.increments_x[2],
            semi.increments_y[2],
            false,
            A_l = semi.Wd_left,
            A_r = semi.Wd_right,
            A_r_bound = semi.Wd_boundary_right,
        ) .+ view(v, range_Bz)
    apply_ess_zero_boundary_cons!(
        view(x, range_Ex),
        view(x, range_Ey),
        view(x, range_Bz),
        semi,
    )
    return nothing
end

"""
Compute operators
"""
function SemiDiscretizationFEEC(N, W, D, nodes)

    # Mass matrix inverse
    Winv = diagm(1 ./ diag(W))

    # Delta matrix
    delta = zeros(Float64, N, N + 1)
    for i = 1:N
        delta[i, i] = -1
        delta[i, i+1] = 1
    end

    # Gluing operator
    G1 = zeros(Float64, N + 1, N + 1)
    G1[1, 1] = 0.5
    G1[1, N+1] = 0.5
    G1[N+1, 1] = 0.5
    G1[N+1, N+1] = 0.5
    for i = 2:N
        G1[i, i] = 1
    end

    # Edge basis functions
    V = zeros(Float64, N + 1, N)
    for j = 1:N
        for i = 1:(N+1)
            for k = 1:j
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
function SemiDiscretizationFEECSparse(
    N,
    W,
    D_,
    nodes,
    n_elements_direction,
    a,
    b;
    periodic = (true, true),
    essential = (true, true),
)
    dx = (b - a) / n_elements_direction
    esf = dx / (nodes[end] - nodes[1])
    element_nodes = esf .* (nodes .- nodes[1])
    n_elements = n_elements_direction^2
    W_diag = esf * Diagonal(diag(W))
    D = sparse(D_)
    D /= esf

    increment_x = (!periodic[1], false, !periodic[1])
    increment_y = (false, !periodic[2], !periodic[2])

    right_neighbors = fill(-1, n_elements)
    upper_neighbors = fill(-1, n_elements)
    left_neighbors = fill(-1, n_elements)
    lower_neighbors = fill(-1, n_elements)

    for i = 1:n_elements_direction-1
        for j = 1:n_elements_direction
            upper_neighbors[(i-1)*n_elements_direction+j] = i * n_elements_direction + j
        end
    end

    if periodic[2]
        for j = 1:n_elements_direction
            upper_neighbors[(n_elements_direction-1)*n_elements_direction+j] = j
        end
    end

    for i = 1:n_elements_direction
        for j = 1:n_elements_direction-1
            right_neighbors[(i-1)*n_elements_direction+j] =
                (i - 1) * n_elements_direction + j + 1
        end
        if periodic[1]
            right_neighbors[i*n_elements_direction] = (i - 1) * n_elements_direction + 1
        end
    end

    for i = 1:n_elements
        if right_neighbors[i] != -1
            left_neighbors[right_neighbors[i]] = i
        end
        if upper_neighbors[i] != -1
            lower_neighbors[upper_neighbors[i]] = i
        end
    end

    # intertwined mass matrix
    W_hat = Diagonal(view(W_diag, 1:N, 1:N))
    W_hat[1, 1] += W_diag[N+1, N+1]
    W_hat_inv = inv(W_hat)

    if periodic[1] && periodic[2]
        W_hat_left_inv = Diagonal{Float64}([])
        W_hat_right = Diagonal{Float64}([])
        W_hat_right_inv = Diagonal{Float64}([])
    elseif n_elements_direction == 1
        W_hat_left_inv = Diagonal([])
        W_hat_right = Diagonal(W_diag)
        W_hat_right_inv = inv(W_diag)
    else
        W_hat_left_inv = Diagonal(inv(view(W_diag, 1:N, 1:N)))
        W_hat_right = Diagonal(copy(W_diag))
        W_hat_right[1, 1] += W_diag[N+1, N+1]
        W_hat_right_inv = inv(W_hat_right)
    end

    # Delta matrix

    delta = spzeros(Float64, N, N)
    delta[N, N] = -1.0
    for i = 1:N-1
        delta[i, i] = -1
        delta[i, i+1] = 1
    end
    delta_boundary = spzeros(Float64, N, N)
    delta_boundary[N, 1] = 1.0

    if periodic[1] && periodic[2]
        delta_right = spzeros(Float64, 0, 0)
        delta_boundary_right = spzeros(Float64, 0, 0)
    elseif n_elements_direction == 1
        delta_right = spzeros(Float64, N, N + 1)
        for i = 1:N
            delta_right[i, i] = -1
            delta_right[i, i+1] = 1
        end
        delta_boundary_right = spzeros(Float64, N, N + 1)
        delta_boundary_right[N, 1] = 1.0
    else
        delta_right = spzeros(Float64, N, N + 1)
        for i = 1:N
            delta_right[i, i] = -1
            delta_right[i, i+1] = 1
        end
        delta_boundary_right = spzeros(Float64, N, N + 1)
        delta_boundary_right[N, 1] = 1.0
    end

    # Vandermonde for histopolation basis functions
    V = spzeros(Float64, N + 1, N)
    for j = 1:N
        for i = 1:(N+1)
            for k = 1:j
                V[i, j] -= D[i, k]
            end
        end
    end
    droptol!(V, 1e-14)

    Wd = W_hat_inv * transpose(delta) * transpose(V) * W_diag * V
    Wd_boundary = W_hat_inv * transpose(delta_boundary) * transpose(V) * W_diag * V

    if periodic[1] && periodic[2]
        Wd_left = spzeros(Float64, 0, 0)
        Wd_right = spzeros(Float64, 0, 0)
        Wd_boundary_right = spzeros(Float64, 0, 0)
    elseif n_elements_direction == 1
        Wd_left = spzeros(Float64, 0, 0)
        Wd_boundary_right = spzeros(Float64, 0, 0)
        Wd_right = W_hat_right_inv * transpose(delta_right) * transpose(V) * W_diag * V
    else
        Wd_left = W_hat_left_inv * transpose(delta) * transpose(V) * W_diag * V
        Wd_right = W_hat_right_inv * transpose(delta_right) * transpose(V) * W_diag * V
        Wd_boundary_right =
            W_hat_right_inv * transpose(delta_boundary_right) * transpose(V) * W_diag * V
    end

    return SemiDiscretizationFEECSparse(
        Float64(a),
        Float64(b),
        periodic,
        essential,
        increment_x,
        increment_y,
        N,
        n_elements_direction,
        n_elements,
        element_nodes,
        right_neighbors,
        upper_neighbors,
        left_neighbors,
        lower_neighbors,
        D,
        W_diag,
        W_hat,
        W_hat_right,
        W_hat_inv,
        W_hat_left_inv,
        W_hat_right_inv,
        V,
        delta,
        delta_boundary,
        delta_right,
        delta_boundary_right,
        Wd,
        Wd_boundary,
        Wd_left,
        Wd_right,
        Wd_boundary_right,
    )
end

@inline function product_kronecker_general(
    A,
    B,
    v,
    block_dims,
    n_dir,
    periodic,
    increment_x = false,
    increment_y = false;
    A_l = A,
    A_r = A,
    B_l = B,
    B_r = B,
)
    if periodic[1] && periodic[2]
        border_matrix_A_rows = size(A, 1)
        border_matrix_B_rows = size(B, 1)
    else
        border_matrix_A_rows = size(A_r, 1)
        border_matrix_B_rows = size(B_r, 1)
    end
    block_dims_info, block_size_info, new_block_size_info = block_information(
        block_dims,
        increment_x,
        increment_y,
        true,
        true,
        size(A, 1),
        border_matrix_A_rows,
        size(B, 1),
        border_matrix_B_rows,
    )
    A_cur = A
    B_cur = B
    bd_cur = block_dims
    obs_cur = block_size_info[1]
    nbs_cur = new_block_size_info[1]
    old = 1
    new = 1
    b = Vector{Float64}(undef, vector_length(n_dir, new_block_size_info))
    if periodic[1] && periodic[2]
        for i = 1:n_dir
            for j = 1:n_dir
                b[new:new+nbs_cur-1] =
                    vec(B * reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A))
                old += obs_cur
                new += nbs_cur
            end
        end
    elseif !periodic[1] && !periodic[2]
        for i = 1:n_dir
            bd_cur = block_dims_info[1]
            obs_cur = block_size_info[1]
            nbs_cur = new_block_size_info[1]
            if i == n_dir
                A_cur = A_r
                bd_cur = block_dims_info[3]
                obs_cur = block_size_info[3]
                nbs_cur = new_block_size_info[3]
            elseif i == 1
                A_cur = A_l
            else
                A_cur = A
            end
            for j = 1:n_dir
                if j == n_dir
                    B_cur = B_r
                    if i == n_dir
                        bd_cur = block_dims_info[4]
                        obs_cur = block_size_info[4]
                        nbs_cur = new_block_size_info[4]
                    else
                        bd_cur = block_dims_info[2]
                        obs_cur = block_size_info[2]
                        nbs_cur = new_block_size_info[2]
                    end
                elseif j == 1
                    B_cur = B_l
                else
                    B_cur = B
                end
                b[new:new+nbs_cur-1] = vec(
                    B_cur *
                    reshape(view(v, (old:old+obs_cur-1)), bd_cur) *
                    transpose(A_cur),
                )
                old += obs_cur
                new += nbs_cur
            end
        end
    elseif periodic[1]
        for i = 1:n_dir
            if i == n_dir
                A_cur = A_r
                bd_cur = block_dims_info[3]
                obs_cur = block_size_info[3]
                nbs_cur = new_block_size_info[3]
            elseif i == 1
                A_cur = A_l
            else
                A_cur = A
            end
            for j = 1:n_dir
                b[new:new+nbs_cur-1] = vec(
                    B * reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A_cur),
                )
                old += obs_cur
                new += nbs_cur
            end
        end
    else
        for i = 1:n_dir
            bd_cur = block_dims_info[1]
            obs_cur = block_size_info[1]
            nbs_cur = new_block_size_info[1]
            for j = 1:n_dir
                if j == n_dir
                    B_cur = B_r
                    bd_cur = block_dims_info[2]
                    obs_cur = block_size_info[2]
                    nbs_cur = new_block_size_info[2]
                elseif j == 1
                    B_cur = B_l
                else
                    B_cur = B
                end
                b[new:new+nbs_cur-1] = vec(
                    B_cur * reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A),
                )
                old += obs_cur
                new += nbs_cur
            end
        end
    end
    return b
end

"""
Emulates the product of a vector with a block-diagonal matrix, 
where the blocks are Kronecker product matrices.
We assume here that one of the Kronecker 
product arguments is the identity matrix.
"""

@inline function product_kronecker(
    A,
    v,
    block_dims,
    n_dir,
    periodic,
    increment_x = false,
    increment_y = false,
    left = true;
    A_l = A,
    A_r = A,
)
    if left
        return product_kronecker_left(
            A,
            v,
            block_dims,
            n_dir,
            periodic,
            increment_x,
            increment_y,
            A_l = A_l,
            A_r = A_r,
        )
    else
        return product_kronecker_right(
            A,
            v,
            block_dims,
            n_dir,
            periodic,
            increment_x,
            increment_y,
            A_l = A_l,
            A_r = A_r,
        )
    end
end

@inline function product_kronecker_left(
    A,
    v,
    block_dims,
    n_dir,
    periodic,
    increment_x = false,
    increment_y = false;
    A_l = A,
    A_r = A,
)
    if periodic[1] && periodic[2]
        border_matrix_rows = size(A, 1)
    else
        border_matrix_rows = size(A_r, 1)
    end
    block_dims_info, block_size_info, new_block_size_info = block_information(
        block_dims,
        increment_x,
        increment_y,
        true,
        false,
        size(A, 1),
        border_matrix_rows,
    )
    A_cur = A
    bd_cur = block_dims
    obs_cur = block_size_info[1]
    nbs_cur = new_block_size_info[1]
    old = 1
    new = 1
    b = Vector{Float64}(undef, vector_length(n_dir, new_block_size_info))
    if periodic[1] && periodic[2]
        for i = 1:n_dir
            for j = 1:n_dir
                b[new:new+nbs_cur-1] =
                    vec(reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A))
                old += obs_cur
                new += nbs_cur
            end
        end
    elseif !periodic[1] && !periodic[2]
        for i = 1:n_dir
            bd_cur = block_dims
            obs_cur = block_size_info[1]
            nbs_cur = new_block_size_info[1]
            if i == n_dir
                A_cur = A_r
                bd_cur = block_dims_info[3]
                obs_cur = block_size_info[3]
                nbs_cur = new_block_size_info[3]
            elseif i == 1
                A_cur = A_l
            else
                A_cur = A
            end
            for j = 1:n_dir
                if j == n_dir
                    if i == n_dir
                        bd_cur = block_dims_info[4]
                        obs_cur = block_size_info[4]
                        nbs_cur = new_block_size_info[4]
                    else
                        bd_cur = block_dims_info[2]
                        obs_cur = block_size_info[2]
                        nbs_cur = new_block_size_info[2]
                    end
                end
                b[new:new+nbs_cur-1] =
                    vec(reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A_cur))
                old += obs_cur
                new += nbs_cur
            end
        end
    elseif periodic[1]
        for i = 1:n_dir
            if i == n_dir
                A_cur = A_r
                bd_cur = block_dims_info[3]
                obs_cur = block_size_info[3]
                nbs_cur = new_block_size_info[3]
            elseif i == 1
                A_cur = A_l
            else
                A_cur = A
            end
            for j = 1:n_dir
                b[new:new+nbs_cur-1] =
                    vec(reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A_cur))
                old += obs_cur
                new += nbs_cur
            end
        end
    else
        for i = 1:n_dir
            bd_cur = block_dims
            obs_cur = block_size_info[1]
            nbs_cur = new_block_size_info[1]
            for j = 1:n_dir
                if j == n_dir
                    bd_cur = block_dims_info[2]
                    obs_cur = block_size_info[2]
                    nbs_cur = new_block_size_info[2]
                end
                b[new:new+nbs_cur-1] =
                    vec(reshape(view(v, (old:old+obs_cur-1)), bd_cur) * transpose(A))
                old += obs_cur
                new += nbs_cur
            end
        end
    end
    return b
end

@inline function product_kronecker_right(
    A,
    v,
    block_dims,
    n_dir,
    periodic,
    increment_x = false,
    increment_y = false;
    A_l = A,
    A_r = A,
)
    if periodic[1] && periodic[2]
        border_matrix_rows = size(A, 1)
    else
        border_matrix_rows = size(A_r, 1)
    end
    block_dims_info, block_size_info, new_block_size_info = block_information(
        block_dims,
        increment_x,
        increment_y,
        false,
        true,
        size(A, 1),
        border_matrix_rows,
    )
    A_cur = A
    bd_cur = block_dims
    obs_cur = block_size_info[1]
    nbs_cur = new_block_size_info[1]
    old = 1
    new = 1
    b = Vector{Float64}(undef, vector_length(n_dir, new_block_size_info))

    if periodic[1] && periodic[2]
        for i = 1:n_dir
            for j = 1:n_dir
                b[new:new+nbs_cur-1] =
                    vec(A * reshape(view(v, (old:old+obs_cur-1)), bd_cur))
                old += obs_cur
                new += nbs_cur
            end
        end
    elseif !periodic[1] && !periodic[2]
        for i = 1:n_dir
            if i == n_dir
                bd_cur = block_dims_info[3]
                obs_cur = block_size_info[3]
                nbs_cur = new_block_size_info[3]
            else
                bd_cur = block_dims_info[1]
                obs_cur = block_size_info[1]
                nbs_cur = new_block_size_info[1]
            end
            for j = 1:n_dir
                if j == n_dir
                    A_cur = A_r
                    if i == n_dir
                        bd_cur = block_dims_info[4]
                        obs_cur = block_size_info[4]
                        nbs_cur = new_block_size_info[4]
                    else
                        bd_cur = block_dims_info[2]
                        obs_cur = block_size_info[2]
                        nbs_cur = new_block_size_info[2]
                    end
                elseif j == 1
                    A_cur = A_l
                else
                    A_cur = A
                end
                b[new:new+nbs_cur-1] =
                    vec(A_cur * reshape(view(v, (old:old+obs_cur-1)), bd_cur))
                old += obs_cur
                new += nbs_cur
            end
        end
    elseif periodic[1]
        for i = 1:n_dir
            if i == n_dir
                bd_cur = block_dims_info[3]
                obs_cur = block_size_info[3]
                nbs_cur = new_block_size_info[3]
            else
                bd_cur = block_dims_info[1]
                obs_cur = block_size_info[1]
                nbs_cur = new_block_size_info[1]
            end
            for j = 1:n_dir
                b[new:new+nbs_cur-1] =
                    vec(A * reshape(view(v, (old:old+obs_cur-1)), bd_cur))
                old += obs_cur
                new += nbs_cur
            end
        end
    else
        for i = 1:n_dir
            bd_cur = block_dims_info[1]
            obs_cur = block_size_info[1]
            nbs_cur = new_block_size_info[1]
            for j = 1:n_dir
                if j == n_dir
                    A_cur = A_r
                    bd_cur = block_dims_info[2]
                    obs_cur = block_size_info[2]
                    nbs_cur = new_block_size_info[2]
                elseif j == 1
                    A_cur = A_l
                else
                    A_cur = A
                end
                b[new:new+nbs_cur-1] =
                    vec(A_cur * reshape(view(v, (old:old+obs_cur-1)), bd_cur))
                old += obs_cur
                new += nbs_cur
            end
        end
    end
    return b
end


"""
Emulates the product of a vector with a block matrix, 
where the blocks are Kronecker product matrices, 
and the block pattern is permutated from a diagonal form.
We assume here that one of the Kronecker 
product arguments is the identity matrix.
"""

@inline function product_kronecker_boundary(
    A,
    v,
    block_dims,
    offsets,
    offsets_opp,
    n_dir,
    periodic,
    boundary_matrix_opposite,
    increment_x = false,
    increment_y = false,
    left = true;
    A_r = A,
)
    if left
        return product_kronecker_boundary_left(
            A,
            v,
            block_dims,
            offsets,
            offsets_opp,
            n_dir,
            periodic,
            boundary_matrix_opposite,
            increment_x,
            increment_y,
            A_r = A_r,
        )
    else
        return product_kronecker_boundary_right(
            A,
            v,
            block_dims,
            offsets,
            offsets_opp,
            n_dir,
            periodic,
            boundary_matrix_opposite,
            increment_x,
            increment_y,
            A_r = A_r,
        )
    end
end

@inline function product_kronecker_boundary_left(
    A,
    v,
    block_dims,
    offsets,
    offsets_opp,
    n_dir,
    periodic,
    boundary_matrix_opposite,
    increment_x = false,
    increment_y = false;
    A_r = A,
)
    if periodic[2]
        border_matrix_rows = size(A, 1)
    else
        border_matrix_rows = size(A_r, 1)
    end
    block_dims_info, block_size_info, new_block_size_info = block_information(
        block_dims,
        increment_x,
        increment_y,
        true,
        false,
        size(A, 1),
        border_matrix_rows,
    )
    b = zeros(vector_length(n_dir, new_block_size_info))
    for i = 1:n_dir
        for j = 1:n_dir
            neighbor_exists,
            opp_neighbor_exists,
            neighbor_boundary,
            shape,
            old_bounds,
            new_bounds = element_shape_bounds(
                i,
                j,
                offsets,
                offsets_opp,
                n_dir,
                block_dims_info,
                block_size_info,
                new_block_size_info,
            )
            if neighbor_exists
                if (neighbor_boundary && !boundary_matrix_opposite) ||
                   (!opp_neighbor_exists && boundary_matrix_opposite)
                    b[new_bounds[1]:new_bounds[2]] = vec(
                        reshape(view(v, (old_bounds[1]:old_bounds[2])), shape) *
                        transpose(A_r),
                    )
                else
                    b[new_bounds[1]:new_bounds[2]] = vec(
                        reshape(view(v, (old_bounds[1]:old_bounds[2])), shape) *
                        transpose(A),
                    )
                end
            end
        end
    end
    return b
end

@inline function product_kronecker_boundary_right(
    A,
    v,
    block_dims,
    offsets,
    offsets_opp,
    n_dir,
    periodic,
    boundary_matrix_opposite,
    increment_x = false,
    increment_y = false;
    A_r = A,
)
    if periodic[1]
        border_matrix_rows = size(A, 1)
    else
        border_matrix_rows = size(A_r, 1)
    end
    block_dims_info, block_size_info, new_block_size_info = block_information(
        block_dims,
        increment_x,
        increment_y,
        false,
        true,
        size(A, 1),
        border_matrix_rows,
    )
    b = zeros(vector_length(n_dir, new_block_size_info))
    for i = 1:n_dir
        for j = 1:n_dir
            neighbor_exists,
            opp_neighbor_exists,
            neighbor_boundary,
            shape,
            old_bounds,
            new_bounds = element_shape_bounds(
                i,
                j,
                offsets,
                offsets_opp,
                n_dir,
                block_dims_info,
                block_size_info,
                new_block_size_info,
            )
            if neighbor_exists
                if (neighbor_boundary && !boundary_matrix_opposite) ||
                   (!opp_neighbor_exists && boundary_matrix_opposite)
                    b[new_bounds[1]:new_bounds[2]] =
                        vec(A_r * reshape(view(v, (old_bounds[1]:old_bounds[2])), shape))
                else
                    b[new_bounds[1]:new_bounds[2]] =
                        vec(A * reshape(view(v, (old_bounds[1]:old_bounds[2])), shape))
                end
            end
        end
    end
    return b
end

""" 
Combined Kronecker product operator for boundary operators
"""

@inline function product_kronecker_combined(
    A,
    A_bound,
    v,
    block_dims,
    n_dir,
    offsets,
    offsets_opp,
    periodic,
    boundary_matrix_opposite = false,
    increment_x = false,
    increment_y = false,
    left = true;
    A_l = A,
    A_r = A,
    A_r_bound = A_bound,
)
    if n_dir == 1 && offsets[1] == -1
        return product_kronecker(
            A,
            v,
            block_dims,
            n_dir,
            periodic,
            increment_x,
            increment_y,
            left,
            A_l = A_l,
            A_r = A_r,
        )
    end
    return product_kronecker(
        A,
        v,
        block_dims,
        n_dir,
        periodic,
        increment_x,
        increment_y,
        left,
        A_l = A_l,
        A_r = A_r,
    ) + product_kronecker_boundary(
        A_bound,
        v,
        block_dims,
        offsets,
        offsets_opp,
        n_dir,
        periodic,
        boundary_matrix_opposite,
        increment_x,
        increment_y,
        left,
        A_r = A_r_bound,
    )
end

function block_information(
    block_dims,
    increment_x,
    increment_y,
    left,
    right,
    nr_A,
    nr_Ar,
    nr_B = nr_A,
    nr_Br = nr_Ar,
)
    block_dims_bx = (block_dims[1] + increment_x, block_dims[2])
    block_dims_by = (block_dims[1], block_dims[2] + increment_y)
    block_dims_bxy = (block_dims[1] + increment_x, block_dims[2] + increment_y)
    block_size = block_dims[1] * block_dims[2]
    block_size_bx = block_dims_bx[1] * block_dims_bx[2]
    block_size_by = block_dims_by[1] * block_dims_by[2]
    block_size_bxy = block_dims_bxy[1] * block_dims_bxy[2]
    if left && right
        new_block_size = nr_A * nr_B
        new_block_size_bx = nr_A * nr_Br
        new_block_size_by = nr_Ar * nr_B
        new_block_size_bxy = nr_Ar * nr_Br
    elseif left
        new_block_size = block_dims[1] * nr_A
        new_block_size_bx = block_dims_bx[1] * nr_A
        new_block_size_by = block_dims_by[1] * nr_Ar
        new_block_size_bxy = block_dims_bxy[1] * nr_Ar
    else
        new_block_size = nr_A * block_dims[2]
        new_block_size_bx = nr_Ar * block_dims_bx[2]
        new_block_size_by = nr_A * block_dims_by[2]
        new_block_size_bxy = nr_Ar * block_dims_bxy[2]
    end
    return (block_dims, block_dims_bx, block_dims_by, block_dims_bxy),
    (block_size, block_size_bx, block_size_by, block_size_bxy),
    (new_block_size, new_block_size_bx, new_block_size_by, new_block_size_bxy)
end

function vector_length(n_dir, block_size_info)
    return block_size_info[1] * (n_dir - 1)^2 +
           block_size_info[2] * (n_dir - 1) +
           block_size_info[3] * (n_dir - 1) +
           block_size_info[4]
end

@inline function element_shape_bounds(
    i,
    j,
    offsets,
    offsets_opp,
    n_dir,
    block_dims_info,
    block_size_info,
    new_block_size_info,
)
    element_index = (i - 1) * n_dir + j
    neighbor_index = offsets[element_index]
    opp_neighbor_index = offsets_opp[element_index]
    neighbor_boundary = false
    opp_neighbor_exists = true
    if neighbor_index == -1
        return false, false, false, (0, 0), (0, 0), (0, 0)
    elseif offsets[neighbor_index] == -1
        neighbor_boundary = true
    end
    if opp_neighbor_index == -1
        opp_neighbor_exists = false
    end
    i_n = div(neighbor_index - 1, n_dir) + 1
    j_n = neighbor_index - (i_n - 1) * n_dir
    if i < n_dir && j < n_dir
        new_bounds = (
            new_block_size_info[1] * (n_dir - 1) * (i - 1) +
            new_block_size_info[2] * (i - 1) +
            new_block_size_info[1] * (j - 1) +
            1,
            new_block_size_info[1] * (n_dir - 1) * (i - 1) +
            new_block_size_info[2] * (i - 1) +
            new_block_size_info[1] * j,
        )
    elseif i < n_dir
        new_bounds = (
            new_block_size_info[1] * (n_dir - 1) * i + new_block_size_info[2] * (i - 1) + 1,
            new_block_size_info[1] * (n_dir - 1) * i + new_block_size_info[2] * i,
        )
    elseif j < n_dir
        new_bounds = (
            new_block_size_info[1] * (n_dir - 1)^2 +
            new_block_size_info[2] * (n_dir - 1) +
            new_block_size_info[3] * (j - 1) +
            1,
            new_block_size_info[1] * (n_dir - 1)^2 +
            new_block_size_info[2] * (n_dir - 1) +
            new_block_size_info[3] * j,
        )
    else
        new_bounds = (
            new_block_size_info[1] * (n_dir - 1)^2 +
            new_block_size_info[2] * (n_dir - 1) +
            new_block_size_info[3] * (n_dir - 1) +
            1,
            vector_length(n_dir, new_block_size_info),
        )
    end

    if i_n < n_dir && j_n < n_dir
        old_bounds = (
            block_size_info[1] * (n_dir - 1) * (i_n - 1) +
            block_size_info[2] * (i_n - 1) +
            block_size_info[1] * (j_n - 1) +
            1,
            block_size_info[1] * (n_dir - 1) * (i_n - 1) +
            block_size_info[2] * (i_n - 1) +
            block_size_info[1] * j_n,
        )
        shape = block_dims_info[1]
    elseif i_n < n_dir
        old_bounds = (
            block_size_info[1] * (n_dir - 1) * i_n + block_size_info[2] * (i_n - 1) + 1,
            block_size_info[1] * (n_dir - 1) * i_n + block_size_info[2] * i_n,
        )
        shape = block_dims_info[2]
    elseif j_n < n_dir
        old_bounds = (
            block_size_info[1] * (n_dir - 1)^2 +
            block_size_info[2] * (n_dir - 1) +
            block_size_info[3] * (j_n - 1) +
            1,
            block_size_info[1] * (n_dir - 1)^2 +
            block_size_info[2] * (n_dir - 1) +
            block_size_info[3] * j_n,
        )
        shape = block_dims_info[3]
    else
        old_bounds = (
            block_size_info[1] * (n_dir - 1)^2 +
            block_size_info[2] * (n_dir - 1) +
            block_size_info[3] * (n_dir - 1) +
            1,
            vector_length(n_dir, block_size_info),
        )
        shape = block_dims_info[4]
    end
    return true, opp_neighbor_exists, neighbor_boundary, shape, old_bounds, new_bounds
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
    div =
        product_kronecker_combined(
            semi.delta,
            semi.delta_boundary,
            Ex,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.right_neighbors,
            semi.left_neighbors,
            semi.periodic,
            false,
            semi.increments_x[1],
            semi.increments_y[1],
            false,
            A_r = semi.delta_right,
            A_r_bound = semi.delta_boundary_right,
        ) + product_kronecker_combined(
            semi.delta,
            semi.delta_boundary,
            Ey,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.upper_neighbors,
            semi.lower_neighbors,
            semi.periodic,
            false,
            semi.increments_x[2],
            semi.increments_y[2],
            true,
            A_r = semi.delta_right,
            A_r_bound = semi.delta_boundary_right,
        )
    return product_kronecker_general(
        semi.V,
        semi.V,
        div,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.periodic,
        false,
        false,
    )
end


"""
Compute curl of B (scalar in 2D)
"""
function compute_curl(semi::SemiDiscretizationFEECSparse, B)
    B_x = product_kronecker_combined(
        semi.delta,
        semi.delta_boundary,
        B,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.right_neighbors,
        semi.left_neighbors,
        semi.periodic,
        false,
        semi.increments_x[3],
        semi.increments_y[3],
        false,
        A_r = semi.delta_right,
        A_r_bound = semi.delta_boundary_right,
    )
    B_y = product_kronecker_combined(
        semi.delta,
        semi.delta_boundary,
        B,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.upper_neighbors,
        semi.lower_neighbors,
        semi.periodic,
        false,
        semi.increments_x[3],
        semi.increments_y[3],
        true,
        A_r = semi.delta_right,
        A_r_bound = semi.delta_boundary_right,
    )
    return B_y, -B_x
end

"""
Compute RHS
"""
function compute_rhs!(du, u, semi::SemiDiscretizationFEEC, t; strong = false)
    Ex, Ey, Bz = u
    Ex_t, Ey_t = compute_curl(semi, Bz)
    Bz_t =
        semi.G1 * (semi.Winv * semi.K * Ey - Ex * transpose(semi.K) * semi.Winv) * semi.G1
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
    w_0 = semi.W[1, 1]
    w_N = semi.W[N+1, N+1]
    w_inv = 1 / (w_0 + w_N)
    ind = 1
    for i = 1:m
        for j = 1:m
            i_l = i - 1
            j_l = j - 1
            if j == 1
                semi.periodic[1] ? j_l = m : j_l = -1
            end
            if i == 1
                semi.periodic[2] ? i_l = m : i_l = -1
            end
            !semi.periodic[1] && j == m ? inc_x = 1 : inc_x = 0
            !semi.periodic[2] && i == m ? inc_y = 1 : inc_y = 0
            semi.periodic[1] ? offset_Ex = ((i - 1) * m + (j - 1)) * N * (N + 1) :
            offset_Ex = ((i - 1) * (m - 1) + (j - 1)) * N * (N + 1) + (i - 1) * (N + 1)^2
            semi.periodic[1] ? offset_neighbor_y = ((i_l - 1) * m + (j - 1)) * N * (N + 1) :
            offset_neighbor_y =
                ((i_l - 1) * (m - 1) + (j - 1)) * N * (N + 1) + (i_l - 1) * (N + 1)^2
            offset_Ey = ((i - 1) * m + (j - 1)) * N * (N + 1)
            offset_neighbor_x = ((i - 1) * m + (j_l - 1)) * N * (N + 1)
            if i == m && !semi.periodic[2]
                offset_Ey += (j - 1) * (N + 1)
                offset_neighbor_x += (j_l - 1) * (N + 1)
            end
            offset_Bz = (i-1) * m * N^2 + (j-1) * N^2 + (i-1) * !semi.periodic[1] * N + (j-1) * inc_y * N
            Bz_t[(offset_Bz+1):(offset_Bz+(N+inc_x)*(N+inc_y))] = vec(semi.D[1:(N+inc_x),:] * transpose(reshape(view(Ex, (offset_Ex+1):(offset_Ex+(N+inc_x)*(N+1))), N + inc_x, N + 1))
                                                                  - transpose(semi.D[1:(N+inc_y),:] * reshape(view(Ey, (offset_Ex+1):(offset_Ex+(N+1)*(N+inc_y))), N + 1, N + inc_y)))
            for l = 1:(N+inc_y)
                offset_local = offset_Ey + (l - 1) * (N + 1)
                offset_neighbor_local = offset_neighbor_x + (l - 1) * (N + 1)
                for k = 1:(N+inc_x)
                    if k == 1 || l == 1 || k == N + 1 || l == N + 1
                        if k == 1
                            if !semi.periodic[1] && j == 1
                                Bz_t[ind] =
                                    -sum(semi.D[k, r] * Ey[offset_local+r] for r = 1:(N+1))
                                Bz_t[ind] -= Ey[offset_local+1] / w_0
                            else
                                s = Ey[offset_neighbor_local+(N+1)]
                                s -= Ey[offset_local+1]
                                s -=
                                    w_N * sum(
                                        semi.D[N+1, r] * Ey[offset_neighbor_local+r] for
                                        r = 1:(N+1)
                                    )
                                s -=
                                    w_0 * sum(semi.D[1, r] * Ey[offset_local+r] for r = 1:(N+1))
                                Bz_t[ind] = w_inv * s
                            end
                        elseif k == N + 1
                            Bz_t[ind] = -sum(semi.D[k, r] * Ey[offset_local+r] for r = 1:(N+1))
                            Bz_t[ind] += Ey[offset_local+N+1] / w_N
                        else
                            Bz_t[ind] = -sum(semi.D[k, r] * Ey[offset_local+r] for r = 1:(N+1))
                        end
                        if l == 1
                            if !semi.periodic[2] && i == 1
                                Bz_t[ind] += sum(
                                    semi.D[l, r] * Ex[offset_Ex+(r-1)*(N+inc_x)+k] for
                                    r = 1:(N+1)
                                )
                                Bz_t[ind] += Ex[offset_Ex+k] / w_0
                            else
                                s = -Ex[offset_neighbor_y+N*(N+inc_x)+k]
                                s += Ex[offset_Ex+k]
                                s +=
                                    w_N * sum(
                                        semi.D[N+1, r] *
                                        Ex[offset_neighbor_y+(r-1)*(N+inc_x)+k] for r = 1:(N+1)
                                    )
                                s +=
                                    w_0 * sum(
                                        semi.D[1, r] * Ex[offset_Ex+(r-1)*(N+inc_x)+k] for
                                        r = 1:(N+1)
                                    )
                                Bz_t[ind] += w_inv * s
                            end
                        elseif l == N + 1
                            Bz_t[ind] += sum(
                                semi.D[l, r] * Ex[offset_Ex+(r-1)*(N+inc_x)+k] for r = 1:(N+1)
                            )
                            Bz_t[ind] -= Ex[offset_Ex+N*(N+inc_x)+k] / w_N
                        else
                            Bz_t[ind] += sum(
                                semi.D[l, r] * Ex[offset_Ex+(r-1)*(N+inc_x)+k] for r = 1:(N+1)
                            )
                        end
                    end
                    ind += 1
                end
            end
        end
    end
    du[3] = Bz_t
    apply_ess_zero_boundary_cons!(du[1], du[2], du[3], semi)
    return nothing
end

function compute_rhs_weak!(du, u, semi::SemiDiscretizationFEECSparse, t)
    Ex, Ey, Bz = u
    B_x = product_kronecker_combined(
        semi.Wd,
        semi.Wd_boundary,
        Ex,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.lower_neighbors,
        semi.upper_neighbors,
        semi.periodic,
        true,
        semi.increments_x[1],
        semi.increments_y[1],
        true,
        A_l = semi.Wd_left,
        A_r = semi.Wd_right,
        A_r_bound = semi.Wd_boundary_right,
    )
    B_y = product_kronecker_combined(
        semi.Wd,
        semi.Wd_boundary,
        Ey,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.left_neighbors,
        semi.right_neighbors,
        semi.periodic,
        true,
        semi.increments_x[2],
        semi.increments_y[2],
        false,
        A_l = semi.Wd_left,
        A_r = semi.Wd_right,
        A_r_bound = semi.Wd_boundary_right,
    )
    du[3] = B_y - B_x
    du[1], du[2] = compute_curl(semi, Bz)
    # function call can change all three vectors depending on specified boundary conditions
    apply_ess_zero_boundary_cons!(du[1], du[2], du[3], semi)
    return nothing
end

"""
Applies essential zero boundary conditions in non-periodic directions with essential boundary conditions specified in the semi-discretization
"""

function apply_ess_zero_boundary_cons!(Ex, Ey, Bz, semi)
    N = semi.N
    ne = semi.n_elements_direction
    if !semi.periodic[1] && semi.essential[1]
        offset = 0
        offset_right_side = (ne - 1) * N^2
        offset_right_element = N * (N + 1)
        for i = 1:(ne-!semi.periodic[2])
            for j = 1:N
                Ex[offset+(j-1)*N+1] = 0
                Bz[offset+(j-1)*N+1] = 0
            end
            offset += offset_right_side
            for j = 1:N
                Ex[offset+j*(N+1)] = 0
                Bz[offset+j*(N+1)] = 0
            end
            offset += offset_right_element
        end
        if !semi.periodic[2]
            offset_right_side_Bz = (ne - 1) * N * (N + 1)
            offset_Bz = offset
            for j = 1:N
                Ex[offset+(j-1)*N+1] = 0
                Bz[offset_Bz+(j-1)*N+1] = 0
            end
            Bz[offset+N^2+1] = 0
            offset += offset_right_side
            offset_Bz += offset_right_side_Bz
            for j = 1:N
                Ex[offset+j*(N+1)] = 0
                Bz[offset_Bz+j*(N+1)] = 0
            end
            Bz[offset+(N+1)^2] = 0
        end
    end
    if !semi.periodic[2] && semi.essential[2]
        offset = 0
        for i = 1:ne
            for j = 1:N
                Ey[offset+j] = 0
                Bz[offset+j] = 0
            end
            offset += N^2
        end
        if !semi.periodic[1]
            Bz[offset-N^2+N+1] = 0
        end
        offset = (ne - 1) * ne * N^2
        offset_Bz = (ne - 1)^2 * N^2 + (ne - 1) * N * (N + !semi.periodic[1])
        for i = 1:(ne-1)
            offset += N^2
            offset_Bz += N^2
            for j = 1:N
                Ey[offset+j] = 0
                Bz[offset_Bz+j] = 0
            end
            offset += N
            offset_Bz += N
        end
        offset += N^2
        offset_Bz += N * (N + !semi.periodic[1])
        for j = 1:N
            Ey[offset+j] = 0
            Bz[offset_Bz+j] = 0
        end
        if !semi.periodic[1]
            Bz[offset_Bz+N+1] = 0
        end
    end
    return nothing
end


@inline function initial_condition_periodic(x, t)
    Ex =
        -0.5 *
        sqrt(2) *
        cos(pi * x[1] + pi) *
        sin(pi * x[2] + pi) *
        sin((pi * 2 / sqrt(2)) * t)
    Ey =
        0.5 *
        sqrt(2) *
        sin(pi * x[1] + pi) *
        cos(pi * x[2] + pi) *
        sin((pi * 2 / sqrt(2)) * t)
    Bz = cos(pi * x[1] + pi) * cos(pi * x[2] + pi) * cos((pi * 2 / sqrt(2)) * t)
    return (Ex, Ey, Bz)
end

@inline function initial_condition_non_periodic(x, t)
    Ex = cos(5 * pi * x[1]) * cos(5 * pi * x[2]) * sin(5 * pi * sqrt(2) * t)
    Ey = sin(5 * pi * x[1]) * sin(5 * pi * x[2]) * sin(5 * pi * sqrt(2) * t)
    Bz = sqrt(2) * cos(5 * pi * x[1]) * sin(5 * pi * x[2]) * cos(5 * pi * sqrt(2) * t)
    return (Ex, Ey, Bz)
end

function initial_condition_nodal(f, semi::SemiDiscretizationFEECSparse, t)
    md = semi.n_elements_direction
    dx = (semi.b - semi.a) / md
    m = semi.n_elements
    N = semi.N
    px = 1
    nodes = semi.element_nodes
    bz_length = m * N^2
    ex_length = m * N * (N + 1)
    ey_length = m * N * (N + 1)
    if !semi.periodic[1] && !semi.periodic[2]
        bz_length += 2 * md * N + 1
        ex_length += md * (N + 1)
        ey_length += md * (N + 1)
    elseif !semi.periodic[1]
        bz_length += md * N
        ex_length += md * (N + 1)
    elseif !semi.periodic[2]
        bz_length += md * N
        ey_length += md * (N + 1)
    end
    if semi.periodic[1]
        px = 0
    end
    # Mag field
    Bz = zeros(bz_length)
    idx = 0
    for o = 1:md
        for k = 1:md
            for j = 1:N
                if k == md && !semi.periodic[1]
                    for i = 1:(N+1)
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Bz[idx] = f((x, y), t)[3]
                    end
                else
                    for i = 1:N
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Bz[idx] = f((x, y), t)[3]
                    end
                end
            end
            if o == md && !semi.periodic[2]
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] + semi.a
                    idx += 1
                    Bz[idx] = f((x, semi.b), t)[3]
                end
            end
        end
    end
    if !semi.periodic[1] && !semi.periodic[2]
        Bz[end] = f((semi.b, semi.b), t)[3]
    end
    # Electric field
    Ex = zeros(ex_length)
    idx = 0
    for o = 1:md
        for k = 1:md
            for j = 1:(N+1)
                if k == md && !semi.periodic[1]
                    for i = 1:(N+1)
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Ex[idx] = f((x, y), t)[1]
                    end
                else
                    for i = 1:N
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Ex[idx] = f((x, y), t)[1]
                    end
                end
            end
        end
    end
    Ey = zeros(ey_length)
    idx = 0
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:(N+1)
                    x = (k - 1) * dx + nodes[i] + semi.a
                    y = (o - 1) * dx + nodes[j] + semi.a
                    idx += 1
                    Ey[idx] = f((x, y), t)[2]
                end
            end
            if o == md && !semi.periodic[2]
                for i = 1:(N+1)
                    x = (k - 1) * dx + nodes[i] + semi.a
                    idx += 1
                    Ey[idx] = f((x, semi.b), t)[2]
                end
            end
        end
    end
    return [Ex, Ey, Bz]
end

function initial_condition_projected(f, semi::SemiDiscretizationFEECSparse, t)
    md = semi.n_elements_direction
    dx = (semi.b - semi.a) / md
    m = semi.n_elements
    N = semi.N
    px = 1
    points, weights = gauss(20, 0, 1.0)
    nodes = semi.element_nodes
    bz_length = m * N^2
    ex_length = m * N^2
    ey_length = m * N^2
    if !semi.periodic[1] && !semi.periodic[2]
        bz_length += 2 * md * N + 1
        ex_length += md * N
        ey_length += md * N
    elseif !semi.periodic[1]
        bz_length += md * N
        ex_length += md * N
    elseif !semi.periodic[2]
        bz_length += md * N
        ey_length += md * N
    end
    if semi.periodic[1]
        px = 0
    end

    # Mag field
    Bz = zeros(bz_length)
    idx = 0
    for o = 1:md
        for k = 1:md
            for j = 1:N
                if k == md && !semi.periodic[1]
                    for i = 1:(N+1)
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Bz[idx] = f((x, y), t)[3]
                    end
                else
                    for i = 1:N
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Bz[idx] = f((x, y), t)[3]
                    end
                end
            end
            if o == md && !semi.periodic[2]
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] + semi.a
                    idx += 1
                    Bz[idx] = f((x, semi.b), t)[3]
                end
            end
        end
    end
    if !semi.periodic[1] && !semi.periodic[2]
        Bz[end] = f((semi.b, semi.b), t)[3]
    end
    # Electric field
    Ex = zeros(ex_length)
    idx = 0
    for o = 1:md
        for k = 1:md
            for j = 1:N
                subl = nodes[j+1] - nodes[j]
                if k == md && !semi.periodic[1]
                    for i = 1:(N+1)
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Ex[idx] = sum(
                            subl * weights[k] * f((x, y + subl * points[k]), t)[1] for
                            k = 1:20
                        )
                    end
                else
                    for i = 1:N
                        x = (k - 1) * dx + nodes[i] + semi.a
                        y = (o - 1) * dx + nodes[j] + semi.a
                        idx += 1
                        Ex[idx] = sum(
                            subl * weights[k] * f((x, y + subl * points[k]), t)[1] for
                            k = 1:20
                        )
                    end
                end
            end
        end
    end
    Ey = zeros(ey_length)
    idx = 0
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:N
                    subl = nodes[i+1] - nodes[i]
                    x = (k - 1) * dx + nodes[i] + semi.a
                    y = (o - 1) * dx + nodes[j] + semi.a
                    idx += 1
                    Ey[idx] = sum(
                        subl * weights[k] * f((x + subl * points[k], y), t)[2] for
                        k = 1:20
                    )
                end
            end
            if o == md && !semi.periodic[2]
                for i = 1:N
                    subl = nodes[i+1] - nodes[i]
                    x = (k - 1) * dx + nodes[i] + semi.a
                    idx += 1
                    Ey[idx] = sum(
                        subl * weights[k] * f((x + subl * points[k], semi.b), t)[2] for
                        k = 1:20
                    )
                end
            end
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
function initial_condition_nodal(semi, t)
    x = y = semi.nodes
    N = semi.N
    # Mag field
    Bz = zeros(Float64, N + 1, N + 1)
    for j = 1:(N+1)
        for i = 1:(N+1)
            Bz[i, j] =
                cos(pi * x[i] + pi) * cos(pi * y[j] + pi) * cos((pi * 2 / sqrt(2)) * t)
        end
    end
    # Electric field
    Ex = zeros(Float64, N + 1, N + 1)
    for j = 1:(N+1)
        for i = 1:(N+1)
            Ex[i, j] =
                -0.5 *
                sqrt(2) *
                cos(pi * x[i] + pi) *
                sin(pi * y[j] + pi) *
                sin((pi * 2 / sqrt(2)) * t)
        end
    end
    Ey = zeros(Float64, N + 1, N + 1)
    for j = 1:(N+1)
        for i = 1:(N+1)
            Ey[i, j] =
                0.5 *
                sqrt(2) *
                sin(pi * x[i] + pi) *
                cos(pi * y[j] + pi) *
                sin((pi * 2 / sqrt(2)) * t)
        end
    end

    return (Ex, Ey, Bz)
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
    m = semi.n_elements
    N = semi.N
    nodes = semi.element_nodes
    # Mag field
    Bz = zeros(Float64, m * N^2)
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Bz[idx] =
                        cos(pi * x + pi) * cos(pi * y + pi) * cos((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    # Electric field
    Ex = zeros(Float64, m * N * (N + 1))
    for o = 1:md
        for k = 1:md
            for j = 1:(N+1)
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx =
                        (o - 1) * md * N * (N + 1) + (k - 1) * N * (N + 1) + (j - 1) * N + i
                    Ex[idx] =
                        -0.5 *
                        sqrt(2) *
                        cos(pi * x + pi) *
                        sin(pi * y + pi) *
                        sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    Ey = zeros(Float64, m * N * (N + 1))
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:(N+1)
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx =
                        (o - 1) * md * N * (N + 1) +
                        (k - 1) * N * (N + 1) +
                        (j - 1) * (N + 1) +
                        i
                    Ey[idx] =
                        0.5 *
                        sqrt(2) *
                        sin(pi * x + pi) *
                        cos(pi * y + pi) *
                        sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    return (Ex, Ey, Bz)
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
    for j = 1:(N+1)
        for i = 1:(N+1)
            Bz[i, j] =
                cos(pi * x[i] + pi) * cos(pi * y[j] + pi) * cos((pi * 2 / sqrt(2)) * t)
        end
    end
    # Electric field
    Ex = zeros(Float64, N + 1, N)
    for j = 1:N
        for i = 1:(N+1)
            Ex[i, j] =
                0.5 *
                sqrt(2) *
                (
                    cos(pi * x[i] + pi) *
                    (1 / pi) *
                    (cos(pi * y[j+1] + pi) - cos(pi * y[j] + pi))
                ) *
                sin((pi * 2 / sqrt(2)) * t)
        end
    end
    Ey = zeros(Float64, N, N + 1)
    for j = 1:(N+1)
        for i = 1:N
            Ey[i, j] =
                0.5 *
                sqrt(2) *
                (
                    cos(pi * y[j] + pi) *
                    (-1 / pi) *
                    (cos(pi * x[i+1] + pi) - cos(pi * x[i] + pi))
                ) *
                sin((pi * 2 / sqrt(2)) * t)
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
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Bz[idx] =
                        cos(pi * x + pi) * cos(pi * y + pi) * cos((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    # Electric field
    Ex = zeros(Float64, m * N^2)
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    yp1 = (o - 1) * dx + nodes[j+1] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Ex[idx] =
                        0.5 *
                        sqrt(2) *
                        (
                            cos(pi * x + pi) *
                            (1 / pi) *
                            (cos(pi * yp1 + pi) - cos(pi * y + pi))
                        ) *
                        sin((pi * 2 / sqrt(2)) * t)
                end
            end
        end
    end
    Ey = zeros(Float64, m * N^2)
    for o = 1:md
        for k = 1:md
            for j = 1:N
                for i = 1:N
                    x = (k - 1) * dx + nodes[i] - 1
                    y = (o - 1) * dx + nodes[j] - 1
                    xp1 = (k - 1) * dx + nodes[i+1] - 1
                    idx = (o - 1) * md * N^2 + (k - 1) * N^2 + (j - 1) * N + i
                    Ey[idx] =
                        0.5 *
                        sqrt(2) *
                        (
                            cos(pi * y + pi) *
                            (-1 / pi) *
                            (cos(pi * xp1 + pi) - cos(pi * x + pi))
                        ) *
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
    Ex_nodal = product_kronecker(
        semi.V,
        Ex,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.periodic,
        semi.increments_x[1],
        semi.increments_y[1],
        true,
    )
    Ey_nodal = product_kronecker(
        semi.V,
        Ey,
        (semi.N, semi.N),
        semi.n_elements_direction,
        semi.periodic,
        semi.increments_x[2],
        semi.increments_y[2],
        false,
    )
    Bz_nodal = Bz
    return Ex_nodal, Ey_nodal, Bz_nodal
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
    G1[1, N+1] = 0.5
    G1[N+1, 1] = 0.5
    G1[N+1, N+1] = 0.5
    for i = 2:N
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
    Bz_t =
        semi.G1 * (semi.Winv * semi.K * Ey - Ex * transpose(semi.K) * semi.Winv) * semi.G1
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
    p1 = heatmap(
        vec(semi.nodes),
        vec(semi.nodes),
        transpose(Ex_nodal),
        levels = 20,
        color = :turbo,
    )
    p2 = heatmap(
        vec(semi.nodes),
        vec(semi.nodes),
        transpose(Ey_nodal),
        levels = 20,
        color = :turbo,
    )
    p3 = heatmap(
        vec(semi.nodes),
        vec(semi.nodes),
        transpose(Bz),
        levels = 20,
        color = :turbo,
    )
    p4 = heatmap(
        vec(semi.nodes),
        vec(semi.nodes),
        transpose(divE),
        levels = 20,
        color = :turbo,
    )
    return plot(
        p1,
        p2,
        p3,
        p4,
        layout = (2, 2),
        title = [prefix * "Ex" prefix * "Ey" prefix * "Bz" prefix * "divE"],
        right_margin = 10mm,
    )
end

function plot_variables(semi::SemiDiscretizationFEECSparse, u; prefix = "")
    Ex, Ey, Bz = u
    domain_length = semi.b - semi.a
    # Transform to nodal
    md = semi.n_elements_direction
    dx = domain_length / md
    N = semi.N
    points_dc = zeros(md * (N + 1))
    points_cont = zeros(md * N + 1)
    for i = 0:(md-1)
        for j = 1:N
            points_cont[i*N+j] = i * dx + semi.element_nodes[j]
        end
        for j = 2:N
            points_dc[i*(N+1)+j] = i * dx + semi.element_nodes[j]
        end
        points_dc[i*(N+1)+1] = i * dx + 1e-10
        points_dc[(i+1)*(N+1)] = (i + 1) * dx - 1e-10
    end
    points_dc .+= semi.a
    points_cont .+= semi.a
    points_cont[end] = semi.b
    points_dc[1] = semi.a
    points_dc[end] = semi.b

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
    return plot(
        p1,
        p2,
        p3,
        p4,
        layout = (2, 2),
        title = [prefix * "Ex" prefix * "Ey" prefix * "Bz" prefix * "divE"],
        right_margin = 10mm,
    )
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

function discrete_gradient!(u, semi::SemiDiscretizationFEECSparse, dt)
    Ex, Ey, Bz = u

    b = [
        Ex +
        0.5 *
        dt *
        product_kronecker_combined(
            semi.delta,
            semi.delta_boundary,
            Bz,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.upper_neighbors,
            semi.lower_neighbors,
            semi.periodic,
            false,
            semi.increments_x[3],
            semi.increments_y[3],
            true,
            A_r = semi.delta_right,
            A_r_bound = semi.delta_boundary_right,
        )
        Ey -
        0.5 *
        dt *
        product_kronecker_combined(
            semi.delta,
            semi.delta_boundary,
            Bz,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.right_neighbors,
            semi.left_neighbors,
            semi.periodic,
            false,
            semi.increments_x[3],
            semi.increments_y[3],
            false,
            A_r = semi.delta_right,
            A_r_bound = semi.delta_boundary_right,
        )
        Bz -
        0.5 *
        dt *
        product_kronecker_combined(
            semi.Wd,
            semi.Wd_boundary,
            Ex,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.lower_neighbors,
            semi.upper_neighbors,
            semi.periodic,
            true,
            semi.increments_x[1],
            semi.increments_y[1],
            true,
            A_l = semi.Wd_left,
            A_r = semi.Wd_right,
            A_r_bound = semi.Wd_boundary_right,
        ) +
        0.5 *
        dt *
        product_kronecker_combined(
            semi.Wd,
            semi.Wd_boundary,
            Ey,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.left_neighbors,
            semi.right_neighbors,
            semi.periodic,
            true,
            semi.increments_x[2],
            semi.increments_y[2],
            false,
            A_l = semi.Wd_left,
            A_r = semi.Wd_right,
            A_r_bound = semi.Wd_boundary_right,
        )
    ]
    A = LinearProblemFEECSparse(semi, dt)
    apply_ess_zero_boundary_cons!(
        view(b, A.ranges[1]),
        view(b, A.ranges[2]),
        view(b, A.ranges[3]),
        semi,
    )
    x = reduce(vcat, u)
    x, _ = krylov_solve(:gmres, A, b, x, atol = 1e-12, rtol = 1e-12)
    u[1] .= view(x, A.ranges[1])
    u[2] .= view(x, A.ranges[2])
    u[3] .= view(x, A.ranges[3])
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

    e =
        transpose(Ex) * product_kronecker_general(
            transpose(semi.V) * semi.W * semi.V,
            semi.W_hat,
            Ex,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.periodic,
            semi.increments_x[1],
            semi.increments_y[1],
            B_l = view(semi.W, 1:semi.N, 1:semi.N),
            B_r = semi.W_hat_right,
        )
    e +=
        transpose(Ey) * product_kronecker_general(
            semi.W_hat,
            transpose(semi.V) * semi.W * semi.V,
            Ey,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.periodic,
            semi.increments_x[2],
            semi.increments_y[2],
            A_l = view(semi.W, 1:semi.N, 1:semi.N),
            A_r = semi.W_hat_right,
        )
    e +=
        transpose(Bz) * product_kronecker_general(
            semi.W_hat,
            semi.W_hat,
            Bz,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.periodic,
            semi.increments_x[3],
            semi.increments_y[3],
            A_l = view(semi.W, 1:semi.N, 1:semi.N),
            A_r = semi.W_hat_right,
            B_l = view(semi.W, 1:semi.N, 1:semi.N),
            B_r = semi.W_hat_right,
        )
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

    de =
        transpose(Ex) * product_kronecker_general(
            transpose(semi.V) * semi.W * semi.V,
            semi.W_hat,
            dEx,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.periodic,
            semi.increments_x[1],
            semi.increments_y[1],
            B_l = view(semi.W, 1:semi.N, 1:semi.N),
            B_r = semi.W_hat_right,
        )
    de +=
        transpose(Ey) * product_kronecker_general(
            semi.W_hat,
            transpose(semi.V) * semi.W * semi.V,
            dEy,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.periodic,
            semi.increments_x[2],
            semi.increments_y[2],
            A_l = view(semi.W, 1:semi.N, 1:semi.N),
            A_r = semi.W_hat_right,
        )
    de +=
        transpose(Bz) * product_kronecker_general(
            semi.W_hat,
            semi.W_hat,
            dBz,
            (semi.N, semi.N),
            semi.n_elements_direction,
            semi.periodic,
            semi.increments_x[3],
            semi.increments_y[3],
            A_l = view(semi.W, 1:semi.N, 1:semi.N),
            A_r = semi.W_hat_right,
            B_l = view(semi.W, 1:semi.N, 1:semi.N),
            B_r = semi.W_hat_right,
        )
    return de
end

function determine_timestep(semi, cfl; implicit = false, constant = false)
    if constant
        return 2e-5
    else
        return cfl * minimum(diag(semi.W))
    end
end

function timedisc!(
    u,
    semi,
    tspan,
    cfl;
    dt_analysis = 0.1,
    save_visu = false,
    implicit = false,
    strong = false,
    constant = false,
)
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
    println(
        @sprintf(
            "%5s %12s %12s %12s %12s",
            "iter",
            "time",
            "total_energy",
            "max|div(E)|",
            "d(Ener)/dt"
        )
    )
    println(
        @sprintf(
            "%5d %12.6e %12.6e %12.6e %12.6e",
            0,
            t,
            energy[1],
            div[1],
            compute_denergy_dt(semi, u, du)
        )
    )
    t_analysis = 0
    if save_visu
        plot = plot_variables(semi, u)
        savefig(plot, @sprintf("output_%.4f.png", t_analysis))
    end
    for n = 1:100000000
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
            #if dt < determine_timestep(semi, cfl, implicit)
            #ssprk33!(du, u, semi, t, dt, r0, strong = strong)
            discrete_gradient!(u, semi, dt)
            #=
            else
                Ex, Ey, Bz = u
                b = [Ex + 0.5 * dt * semi.delta_y * Bz; Ey - 0.5 * dt * semi.delta_x * Bz; 
                                -0.5 * dt * semi.Wd_x * Ex + 0.5 * dt * semi.Wd_y * Ey + Bz]
                b = Q' * (R\b)
                u[1] = b[1:n[1]]
                u[2] = b[n[1]+1:2*n[1]]
                u[3] = b[2*n[1]+1:3*n[1]]
            end 
            =#

        end
        t = t + dt

        if analyze || finish
            compute_rhs!(du, u, semi, t, strong = strong)
            energy[counter] = compute_energy(semi, u)
            div[counter] = maximum(abs.(compute_div(semi, u[1], u[2])))
            println(
                @sprintf(
                    "%5d %12.6e %12.6e %12.6e %12.6e",
                    n,
                    t,
                    energy[counter],
                    div[counter],
                    compute_denergy_dt(semi, u, du)
                )
            )
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
        return sqrt(
            transpose(field) * product_kronecker_general(
                semi.W,
                semi.W,
                field,
                (semi.N + 1, semi.N + 1),
                semi.n_elements_direction,
                semi.periodic,
                false,
                false,
            ),
        )
    elseif cont_x && cont_y
        return sqrt(
            transpose(field) * product_kronecker_general(
                semi.W_hat,
                semi.W_hat,
                field,
                (semi.N, semi.N),
                semi.n_elements_direction,
                semi.periodic,
                !semi.periodic[1],
                !semi.periodic[2],
                A_l = view(semi.W, 1:semi.N, 1:semi.N),
                A_r = semi.W_hat_right,
                B_l = view(semi.W, 1:semi.N, 1:semi.N),
                B_r = semi.W_hat_right,
            ),
        )
    elseif cont_x
        return sqrt(
            transpose(field) * product_kronecker_general(
                semi.W,
                semi.W_hat,
                field,
                (semi.N, semi.N + 1),
                semi.n_elements_direction,
                semi.periodic,
                !semi.periodic[1],
                false,
                B_l = view(semi.W, 1:semi.N, 1:semi.N),
                B_r = semi.W_hat_right,
            ),
        )
    else
        return sqrt(
            transpose(field) * product_kronecker_general(
                semi.W_hat,
                semi.W,
                field,
                (semi.N + 1, semi.N),
                semi.n_elements_direction,
                semi.periodic,
                false,
                !semi.periodic[2],
                A_l = view(semi.W, 1:semi.N, 1:semi.N),
                A_r = semi.W_hat_right,
            ),
        )
    end
end

""" 
Takes a vector of nodal field values and converts it to a matrix for plotting purposes.
To determine the proper dimensions of the matrix, we need to specify in which directions we are continuous or discontinuous.
"""
function convert_to_matrix(
    semi::SemiDiscretizationFEECSparse,
    field,
    cont_x::Bool,
    cont_y::Bool,
)
    periodic = semi.periodic
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
        for i = 1:(md-!periodic[2])
            offset =
                (i - 1) * (md - !periodic[1]) * N^2 + !periodic[1] * (i - 1) * N * (N + 1)
            for j = 1:md
                !periodic[1] && j == md ? range_x = range(((j - 1) * N + 1), (j * N + 1)) :
                range_x = range(((j - 1) * N + 1), (j * N))
                !periodic[1] && j == md ? block_size = (N + 1) * N : block_size = N^2
                !periodic[1] && j == md ? bd = 1 : bd = 0
                range_y = ((i-1)*N+1):(i*N)
                field_matrix[range_y, range_x] = transpose(
                    reshape(view(field, (offset+1):(offset+block_size)), N + bd, N),
                )
                offset += block_size
            end
        end
        if !periodic[2]
            offset =
                (md - 1) * (md - !periodic[1]) * N^2 + !periodic[1] * (md - 1) * N * (N + 1)
            range_y = ((md-1)*N+1):(md*N+1)
            for j = 1:md
                !periodic[1] && j == md ? range_x = range(((j - 1) * N + 1), (j * N + 1)) :
                range_x = range((j - 1) * N + 1, (j * N))
                !periodic[1] && j == md ? block_size = (N + 1)^2 : block_size = (N + 1) * N
                !periodic[1] && j == md ? bd = 1 : bd = 0
                field_matrix[range_y, range_x] = transpose(
                    reshape(view(field, (offset+1):(offset+block_size)), N + bd, N + 1),
                )
                offset += block_size
            end
        end
        if periodic[1]
            field_matrix[:, end] = view(field_matrix, :, 1)
        end
        if periodic[2]
            field_matrix[end, :] = view(field_matrix, 1, :)
        end
    elseif cont_x
        for i = 1:md
            offset = (i - 1) * (md * N * (N + 1) + !periodic[1] * (N + 1))
            range_y = ((i-1)*(N+1)+1):(i*(N+1))
            for j = 1:md
                !periodic[1] && j == md ? range_x = range(((j - 1) * N + 1), (j * N + 1)) :
                range_x = range(((j - 1) * N + 1), (j * N))
                !periodic[1] && j == md ? block_size = (N + 1)^2 : block_size = (N + 1) * N
                !periodic[1] && j == md ? bd = 1 : bd = 0
                field_matrix[range_y, range_x] = transpose(
                    reshape(view(field, (offset+1):(offset+block_size)), N + bd, N + 1),
                )
                offset += block_size
            end
        end
        if periodic[1]
            field_matrix[:, end] = view(field_matrix, :, 1)
        end
    elseif cont_y
        for i = 1:md
            offset = (i - 1) * md * N * (N + 1)
            block_size = N * (N + 1)
            range_y = ((i-1)*N+1):(i*N)
            for j = 1:md
                range_x = ((j-1)*(N+1)+1):(j*(N+1))
                field_matrix[range_y, range_x] = transpose(
                    reshape(view(field, (offset+1):(offset+block_size)), N + 1, N),
                )
                offset += block_size
            end
        end
        if !periodic[2]
            offset = (md - 1) * md * N * (N + 1)
            block_size = (N + 1)^2
            range_y = ((md-1)*N+1):(md*N+1)
            for j = 1:md
                range_x = ((j-1)*(N+1)+1):(j*(N+1))
                field_matrix[range_y, range_x] = transpose(
                    reshape(view(field, (offset+1):(offset+block_size)), N + 1, N + 1),
                )
                offset += block_size
            end
        else
            field_matrix[end, :] = view(field_matrix, 1, :)
        end
    else
        for i = 1:md
            offset = (i - 1) * md * (N + 1)^2
            for j = 1:md
                range_x = ((j-1)*(N+1)+1):(j*(N+1))
                range_y = ((i-1)*(N+1)+1):(i*(N+1))
                field_matrix[range_y, range_x] = transpose(
                    reshape(view(field, (offset+1):(offset+(N+1)^2)), N + 1, N + 1),
                )
                offset += (N + 1)^2
            end
        end
    end
    return field_matrix
end

end
