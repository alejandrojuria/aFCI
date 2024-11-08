module RydbergED

# Alejandro Jose Uria-Alvarez 2023
# Module to implement an exact diagonalization calculation of Rydberg's model to show the existence
# of a fractional Chern insulator with hardcore bosons.

# We follow column-major convention of Julia to store any vector quantity in matrix for faster access (higher cache hit ratio)

export main

import Random
using Distributions: Uniform
using LinearAlgebra: norm, dot, eigvals
using Plots
using SparseArrays
using Base.Threads
using KrylovKit: eigsolve
using LaTeXStrings
using ArgParse
using DelimitedFiles


# --------------------------------------------------------------------------------------------
# ------------------------------------- Crystal routines -------------------------------------
# --------------------------------------------------------------------------------------------

"""
    Crystal{T<:Number}
    Crystal(bravais_lattice{T}::Matrix, motif::Matrix{T}) where {T<:Number}

Crystal struct containing the Bravais lattice and the motif of the system.
"""
struct Crystal{T<:Number}

    bravais_lattice::Matrix{T}
    motif::Matrix{T}
    dim::Int
    natoms::Int
    function Crystal(bravais_lattice::Matrix{T}, motif::Matrix{T}) where {T<:Real}
        if size(bravais_lattice, 1) != 3
            error("Bravais lattice must have three dimensional vectors.")
        end
        if size(motif, 1) != 3
            error("Motif must be composed of three dimensional vectors.")
        end
        dim = size(bravais_lattice, 2)
        natoms = size(motif, 2)
        new{T}(bravais_lattice, motif, dim, natoms)

    end
end

"""
    supercell(unit_cell::Crystal, n::Int...) -> Crystal

Creates a supercell consisting in a unit cell that is repeated n[i] times along each
Bravais vector.
"""
function supercell(unit_cell::Crystal, n::Int...)

    if length(n) > unit_cell.dim
        error("Number of axis given exceeds that of the unit cell. Given: $(length(n)), max. expected $(unit_cell.dim).")
    end

    sc_motif = copy(unit_cell.motif)
    sc_bravais_lattice = copy(unit_cell.bravais_lattice)

    for (i, cells) in enumerate(n)
        reference_motif = sc_motif

        for j in range(1, stop=cells - 1, step=1)
            displaced_motif = reference_motif .+ unit_cell.bravais_lattice[:, i] * j
            sc_motif = hcat(sc_motif, displaced_motif)
        end

        sc_bravais_lattice[:, i] *= cells
    end

    return Crystal(sc_bravais_lattice, sc_motif)
end

"""
    amorphize(crystal::Crystal, disorder::Number) -> Crystal

Displaces the atoms of a crystal randomly sampling from a uniform distribution U(0, disorder).
"""
function amorphize(crystal::Crystal, disorder::Number; planar::Bool=true)

    if disorder == 0
        return crystal
    end
    
    r = rand(Uniform(0, disorder), size(crystal.motif, 2))
    θ = rand(Uniform(0, 2*π), size(crystal.motif, 2))
    if planar
        displacements = r .* hcat(cos.(θ), sin.(θ), zeros(size(crystal.motif, 2)))
    else
        φ = rand(Uniform(0, π), size(crystal.motif, 2))
        displacements = r .* hcat(sin.(θ) .* cos.(φ), sin.(θ) .* cos.(φ), cos.(φ))
    end

    return Crystal(crystal.bravais_lattice, crystal.motif .+ displacements')

end

"""
    _generate_combinations(n::Int) -> Vector{Int}

Generates a vector storing all n-dimensional combinations of a vector of values.
"""
function _generate_combinations(values::Vector{Int}, n::Int, combinations=[Vector{Int}()])

    if n == 0
        return combinations
    end

    new_combinations = Vector{Vector{Int}}()
    for combination in combinations
        for value in values
            new_combination = copy(combination)
            push!(new_combination, value)
            push!(new_combinations, new_combination)
        end
    end
    
    return _generate_combinations(values, n - 1, new_combinations)
end

"""
    _generate_unit_cells(crystal::Crystal, boundary::String="PBC")

Generates a matrix where each column corresponds to the Bravais vector for one unit cell.
Note that this method does not check if the boundary is correct.
"""
function _generate_unit_cells(crystal::Crystal, boundary::String="PBC")

    near_unit_cells = []
    if boundary == "PBC"
        combinations = _generate_combinations([-1, 0, 1], crystal.dim)
        cells = sum(combinations[1]' .* crystal.bravais_lattice, dims=2)
        for combination in combinations[2:end]
            vector = sum(combination' .* crystal.bravais_lattice, dims=2)
            cells = hcat(cells, vector)
        end
        near_unit_cells = cells
    else
        near_unit_cells = hcat([0., 0., 0.])
    end

    return near_unit_cells
end

"""
    identify_bonds(crystal::Crystal, cutoff:Number) -> Vector

Identifies all the bonds between the atoms of the system according to some cutoff length
and the boundary conditions used.
"""
function identify_bonds(crystal::Crystal, cutoff::Number; boundary::String="PBC")

    if boundary ∉ ["OBC", "PBC"]
        error("Boundary conditions must be either 'PBC' or 'OBC'.")
    end
    if cutoff < 0
        error("Cutoff must be a positive number.")
    end

    near_unit_cells = _generate_unit_cells(crystal, boundary) 
    combinations = _generate_combinations([-1, 0, 1], crystal.dim)

    eps = 1e-4
    bonds = []
    atoms = copy(crystal.motif)
    for (n, atom) in enumerate(eachcol(crystal.motif))
        cnt = 0
        for (i, unit_cell) in enumerate(eachcol(near_unit_cells))
            distances = norm.(eachcol(atoms .+ unit_cell .- atom))
            indices = findall(distances .< cutoff + eps)
            for index in indices
                if index == n
                    continue
                end
                bond = [n, index, combinations[i]]
                push!(bonds, bond)
                cnt += 1
            end
        end
        display("Number of neighbours of atom $(n): $(cnt)")
    end

    return bonds

end

"""
    plot_crystal(crystal::Crystal, bonds) -> nothing

Visualizes the crystal. If the bonds are given, they are also drawn.
"""
function plot_crystal(crystal::Crystal, bonds=nothing)

    scatter(xlabel=L"x", ylabel=L"y")
    scatter!(crystal.motif[1, :], crystal.motif[2, :])
    colors = Dict(1 => "black", 2 => "red", 3 => "blue")

    for bond in bonds
        i, j, cell = bond
        cell = vec(sum(cell' .* crystal.bravais_lattice, dims=2))
        
        plot!([crystal.motif[1, i], crystal.motif[1, j] + cell[1]], [crystal.motif[2, i], crystal.motif[2, j] + cell[2]], label="")
    end
end

"""
    clusterN4() -> Crystal

Generates a cluster for filling N=4 with hardcore bosons in a specific configuration.
"""
function clusterN4()

    motif = hcat([0., 0., 0.])

    a1 = [3/2., sqrt(3)/2., 0.]
    a2 = [3/2., -sqrt(3)/2., 0.]
    tau = [3/2., sqrt(3)/2., 0.] - [1., 0., 0.]

    a1p = 3 * a1 - a2
    a2p = 3 * a2 - a1

    bravais_lattice = [a1p a2p]
    motif = [0., 0., 0]
    motif = hcat(motif, tau)
    motif = hcat(motif, a1)
    motif = hcat(motif, a1 + tau)
    motif = hcat(motif, 2 * a1)
    motif = hcat(motif, a1 + a2 + tau)
    motif = hcat(motif, 2 * a1 + a2)
    motif = hcat(motif, a2 + tau)
    motif = hcat(motif, a2 + a1)
    motif = hcat(motif, a2 - a1 + tau)
    motif = hcat(motif, a2)
    motif = hcat(motif, 2 * a2 - a1 + tau)
    motif = hcat(motif, 2 * a2)
    motif = hcat(motif, 2 * a2 + tau)
    motif = hcat(motif, 2 * a2 + a1)
    motif = hcat(motif, 2 * a2 + a1 + tau)

    honeycomb = Crystal(bravais_lattice, motif)

    return honeycomb
end



"""
    clusterN5() -> Crystal

Generates a cluster for filling N=5 with hardcore bosons in a specific configuration.
"""
function clusterN5()

    motif = hcat([0., 0., 0.])

    a1 = [3/2., sqrt(3)/2., 0.]
    a2 = [3/2., -sqrt(3)/2., 0.]
    tau = [3/2., sqrt(3)/2., 0.] - [1., 0., 0.]

    a1p = 3 * a1 + a2
    a2p = 3 * a2 - a1

    bravais_lattice = [a1p a2p]
    motif = [0., 0., 0]
    motif = hcat(motif, tau)
    motif = hcat(motif, a1)
    motif = hcat(motif, a1 + tau)
    motif = hcat(motif, 2 * a1)
    motif = hcat(motif, a1 + a2 + tau)
    motif = hcat(motif, 2 * a1 + a2)
    motif = hcat(motif, a2 + tau)
    motif = hcat(motif, a2 + a1)
    motif = hcat(motif, a2 - a1 + tau)
    motif = hcat(motif, a2)
    motif = hcat(motif, 2 * a2 - a1 + tau)
    motif = hcat(motif, 2 * a2)
    motif = hcat(motif, 2 * a2 + tau)
    motif = hcat(motif, 2 * a2 + a1)
    motif = hcat(motif, 2 * a2 + a1 + tau)

    motif = motif .+ a2 
    motif = hcat(motif, [0., 0., 0.])
    motif = hcat(motif, a2 - a1 + tau)
    motif = hcat(motif, 3 * a2 + 2 * a1)
    motif = hcat(motif, 4 * a2 + a1 + tau)

    honeycomb = Crystal(bravais_lattice, motif)

    return honeycomb
end

# --------------------------------------------------------------------------------------------
# ------------------------------------ Single-particle model ---------------------------------
# --------------------------------------------------------------------------------------------

"""
    RydbergModel(crystal::Crystal; ta::Number=1, tb::Number=1, Δ::Number=0, ms::Number=0, ω::Number=1.0, W::Number=0, cutoff::Number=1.3, boundary::String="PBC", flux::Vector{Number}) -> Matrix{Complex}

The Hamiltonian of the system can be built with either PBC or OBC; in the PBC case it is intended to
be used with k=0 exclusively, so there is no k dependence.
"""
function RydbergModel(
    crystal::Crystal; 
    ta::Number=1.0, 
    tb::Number=1.0,
    Δ::Number=0.0,
    ms::Number=0.0,
    ω::Number=1.0, 
    W::Number=0.0,
    cutoff::Number=1.3, 
    boundary::String="PBC", 
    flux::Vector{<:Number}=[0., 0.]
)

    bonds = identify_bonds(crystal, cutoff; boundary=boundary)
    basisdim = 2 * crystal.natoms

    hamiltonian = zeros(Complex, basisdim, basisdim)
    onsite_energy = zeros(Complex, 2, 2)
    onsite_energy[1, 1] = Δ

    for n in range(1, stop=crystal.natoms, step=1)
        hamiltonian[2 * n - 1:2 * n, 2 * n - 1:2 * n] = onsite_energy
    end

    anderson_term = _anderson_disorder(crystal, W)
    staggered_mass = _staggered_mass(crystal, ms)
    hamiltonian += anderson_term + staggered_mass

    for bond in bonds
        j, i, cell_coefs = bond
        cell_coefs = convert(Array{Number}, cell_coefs)
        cell = vec(sum(cell_coefs' .* crystal.bravais_lattice, dims=2))
        initial_position, final_position = crystal.motif[:, j], crystal.motif[:, i] + cell

        t_flux = _add_flux(1, cell_coefs, crystal, flux)
        hamiltonian[2 * i - 1:2 * i, 2 * j - 1: 2 * j] += _hopping_rydberg(
                                                                        initial_position, 
                                                                        final_position, 
                                                                        ta, 
                                                                        tb,
                                                                        ω,
                                                                        t_flux
                                                                        )
    end

    return hamiltonian, bonds

end

"""
    RydbergModel_flux_per_orbital(crystal::Crystal; ta::Number=1, tb::Number=1, Δ::Number=0, ms::Number=0, ω::Number=1.0, W::Number=0, cutoff::Number=1.3, boundary::String="PBC", flux::Vector{Number}) -> Matrix{Complex}

The Hamiltonian of the system can be built with either PBC or OBC; in the PBC case it is intended to
be used with k=0 exclusively, so there is no k dependence. Based on the Rydberg model, but now the flux is applied only to some of the orbitals.
"""
function RydbergModel_flux_per_orbital(
    crystal::Crystal; 
    ta::Number=1.0, 
    tb::Number=1.0,
    Δ::Number=0.0,
    ms::Number=0.0,
    ω::Number=1.0, 
    W::Number=0.0,
    cutoff::Number=1.3, 
    boundary::String="PBC", 
    flux::Vector{<:Number}=[0., 0.],
    orbitals::Vector{Int}=[1, 1]
)

    display("Specified orbitals: $(orbitals)")

    bonds = identify_bonds(crystal, cutoff; boundary=boundary)
    basisdim = 2 * crystal.natoms

    hamiltonian = zeros(Complex, basisdim, basisdim)
    onsite_energy = zeros(Complex, 2, 2)
    onsite_energy[1, 1] = Δ

    for n in range(1, stop=crystal.natoms, step=1)
        hamiltonian[2 * n - 1:2 * n, 2 * n - 1:2 * n] = onsite_energy
    end

    anderson_term = _anderson_disorder(crystal, W)
    staggered_mass = _staggered_mass(crystal, ms)
    hamiltonian += anderson_term + staggered_mass

    for bond in bonds
        j, i, cell_coefs = bond
        cell_coefs = convert(Array{Number}, cell_coefs)
        cell = vec(sum(cell_coefs' .* crystal.bravais_lattice, dims=2))
        initial_position, final_position = crystal.motif[:, j], crystal.motif[:, i] + cell

        exp_flux = _add_flux_vector(1, cell_coefs, crystal, flux)
        hamiltonian[2 * i - 1:2 * i, 2 * j - 1: 2 * j] += _hopping_rydberg_flux_per_orbital(
                                                                        initial_position, 
                                                                        final_position, 
                                                                        ta, 
                                                                        tb,
                                                                        ω,
                                                                        exp_flux,
                                                                        orbitals
                                                                        )
    end

    return hamiltonian, bonds

end


"""
    _hopping_rydberg(initial_position, final_position, ta, tb, ω) -> Matrix{Complex}

Private method to create the hopping matrix between two atoms of the solid. Note that the
lattice parameter a is hardcoded, a=1.
"""
function _hopping_rydberg(
    initial_position::Vector{T}, 
    final_position::Vector{T}, 
    ta::Number,
    tb::Number,
    ω::Number,
    flux::Number
) where {T<:Real} 

    x, y, z = final_position - initial_position
    r = norm(final_position - initial_position)
    θ = atan(y, x)

    hopping = zeros(Complex, 2, 2)
    hopping[1, 1] = -ta 
    hopping[2, 2] = -tb
    hopping[1, 2] = ω * exp(-2 * im * θ)
    hopping[2, 1] = ω * exp(2 * im * θ)
    
    hopping *= flux/(r^3)

    return hopping
end

"""
    _hopping_rydberg(initial_position, final_position, ta, tb, ω) -> Matrix{Complex}

Private method to create the hopping matrix between two atoms of the solid. Note that the
lattice parameter a is hardcoded, a=1. This version of the hopping matrix allows for a flux to be applied only to one orbital.
"""
function _hopping_rydberg_flux_per_orbital(
    initial_position::Vector{T}, 
    final_position::Vector{T}, 
    ta::Number,
    tb::Number,
    ω::Number,
    flux::Vector{Complex},
    orbitals::Vector{Int}
) where {T<:Real} 

    x, y, z = final_position - initial_position
    r = norm(final_position - initial_position)
    θ = atan(y, x)

    hopping = zeros(Complex, 2, 2)
    hopping[1, 1] = -ta 
    hopping[2, 2] = -tb
    hopping[1, 2] = ω * exp(-2 * im * θ)
    hopping[2, 1] = ω * exp(2 * im * θ)
    
    hopping *= 1/(r^3)

    hopping[orbitals[1], orbitals[1]] *= flux[1]
    hopping[orbitals[2], orbitals[2]] *= flux[2]

    return hopping
end


"""
    _anderson_disorder(crystal::Crystal, disorder::Number) -> Matrix{Complex}

Private method to generate the Anderson disorder matrix for the system.
"""
function _anderson_disorder(
    crystal::Crystal, 
    W::Number
)

    if W == 0
        return zeros(Complex, 2 * crystal.natoms, 2 * crystal.natoms)
    end

    disorder_matrix = zeros(Complex, 2 * crystal.natoms, 2 * crystal.natoms)
    for n in range(1, stop=crystal.natoms, step=1)
        disorder = rand(Uniform(-W/2, W/2))
        display(disorder)
        disorder_matrix[2 * n - 1, 2 * n - 1] = disorder
        disorder_matrix[2 * n, 2 * n] = disorder
    end

    return disorder_matrix
end


"""
    _staggered_mass(crystal::Crystal, M::Number) -> Matrix{Complex}

Private method to generate the staggered mass matrix for the system. Note that this routine
    assumes that the atoms of the motif are always alternating the sublattice (i.e. ABAB...)
"""
function _staggered_mass(
    crystal::Crystal, 
    M::Number
)

    mass_matrix = zeros(Complex, 2 * crystal.natoms, 2 * crystal.natoms)
    for n in range(1, stop=crystal.natoms, step=1)
        mass_matrix[2 * n - 1, 2 * n - 1] = M * (-1)^(n - 1)
        mass_matrix[2 * n, 2 * n] = M * (-1)^(n - 1)
    end

    return mass_matrix
end



"""
    _add_flux(f::Number, cell:Vector{Number}, flux::Number...)

Adds a flux to the hoppings between different unit cells.
"""
function _add_flux(
    t::Number, 
    cell::Vector{<:Number}, 
    crystal::Crystal, 
    flux::Vector{<:Number}
)

    if length(flux) > 2
        error("Flux must be a two-dimensional vector, [phi_x, phi_y]")
    end

    total_flux = 0
    for i=1:2
        if abs(cell[i]) < 1E-7
            continue
        end
        if cell[i] > 0
            total_flux += flux[i]
        else
            total_flux -= flux[i]
        end
    end
    
    new_hopping = t * exp(im * 2 * π * total_flux)
    
    return new_hopping
end

"""
    _add_flux(f::Number, cell:Vector{Number}, flux::Number...)

Adds a flux to the hoppings between different unit cells.
"""
function _add_flux_vector(
    t::Number, 
    cell::Vector{<:Number}, 
    crystal::Crystal, 
    flux::Vector{<:Number}
)

    if length(flux) > 2
        error("Flux must be a two-dimensional vector, [phi_x, phi_y]")
    end

    total_flux = ones(Complex, 2) 
    for i=1:2
        if abs(cell[i]) < 1E-7
            continue
        end
        if cell[i] > 0
            total_flux[i] *= exp( im * 2 * π * flux[i])
        else
            total_flux[i] *= exp(-im * 2 * π * flux[i])
        end
    end

    display(total_flux)
        
    return total_flux
end



# --------------------------------------------------------------------------------------------
# -------------------------------------- Many-body model -------------------------------------
# --------------------------------------------------------------------------------------------

"""
    particle_hilbert_space!(N::Int, n::int, stateList::Vector{Int}, state::Int=0) -> nothing

Generates all the states of the Fock space with `n` electrons out of `N` orbitals. 
The states are stored in integer representation in `stateList`. By default, the function
starts looking for states from 0, but any `state` can be used as the starting point.
"""
function particle_hilbert_space!(N::Int, n::Int, state_list::Vector{Int}, state::Int=0)

    if n == 0
        push!(state_list, state)
        return nothing
    elseif N < n
        return nothing
    end

    particle_hilbert_space!(N - 1, n - 1, state_list, 1 << (N - 1) | state)
    particle_hilbert_space!(N - 1, n, state_list, state)

    return nothing
end


"""
    restricted_particle_hilbert_space!(N::Int, n::Int, state_list::Vector{Int}, state::Int=0) -> nothing

Routine to generate the Hilbert space of the many-body problem such that each atom can only
hold one electron, either of one type of orbital or the other.
"""
function restricted_particle_hilbert_space!(N::Int, n::Int, state_list::Vector{Int}, state::Int=0)

    if n == 0
        push!(state_list, state)
        return nothing
    elseif N < n
        return nothing
    end

    restricted_particle_hilbert_space!(N - 2, n - 1, state_list, 1 << (N - 1) | state)
    restricted_particle_hilbert_space!(N - 2, n - 1, state_list, 1 << (N - 2) | state)
    restricted_particle_hilbert_space!(N - 2, n, state_list, state)

    return nothing

end


"""
    find_state_index(state::Int, state_list::Vector{Int}) -> Int

Returns the index of a given many-body state in the basis.
"""
function find_state_index(state::Int, state_list::Vector{Int})

    index = findfirst(isequal(state), state_list)
    return index

end

"""
    hopping_hamiltonian(site_index::Int, state:Int, h::SparseMatrixCSC{Complex}) -> (Vector{Complex}, Vector{Int})

Acts on a given state with the single-particle hamiltonian on the specified position to give all
the resulting states plus the corresponding matrix elements.
"""
function hopping_hamiltonian(site_index::Int, state::Int, h::SparseMatrixCSC{Complex, Int}, state_list::Vector{Int})

    if (state >> (site_index - 1) & 1) == 0
        return Complex[], Int[]
    end
    indices = Int[]
    elements = Complex[]
    for i=h.colptr[site_index]:(h.colptr[site_index + 1] - 1)
        row_index = h.rowval[i]
        marker = state >> (row_index - 1) & 1
        if marker == 1
            continue
        end
        new_state = state ⊻ (1 << (site_index - 1)) ⊻ (1 << (row_index - 1))

        state_index = find_state_index(new_state, state_list)
        if state_index === nothing
            continue
        end

        push!(indices, state_index)
        push!(elements, h.nzval[i])
    end

    return elements, indices
end

"""
    onsite_hamiltonian(state::Int, h::SparseMatrixCSC{Complex, Int}) -> Complex

Computes the many-body matrix element corresponding to the onsite energy.
"""
function onsite_hamiltonian(state::Int, onsite_energies::Vector)

    energy = 0im
    for site=1:length(onsite_energies)
        if (state >> (site - 1) & 1) == 0
            continue
        end
        energy += onsite_energies[site]
    end

    return energy
end


"""
    hubbard_hamiltonian(state::Int, strength::Float, natoms::Int) -> Complex

Computes the energy of a many-body state due to on-site Hubbard repulsion (interaction between
the two orbitals present at each atom).
"""
function hubbard_hamiltonian(state::Int, strength::Float64, natoms::Int)

    energy = 0im
    for site=1:natoms
        marker = (state >> (2*site - 1)) & (state >> (2*site - 2)) & 1
        if marker == 1
            energy += strength
        end
    end

    return energy
end


"""
    many_body_hamiltonian_threaded(state_list::Vector{Int}, h::SparseMatrixCSC{Complex, Int}, V::Float=0) -> SparseMatrixCSC{Complex, Int}

Threaded constrution the many-body representation of the Agarwala Hamiltonian with hardcore bosons.
Parallelization of sparse hamiltonian based on:
https://discourse.julialang.org/t/multithreading-on-entries-sparse-matrix/70023/4
Stores the Hamiltonian in a sparse matrix.
"""
function many_body_hamiltonian_threaded(state_list::Vector{Int}, h::SparseMatrixCSC{Complex, Int}, V::Float64=0.0)

    onsite_energies = [h[i, i] for i in 1:h.n]

    n = length(state_list)
    nchunks::Int = nthreads()
    if nchunks > n
        nchunks = 1
    end
    chunk_size = n ÷ nchunks
    j_chunks = [(k - 1) * chunk_size + 1:k * chunk_size for k in 1:nchunks]
    j_chunks[end] = (nchunks - 1) * chunk_size + 1:n

    _colptrs = [Int[] for _ in 1:nchunks]
    push!(_colptrs[1], 1)
    _rowvals = [Int[] for _ in 1:nchunks]
    _nzvals  = [ComplexF64[] for _ in 1:nchunks]

    @threads for k in 1:nchunks
        cnt = (k == 1) ? 1 : 0
        for j in j_chunks[k]
            state = state_list[j]
            col_indices = Int[]
            col_values  = Complex[]

            onsite = onsite_hamiltonian(state, onsite_energies)
            # onsite += hubbard_hamiltonian(state, V, h.n ÷ 2)
            cnt += 1
            push!(col_indices, j)
            push!(col_values, onsite)

            for site in 1:size(h, 1)
                (elements, indices) = hopping_hamiltonian(site, state, h, state_list)
                cnt += length(indices)
                append!(col_indices, indices)
                append!(col_values, elements)
            end
            push!(_colptrs[k], cnt)

            sorted_indices = sortperm(col_indices)
            append!(_rowvals[k], col_indices[sorted_indices])
            append!(_nzvals[k], col_values[sorted_indices])
        end
    end
    for k in 2:nchunks
        _colptrs[k] .+= _colptrs[k - 1][end]
    end

    colptr = vcat(_colptrs...)
    rowval = vcat(_rowvals...)
    nzval  = vcat(_nzvals...)

    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end


"""
    many_body_hamiltonian_sparse(state_list::Vector{Int}, h::SparseMatrixCSC{Complex, Int}) -> SparseMatrixCSC{Complex, Int}

Serial construction of the many-body hamiltonian for the Agarwala model with hardcore bosons.
Stores the Hamiltonian in a sparse matrix.
"""
function many_body_hamiltonian_sparse(state_list::Vector{Int}, h::SparseMatrixCSC{Complex, Int})

    onsite_energies = [h[i, i] for i in 1:h.n]

    hamiltonian = spzeros(Complex, length(state_list), length(state_list))
    for j=1:length(state_list)
        state = state_list[j]
        for site in 1:size(h, 1)
            (elements, indices) = hopping_hamiltonian(site, state, h, state_list)
            hamiltonian[indices, j] = elements
        end
        onsite = onsite_hamiltonian(state, onsite_energies)
        hamiltonian[j, j] = onsite
    end

    return hamiltonian
end

"""
    many_body_hamiltonian_dense(state_list::Vector{Int}, h::SparseMatrixCSC{Complex, Int}) -> Matrix{Complex}

Serial construction of the many-body hamiltonian for the Agarwala model with hardcore bosons.
Stores the Hamiltonian in a dense matrix.
"""
function many_body_hamiltonian_dense(state_list::Vector{Int}, h::SparseMatrixCSC{Complex, Int})

    onsite_energies = [h[i, i] for i in 1:h.n]

    hamiltonian = zeros(Complex, length(state_list), length(state_list))
    for j=1:length(state_list)
        state = state_list[j]
        for site in 1:size(h, 1)
            (elements, indices) = hopping_hamiltonian(site, state, h, state_list)
            hamiltonian[indices, j] = elements
        end
        onsite = onsite_hamiltonian(state, onsite_energies)
        hamiltonian[j, j] = onsite
    end

    return hamiltonian
end



# --------------------------------------------------------------------------------------------
# ------------------------------------------- Main -------------------------------------------
# --------------------------------------------------------------------------------------------

function run()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "Number of bosons"
            arg_type = Int
            default = 1
        "--Nx"
            help = "Number of cells in x direction"
            arg_type = Int
            default = 2
        "--Ny"
            help = "Number of cells in x direction"
            arg_type = Int
            default = 2
        "--neigvals"
            help = "Number of eigenvalues"
            arg_type = Int
            default = 10
        "--M"
            help = "Mass value"
            arg_type = Float64
            default = -1.0
        "--t2"
            help = "t2 value"
            arg_type = Float64
            default = 0.0
        "--Rc"
            help = "Cutoff value"
            arg_type = Float64
            default = 2.1
        "--disorder"
            help = "Value of disorder"
            arg_type = Float64
            default = 0.0
        "--V"
            help = "Hubbard value"
            arg_type = Float64
            default = 0.0
        "--npoints"
            help = "Number of flux points"
            arg_type = Int
            default = 11

        
    end

    args = parse_args(s)
    
    # Assign parsed arguments
    Nx = args["Nx"]
    Ny = args["Ny"]
    N  = args["N"]
    M  = args["M"]
    t2 = args["t2"]
    Rc = args["Rc"]
    disorder = args["disorder"]
    neigvals = args["neigvals"]
    V = args["V"]
    flux_points = args["npoints"]

    norbitals = 2 * Nx * Ny
    which_eigval::Symbol = :SR

    motif = hcat([0., 0., 0.])

    square = Crystal([[1., 0., 0.] [0., 1., 0.]], motif)
    sc = amorphize(supercell(square, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    # flux_values = [0.0]
    energy_pumping = zeros(Float64, neigvals, flux_points)

    for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

        h, bonds = AgarwalaModel(sc, M=M, boundary="PBC", flux=[ϕ, 0.0], t2=t2, cutoff=Rc)
        eigval = real(eigvals(h))
        h = sparse(h)

        # display("Threaded: ")
        # @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        # @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        # display(eigval[1:neigvals])        

        display("Dense: ")
        @time hamiltonian = many_body_hamiltonian_dense(hilbert_space, h)
        @time eigval = real(eigvals(hamiltonian))
        display(eigval)

        # display("------------------\nSparse: ")
        # @time hamiltonian = many_body_hamiltonian_sparse(hilbert_space, h)
        # display(hamiltonian)

        energy_pumping[:, j] = eigval[1:neigvals]

        hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
        scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    savefig("agarwala_flux_N$(N)_Nx$(Nx)_Ny$(Ny)_M$(M)_t2$(t2)_disorder$(disorder)_Rc$(Rc)_V$(V)_npoints$(flux_points).png")
    writedlm("agarwala_flux_N$(N)_Nx$(Nx)_Ny$(Ny)_M$(M)_t2$(t2)_disorder$(disorder)_Rc$(Rc)_V$(V)_npoints$(flux_points).csv",  energy_pumping, ',')

end

# Run upon execution of the script
# run()

function run_restricted()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "Number of bosons"
            arg_type = Int
            default = 4
        "--Nx"
            help = "Number of cells in x direction"
            arg_type = Int
            default = 4
        "--Ny"
            help = "Number of cells in x direction"
            arg_type = Int
            default = 4
        "--neigvals"
            help = "Number of eigenvalues"
            arg_type = Int
            default = 10
        "--Rc"
            help = "Cutoff value"
            arg_type = Float64
            default = 2.1
        "--disorder"
            help = "Value of disorder"
            arg_type = Float64
            default = 0.0
        "--npoints"
            help = "Number of flux points"
            arg_type = Int
            default = 9
        "--flux"
            help = "Flux value; overrides multiple npoints"
            arg_type = Float64
            default = -1.0
        
    end

    args = parse_args(s)
    
    # Assign parsed arguments
    Nx = args["Nx"]
    Ny = args["Ny"]
    N  = args["N"]
    Rc = args["Rc"]
    disorder = args["disorder"]
    neigvals = args["neigvals"]
    flux_points = args["npoints"]
    flux = args["flux"]

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    norbitals = 4 * Nx * Ny 
    which_eigval::Symbol = :SR

    motif = hcat([0., 0., 0.])

    square = Crystal([[1., 0., 0.] [0., 1., 0.]], motif)
    honeycomb = Crystal([[3/2., sqrt(3)/2., 0.] [3/2., -sqrt(3)/2., 0.]], [[0., 0., 0.] [1., 0., 0.]])

    sc = amorphize(supercell(honeycomb, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    if flux >= 0
        flux_values = [flux]
    end
    energy_pumping = zeros(Float64, neigvals, flux_points)

    for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc)
        eigval = real(eigvals(h))
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])        

        energy_pumping[:, j] = eigval[1:neigvals]

        hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
        scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    gui()

    savefig("rydberg_flux_N$(N)_Nx$(Nx)_Ny$(Ny)_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc).png")
    writedlm("rydberg_flux_N$(N)_Nx$(Nx)_Ny$(Ny)_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc).csv", energy_pumping, ',')

end

function run_clusterN4_restricted()

    Random.seed!(1234)

    Rc = 2.1
    disorder = 0.0
    neigvals = 10
    flux_points = 11
    W = 0
    m_s = 4

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    N = 4

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    honeycomb = clusterN4()
    sc = amorphize(honeycomb, disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    # flux_values = [0.0]
    energy_pumping = zeros(Float64, neigvals, flux_points)

    # Init anderson term outside loop
    anderson_term = _anderson_disorder(sc, W)

    for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc, W=0, ms=m_s)
        h += anderson_term

        eigval = real(eigvals(h))
        display(eigval)
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])        

        energy_pumping[:, j] = eigval[1:neigvals]

        hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
        scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    # gui()

    savefig("rydberg_cluster_flux_N$(N)_npoints$(flux_points)_disorder$(disorder)_delta$(Δ)_W$(W)_ms$(m_s).png")
    writedlm("rydberg_cluster_flux_N$(N)_npoints$(flux_points)_disorder$(disorder)_delta$(Δ)_W$(W)_ms$(m_s).csv", energy_pumping, ',')

end


function run_clusterN4_restricted_vs_rcutoff()
    
    Rc = 2.1
    disorder = 0.0
    neigvals = 10
    flux_points = 9

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    Rc = 2.1
    V = 0.0

    N = 4

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    motif = hcat([0., 0., 0.])

    a1 = [3/2., sqrt(3)/2., 0.]
    a2 = [3/2., -sqrt(3)/2., 0.]
    tau = [3/2., sqrt(3)/2., 0.] - [1., 0., 0.]

    a1p = 3 * a1 - a2
    a2p = 3 * a2 - a1

    bravais_lattice = [a1p a2p]
    motif = [0., 0., 0]
    motif = hcat(motif, tau)
    motif = hcat(motif, a1)
    motif = hcat(motif, a1 + tau)
    motif = hcat(motif, 2 * a1)
    motif = hcat(motif, a1 + a2 + tau)
    motif = hcat(motif, 2 * a1 + a2)
    motif = hcat(motif, a2 + tau)
    motif = hcat(motif, a2 + a1)
    motif = hcat(motif, a2 - a1 + tau)
    motif = hcat(motif, a2)
    motif = hcat(motif, 2 * a2 - a1 + tau)
    motif = hcat(motif, 2 * a2)
    motif = hcat(motif, 2 * a2 + tau)
    motif = hcat(motif, 2 * a2 + a1)
    motif = hcat(motif, 2 * a2 + a1 + tau)

    display(motif)
    display(bravais_lattice)

    honeycomb = Crystal(bravais_lattice, motif)

    sc = amorphize(honeycomb, disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    cutoff_values = [1.1, 1.8, 2.1]
    # flux_values = [0.0]
    energy_pumping = zeros(Float64, neigvals, flux_points)

    for(i, Rc) in enumerate(cutoff_values)
        for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

            h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc)
            plot_crystal(sc, bonds)
            eigval = real(eigvals(h))
            h = sparse(h)

            display("Threaded: ")
            @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
            @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
            display(eigval[1:neigvals])        

            energy_pumping[:, j] = eigval[1:neigvals]

            hamiltonian = nothing # Delete to avoid memory leak
        end

        scatter()
        for energies in eachcol(energy_pumping')
            scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
            scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
        end
        scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

        savefig("rydberg_cluster_flux_N$(N)_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc).png")
        writedlm("rydberg_cluster_flux_N$(N)_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc).csv", energy_pumping, ',')
    end    

end


function run_clusterN5_restricted()
    
    disorder = 0.0
    neigvals = 10
    flux_points = 9

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    Rc = 2.1
    V = 0.0

    N = 5

    norbitals = 2 * 20
    which_eigval::Symbol = :SR

    honeycomb = clusterN5()
    sc = amorphize(honeycomb, disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    energy_pumping = zeros(Float64, neigvals, flux_points)

    for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc)
        eigval = real(eigvals(h))
        h = sparse(h)
        
        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])        

        energy_pumping[:, j] = eigval[1:neigvals]

        hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
        scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    gui()

    savefig("rydberg_cluster_flux_N$(N)_npoints$(flux_points)_disorder$(disorder).png")
    writedlm("rydberg_cluster_flux_N$(N)_npoints$(flux_points)_disorder$(disorder).csv", energy_pumping, ',')

end

function run_clusterN6_restricted()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--flux"
            help = "Flux value; overrides multiple npoints"
            arg_type = Float64
            default = -1.0

    end

    args = parse_args(s)
    
    # Assign parsed arguments
    flux = args["flux"]
    
    disorder = 0.0

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    Rc = 2.1
    V = 0.0

    N = 6
    Nx = 4
    Ny = 3
    neigvals = 10

    norbitals = 2 * 24
    which_eigval::Symbol = :SR

    honeycomb = Crystal([[3/2., sqrt(3)/2., 0.] [3/2., -sqrt(3)/2., 0.]], [[0., 0., 0.] [1., 0., 0.]])
    sc = amorphize(supercell(honeycomb, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    flux_values = [flux]
    energy_pumping = zeros(Float64, neigvals, 1)

    for (j, ϕ) in enumerate(flux_values)

        # display("------------------------------------")
        # display("Flux insertion: $(ϕ)")

        # h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc)
        # eigval = real(eigvals(h))
        # h = sparse(h)
        
        # display("Threaded: ")
        # @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        # @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        # display(eigval[1:neigvals])        

        # energy_pumping[:, j] = eigval[1:neigvals]

        # hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
        scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    gui()

    savefig("rydberg_cluster_N$(N)_disorder$(disorder)_flux$(flux).png")
    writedlm("rydberg_cluster_N$(N)_disorder$(disorder)_flux$(flux).csv", energy_pumping, ',')
end

function run_square_restricted(N, Nx, Ny)
    
    Rc = 2.1
    disorder = 0.0
    neigvals = 10
    flux_points = 11

    ta = -0.3
    tb = 0.7
    Δ = -10
    ω = 1.2
    Rc = 2.1
    V = 0.0

    norbitals = 2 * Nx * Ny
    which_eigval::Symbol = :SR

    motif = hcat([0., 0., 0.])
    bravais_lattice = hcat([1., 0., 0.], [0., 1., 0.])

    crystal = Crystal(bravais_lattice, motif)
    sc = amorphize(supercell(crystal, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    # flux_values = [0.1]
    energy_pumping = zeros(Float64, neigvals, flux_points)

    for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc)
        eigval = real(eigvals(h))
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])        

        energy_pumping[:, j] = eigval[1:neigvals]

        hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
        scatter!(flux_values[2:end] .+ 1, energies[2:end]; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    gui()

    savefig("rydberg_square_flux_N$(N)_npoints$(flux_points)_disorder$(disorder).png")
    writedlm("rydberg_square_flux_N$(N)_npoints$(flux_points)_disorder$(disorder).csv", energy_pumping, ',')


end

function run_clusterN4_vs_W()

    # Random.seed!(1234)

    Rc = 2.1
    disorder = 0.0
    neigvals = 3
    nsamples = 20
    nW = 30

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    ϕ = 0.0
    m_s = 0

    N = 4

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    honeycomb = clusterN4()
    sc = amorphize(honeycomb, disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    W_values = LinRange(0.0, 3, nW)
    # flux_values = [0.1]
    energies = zeros(Float64, neigvals, nW, nsamples)

    # Init anderson term outside loop

    for (i, W) in enumerate(W_values)
        for j in range(1, nsamples)

            anderson_term = _anderson_disorder(sc, W)

            h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc, W=0, ms=m_s)
            h += anderson_term

            eigval = real(eigvals(h))
            h = sparse(h)

            display("Threaded: ")
            @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
            @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
            display(eigval[1:neigvals])        

            energies[:, i, j] = eigval[1:neigvals]

            hamiltonian = nothing # Delete to avoid memory leak
        
        end
    end

    writedlm("rydberg_cluster_N$(N)_E1_v_W_disorder$(disorder)_delta$(Δ)_ms$(m_s)_nW$(nW)_nsamples$(nsamples)_4.csv", energies[1, :, :], ',')
    writedlm("rydberg_cluster_N$(N)_E2_v_W_disorder$(disorder)_delta$(Δ)_ms$(m_s)_nW$(nW)_nsamples$(nsamples)_4.csv", energies[2, :, :], ',')
    writedlm("rydberg_cluster_N$(N)_E3_v_W_disorder$(disorder)_delta$(Δ)_ms$(m_s)_nW$(nW)_nsamples$(nsamples)_4.csv", energies[3, :, :], ',')

end


function run_clusterN4_vs_disorder()

    # Random.seed!(1234)

    Rc = 2.1
    neigvals = 3
    nsamples = 20
    ndisorder = 30

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0
    W = 0

    ϕ = 0.0
    m_s = 0

    N = 4

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    honeycomb = clusterN4()

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    disorder_values = LinRange(0.0, 0.3, ndisorder)
    # flux_values = [0.1]
    energies = zeros(Float64, neigvals, ndisorder, nsamples)

    # Init anderson term outside loop

    for (i, disorder) in enumerate(disorder_values)
        for j in range(1, nsamples)
            display("Disorder: $(disorder), sample: $(j)")

            sc = amorphize(honeycomb, disorder)

            h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc, W=0, ms=m_s)

            eigval = real(eigvals(h))
            h = sparse(h)

            display("Threaded: ")
            @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
            @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
            display(eigval[1:neigvals])        

            energies[:, i, j] = eigval[1:neigvals]

            hamiltonian = nothing # Delete to avoid memory leak
        
        end
    end

    writedlm("rydberg_cluster_N$(N)_E1_v_disorder_delta$(Δ)_ms$(m_s)_ndisorder$(ndisorder)_nsamples$(nsamples)_3.csv", energies[1, :, :], ',')
    writedlm("rydberg_cluster_N$(N)_E2_v_disorder_delta$(Δ)_ms$(m_s)_ndisorder$(ndisorder)_nsamples$(nsamples)_3.csv", energies[2, :, :], ',')
    writedlm("rydberg_cluster_N$(N)_E3_v_disorder_delta$(Δ)_ms$(m_s)_ndisorder$(ndisorder)_nsamples$(nsamples)_3.csv", energies[3, :, :], ',')

end


function run_clusterN4_gap_vs_ms_W()

    # Random.seed!(1234)

    Rc = 2.1
    disorder = 0.0
    neigvals = 3
    nsamples = 10
    npoints = 20

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    ϕ = 0.0

    N = 4

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    honeycomb = clusterN4()
    sc = amorphize(honeycomb, disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    W_values = LinRange(0.0, 5, npoints)
    ms_values = LinRange(0, 2.5, npoints)
    qs_gaps = zeros(Float64, npoints, npoints)
    mb_gaps = zeros(Float64, npoints, npoints)

    for (i, W) in enumerate(W_values)
        for (j, m_s) in enumerate(ms_values)

            qs_gap = 0
            mb_gap = 0

            for k in range(1, nsamples)

                anderson_term = _anderson_disorder(sc, W)

                h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc, W=0, ms=m_s)
                h += anderson_term

                eigval = real(eigvals(h))
                h = sparse(h)

                display("Threaded: ")
                @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
                @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
                display(eigval[1:neigvals])        

                qs_gap += eigval[2] - eigval[1]
                mb_gap += eigval[3] - eigval[2]

                hamiltonian = nothing # Delete to avoid memory leak

            end

            qs_gaps[i, j] = qs_gap / nsamples
            mb_gaps[i, j] = mb_gap / nsamples

        end
    end

    writedlm("rydberg_cluster_N$(N)_qsgap_v_ms_W_disorder$(disorder)_delta$(Δ)_np$(npoints)_nsamples$(nsamples).csv", qs_gaps, ',')
    writedlm("rydberg_cluster_N$(N)_mbgap_v_ms_W_disorder$(disorder)_delta$(Δ)_np$(npoints)_nsamples$(nsamples).csv", mb_gaps, ',')


end


function run_clusterN4_gap_vs_ms_disorder()

    # Random.seed!(1234)

    Rc = 2.1
    disorder = 0.0
    neigvals = 3
    nsamples = 20
    npoints = 30

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    ϕ = 0.0

    N = 4

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    honeycomb = clusterN4()

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    disorder_values = LinRange(0.0, 0.5, npoints)
    ms_values = LinRange(-2.5, 2.5, npoints)
    # flux_values = [0.1]
    energies = zeros(Float64, neigvals, npoints, npoints, nsamples)

    for (i, disorder) in enumerate(disorder_values)
        for (j, m_s) in enumerate(ms_values)
            for k in range(1, nsamples)

                sc = amorphize(honeycomb, disorder)
                h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc, W=0, ms=m_s)

                eigval = real(eigvals(h))
                h = sparse(h)

                display("Threaded: ")
                @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
                @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
                display(eigval[1:neigvals])        

                energies[:, i, j, k] = eigval[1:neigvals]

                hamiltonian = nothing # Delete to avoid memory leak

            end
        end
    end

    writedlm("rydberg_cluster_N$(N)_E_v_ms_disorder_delta$(Δ)_ms$(m_s)_nW$(nW)_nsamples$(nsamples)_2.csv", energies, ',')

end

function run_integer()

    Random.seed!(1233)

    Rc = 2.1
    disorder = 0.0
    neigvals = 10
    flux_points = 31
    W = 0
    m_s = 0.0

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    Nx = 2
    Ny = 2
    N = 4

    norbitals = 4 * Nx * Ny 
    which_eigval::Symbol = :SR

    honeycomb = Crystal([[3/2., sqrt(3)/2., 0.] [3/2., -sqrt(3)/2., 0.]], [[0., 0., 0.] [1., 0., 0.]])
    sc = amorphize(supercell(honeycomb, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    # flux_values = [0.0]
    energy_pumping = zeros(Float64, neigvals, flux_points)

    # Init anderson term outside loop
    anderson_term = _anderson_disorder(sc, W)

    for (j, ϕ) in enumerate(flux_values)

        display("------------------------------------")
        display("Flux insertion: $(ϕ)")

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[0.0, ϕ], cutoff=Rc, W=0, ms=m_s)
        h += anderson_term

        eigval = real(eigvals(h))
        display(eigval)
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])        

        energy_pumping[:, j] = eigval[1:neigvals]

        hamiltonian = nothing # Delete to avoid memory leak

    end

    scatter()
    for energies in eachcol(energy_pumping')
        scatter!(flux_values, energies; markershape=:hline, markercolor="red", legend=false)
    end
    scatter!(xlabel=L"\phi", ylabel=L"E\ (eV)")

    # gui()

    savefig("rydberg_integer_flux_N$(N)_npoints$(flux_points)_disorder$(disorder)_delta$(Δ)_W$(W)_ms$(m_s).png")
    writedlm("rydberg_integer_flux_N$(N)_npoints$(flux_points)_disorder$(disorder)_delta$(Δ)_W$(W)_ms$(m_s).csv", energy_pumping, ',')

end



if abspath(PROGRAM_FILE) == @__FILE__
    display("Computing energy spectrum vs disorder...")
    run_clusterN4_vs_disorder()
end

end