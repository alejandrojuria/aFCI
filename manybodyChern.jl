
module manybodyChern

include("RydbergED.jl")

using .RydbergED: Crystal, restricted_particle_hilbert_space!, RydbergModel, many_body_hamiltonian_threaded, clusterN4, amorphize, _anderson_disorder, clusterN5, supercell, RydbergModel_flux_per_orbital
using KrylovKit: eigsolve
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Random

function mbChern(flux_points::Number, W, m_s, N=4, disorder=0)

    # Random.seed!(1236)

    Rc = 2.1
    neigvals = 10

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    norbitals = 2 * 16
    which_eigval::Symbol = :SR

    if N == 4
        honeycomb = clusterN4()
    elseif N == 5
        honeycomb = clusterN5()
    end
    
    sc = amorphize(honeycomb, disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1.0
    flux_values = LinRange(0.0, max_flux, flux_points)
    display(flux_values)

    # Init anderson term outside loop
    anderson_term = _anderson_disorder(sc, W)

    # First store compute and store eigenstates for all flux values
    eigvecs = Array{Array{ComplexF64, 1}, 3}(undef, length(flux_values), length(flux_values), 2)

    for (i, ϕ_x) in enumerate(flux_values)
        for (j, ϕ_y) in enumerate(flux_values)

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[ϕ_x, ϕ_y], cutoff=Rc, W=0, ms=m_s)
        h += anderson_term
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])    
        
        eigvecs[i, j, 1] = eigvec[1]
        eigvecs[i, j, 2] = eigvec[2]

        end
    end

    hamiltonian = nothing # Delete to avoid memory leak

    # Now compute the plaquette Chern number
    chern = 0
    berry_curvature = []
    for i in range(1, flux_points - 1)
        for j in range(1, flux_points - 1)

            plaquette = [eigvecs[i, j, :], eigvecs[i+1, j, :], eigvecs[i+1, j+1, :], eigvecs[i, j+1, :], eigvecs[i, j, :]]
            links = []
            for n in range(1, length(plaquette) - 1)
                link = det(plaquette[n]' .* plaquette[n+1])
                link /= abs(link)
                append!(links, link)
            end

            plaquette_chern = real(-1im * log(prod(links))/(2*π))
            display("Plaquette Chern: $(plaquette_chern*flux_points*flux_points)")
            append!(berry_curvature, plaquette_chern)

            chern += plaquette_chern
        end
    end

    # writedlm("berry_curvature_N4_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc)_W$(W)_ms$(m_s).csv", berry_curvature, ',')

    display("Chern number: $(chern)")
    return chern
end

function mbChern_v_disorder(N=4)
    
    display("Many-body Chern number vs disorder strength...")

    nsamples = 20
    W_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    W = 0
    flux_points = 5

    cherns = Array{Float64, 2}(undef, nsamples, length(W_values))
    for (i, disorder) in enumerate(W_values)
        for j in range(1, nsamples)
            display("disorder: $(disorder), sample: $(j)")
            cherns[j, i] = mbChern(flux_points, W, 0, N, disorder)
        end
    end

    writedlm("chern_v_disorder_N$(N)_npoints$(flux_points)_nsamples$(nsamples)_sigma0p15to0p45_2.csv", cherns, ',')

end

function mbChern_v_W(ms=0, N=4)
    
    display("Many-body Chern number vs disorder strength...")

    nsamples = 10
    W_values = LinRange(0.0, 5.0, 16)
    flux_points = 5

    cherns = Array{Float64, 2}(undef, nsamples, length(W_values))
    for (i, W) in enumerate(W_values)
        for j in range(1, nsamples)
            display("W: $(W), sample: $(j)")
            cherns[j, i] = mbChern(flux_points, W, ms, N)
        end
    end

    writedlm("chern_v_W_N$(N)_ms$(ms)_npoints$(flux_points)_nsamples$(nsamples)_5.csv", cherns, ',')

end

function mbChern_v_staggered(W=0, disorder=0, N=4, nsamples=1)
    
    display("Many-body Chern number vs staggered mass at fixed disorder...")

    npoints = 10
    ms_values = LinRange(-2.5, 2.5, npoints)
    flux_points = 5

    cherns = Array{Float64, 2}(undef, length(ms_values), nsamples)
    for (i, m_s) in enumerate(ms_values)
        for j in range(1, nsamples)
            cherns[i, j] = mbChern(flux_points, W, m_s, N, disorder)
        end
    end

    writedlm("chern_v_ms_N$(N)_W$(W)_disorder$(disorder)_npoints$(flux_points)_nsamples$(nsamples)_npoints$(npoints).csv", cherns, ',')

end


function mbChern_v_ms_anderson()

    display("Many-body Chern number vs staggered mass and Anderson strength...")

    N = 4
    nsamples = 10
    nW = 11
    W_values = LinRange(0.0, 5.0, nW)
    ms_values = LinRange(-2.5, 2.5, nW)
    flux_points = 5

    cherns = Array{Float64, 3}(undef, nW, nW, nsamples)
    for (i, W) in enumerate(W_values)
        for (j, ms) in enumerate(ms_values)
            for k in range(1, nsamples)
                display("W: $(W), ms: $(ms), sample: $(k)")
                cherns[i, j, k] = mbChern(flux_points, W, ms)
            end
        end
    end

    writedlm("chern_v_ms_W_N$(N)_npoints$(flux_points)_nsamples$(nsamples)_nW$(nW).csv", cherns, ',')

end

function mbChern_v_ms_disorder()

    display("Many-body Chern number vs staggered mass and disorder strength...")

    N = 4
    nsamples = 1
    nW = 11
    disorder_values = LinRange(0.0, 0.5, nW)
    ms_values = LinRange(-2.5, 2.5, nW)
    flux_points = 5

    cherns = Array{Float64, 3}(undef, nW, nW, nsamples)
    for (i, disorder) in enumerate(disorder_values)
        for (j, ms) in enumerate(ms_values)
            for k in range(1, nsamples)
                display("disorder: $(disorder), ms: $(ms), sample: $(k)")
                cherns[i, j, k] = mbChern(flux_points, 0, ms, N, disorder)
            end
        end
    end

    writedlm("chern_v_ms_disorder_N$(N)_npoints$(flux_points)_nsamples$(nsamples)_nW$(nW)_1.csv", cherns, ',')

end


function mbChern_integer(flux_points::Number, Nx, Ny, W=0, disorder=0, m_s=0)

    # Random.seed!(1233)

    Rc = 2.1
    neigvals = 10

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    norbitals = 4 * Nx * Ny
    which_eigval::Symbol = :SR

    N = Nx * Ny
    
    honeycomb = Crystal([[3/2., sqrt(3)/2., 0.] [3/2., -sqrt(3)/2., 0.]], [[0., 0., 0.] [1., 0., 0.]])
    sc = amorphize(supercell(honeycomb, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1
    flux_values = LinRange(0.0, max_flux, flux_points)
    display(flux_values)

    # Init anderson term outside loop
    anderson_term = _anderson_disorder(sc, W)

    # First store compute and store eigenstates for all flux values
    eigvecs = Array{Array{ComplexF64, 1}, 3}(undef, length(flux_values), length(flux_values), 1)
    eigvals = Array{Float64, 3}(undef, length(flux_values), length(flux_values), neigvals)

    for (i, ϕ_x) in enumerate(flux_values)
        for (j, ϕ_y) in enumerate(flux_values)

        h, bonds = RydbergModel(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[ϕ_x, ϕ_y], cutoff=Rc, W=0, ms=m_s)
        h += anderson_term
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])    
        
        eigvecs[i, j, 1] = eigvec[2]
        eigvals[i, j, :] = eigval[1:neigvals]

        end
    end

    hamiltonian = nothing # Delete to avoid memory leak

    # Now compute the plaquette Chern number
    chern = 0
    berry_curvature = []
    for i in range(1, flux_points - 1)
        for j in range(1, flux_points - 1)

            plaquette = [eigvecs[i, j, :], eigvecs[i+1, j, :], eigvecs[i+1, j+1, :], eigvecs[i, j+1, :], eigvecs[i, j, :]]
            links = []
            for n in range(1, length(plaquette) - 1)
                link = det(plaquette[n]' .* plaquette[n+1])
                link /= abs(link)
                append!(links, link)
            end

            plaquette_chern = real(-1im * log(prod(links))/(2*π))
            display("Plaquette Chern: $(plaquette_chern*flux_points*flux_points)")
            append!(berry_curvature, plaquette_chern)

            chern += plaquette_chern
        end
    end

    writedlm("berry_curvature_N4_integer_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc)_W$(W)_ms$(m_s).csv", berry_curvature, ',')
    writedlm("eigvals_N4_integer_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc)_W$(W)_ms$(m_s).csv", eigvals, ',')

    display("Chern number: $(chern)")
    return chern
end

function mbChern_integer_flux_per_orbital(flux_points::Number, Nx, Ny, orbitals, W=0, disorder=0, m_s=0)

    # Random.seed!(1233)

    Rc = 2.1
    neigvals = 10

    ta = 1.26
    tb = 0.49
    Δ = 18.52
    ω = 2.38
    V = 0.0

    norbitals = 4 * Nx * Ny
    which_eigval::Symbol = :SR

    N = Nx * Ny
    
    honeycomb = Crystal([[3/2., sqrt(3)/2., 0.] [3/2., -sqrt(3)/2., 0.]], [[0., 0., 0.] [1., 0., 0.]])
    sc = amorphize(supercell(honeycomb, Nx, Ny), disorder)

    hilbert_space = Vector{Int}()
    restricted_particle_hilbert_space!(norbitals, N, hilbert_space)
    display("Hilbert space dim.: $(length(hilbert_space))")

    max_flux    = 1
    flux_values = LinRange(0.0, max_flux, flux_points)
    display(flux_values)

    # Init anderson term outside loop
    anderson_term = _anderson_disorder(sc, W)

    # First store compute and store eigenstates for all flux values
    eigvecs = Array{Array{ComplexF64, 1}, 2}(undef, length(flux_values), length(flux_values))
    eigvals = Array{Float64, 3}(undef, length(flux_values), length(flux_values), neigvals)

    for (i, ϕ_x) in enumerate(flux_values)
        for (j, ϕ_y) in enumerate(flux_values)

        h, bonds = RydbergModel_flux_per_orbital(sc, ta=ta, tb=tb, Δ=Δ, ω=ω, boundary="PBC", flux=[ϕ_x, ϕ_y], cutoff=Rc, W=0, ms=m_s, orbitals=orbitals)
        h += anderson_term
        h = sparse(h)

        display("Threaded: ")
        @time hamiltonian = many_body_hamiltonian_threaded(hilbert_space, h, V)
        @time eigval, eigvec, info = eigsolve(hamiltonian, neigvals, which_eigval; ishermitian=true, krylovdim=50, tol=1e-20)
        display(eigval[1:neigvals])    
        
        eigvecs[i, j] = eigvec[1]
        eigvals[i, j, :] = eigval[1:neigvals]

        end
    end

    hamiltonian = nothing # Delete to avoid memory leak

    # Now compute the plaquette Chern number
    chern = 0
    berry_curvature = []
    for i in range(1, flux_points - 1)
        for j in range(1, flux_points - 1)

            plaquette = [eigvecs[i, j], eigvecs[i+1, j], eigvecs[i+1, j+1], eigvecs[i, j+1], eigvecs[i, j]]

            links = []            
            for n in range(1, length(plaquette) - 1)
                link = dot(plaquette[n], plaquette[n+1])
                link /= abs(link)
                append!(links, link)
            end

            plaquette_chern = -1im * log(prod(links))/(2*π)
            display("Plaquette Chern: $(plaquette_chern*flux_points*flux_points)")
            append!(berry_curvature, real(plaquette_chern))

            chern += plaquette_chern
        end
    end

    writedlm("berry_curvature_N4_integer_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc)_W$(W)_ms$(m_s).csv", berry_curvature, ',')
    writedlm("eigvals_N4_integer_npoints$(flux_points)_disorder$(disorder)_Rc$(Rc)_W$(W)_ms$(m_s).csv", eigvals, ',')

    display("Chern number: $(chern)")
    return chern
end

function mbChern_integer_v_W(ms=0)
    
    display("Many-body Chern number vs disorder strength...")

    nsamples = 50
    W_values = LinRange(0.0, 5.0, 10)
    flux_points = 7
    Nx = 2
    Ny = 2

    cherns = Array{Float64, 2}(undef, nsamples, length(W_values))
    for (i, W) in enumerate(W_values)
        for j in range(1, nsamples)
            display("W: $(W), sample: $(j)")
            cherns[j, i] = real(mbChern_integer(flux_points, Nx, Ny, W, 0, ms))
        end
    end

    writedlm("chern_integer_v_W_Nb4_ms$(ms)_npoints$(flux_points)_nsamples$(nsamples)_3.csv", cherns, ',')

end

function main()
    
    mbChern_integer_flux_per_orbital(8, 2, 2, [1, 2], 0.0, 0.0, 0.0)

end

# main()

end