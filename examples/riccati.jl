using BLASFEO
using BLASFEO.LibBlasfeo

struct OcpStructuredLinearSystem
    N::Int
    x0::BlasfeoDvec
    nx::Vector{Int}
    nu::Vector{Int}
    BAbt::Vector{BlasfeoDmat}
    RSQrq::Vector{BlasfeoDmat}
    L::Vector{BlasfeoDmat}
    ux::Vector{BlasfeoDvec}
    pi::Vector{BlasfeoDvec}
    work_mat::BlasfeoDmat
    work_vec::BlasfeoDvec
end

function OcpStructuredLinearSystem(
    As::Vector{<:AbstractArray{Float64}},
    Bs::Vector{<:AbstractArray{Float64}},
    bs::Vector{<:AbstractArray{Float64}},
    Qs::Vector{<:AbstractArray{Float64}},
    qs::Vector{<:AbstractArray{Float64}},
    Rs::Vector{<:AbstractArray{Float64}},
    rs::Vector{<:AbstractArray{Float64}},
    Ss::Vector{<:AbstractArray{Float64}},
    x0::Vector{Float64}
    )
    # Do no size checks :P
    N = length(Bs)
    nx = [0; [size(A,1) for A in As]]
    nu = [[size(B,2) for B in Bs]; 0]
    ux = [BlasfeoDvec(nx[ii]+nu[ii]+1) for ii in 1:N+1]
    pi = [BlasfeoDvec(nx[ii]) for ii in 1:N+1]

    # Adjust bs[1]
    bs[1] += As[1]*x0

    BAbt = Vector{BlasfeoDmat}(undef, N)
    RSQrq = Vector{BlasfeoDmat}(undef, N+1)
    L = [BlasfeoDmat(nx[ii]+nu[ii]+1, nx[ii]+nu[ii]) for ii in 1:N+1]
    BAbt[1] = BlasfeoDmat(permutedims(hcat(Bs[1],bs[1])))
    RSQrq[1] = BlasfeoDmat([Rs[1]; rs[1]])
    for ii in 2:N
        BAbt[ii] = BlasfeoDmat(permutedims(hcat(Bs[ii],As[ii],bs[ii])))
        RSQ = vcat(
            hcat(Rs[ii],zeros(nu[ii],nx[ii])),
            hcat(Ss[ii], Qs[ii]),
            permutedims(vcat(rs[ii], qs[ii])),
        )
        RSQrq[ii] = BlasfeoDmat(RSQ)
    end
    RSQrq[N+1] = BlasfeoDmat(vcat(Qs[N+1],transpose(qs[N+1])))

    return OcpStructuredLinearSystem(
        N,
        BlasfeoDvec(x0),
        nx,
        nu,
        BAbt,
        RSQrq,
        L,
        ux,
        pi,
        BlasfeoDmat(maximum(nu) + maximum(nx) + 1, maximum(nx)),
        BlasfeoDvec(maximum(nx)),
    )
end


function simple_model(N;x0=[-1;-1.])
    As = [[1. 1.; 0. 1.] for i in 1:N]
    Bs = [[0.; 1;;] for i in 1:N]
    bs = [[0.1; 0.1] for i in 1:N]
    Qs = push!([[1. 0.; 0. 1.] for i in 1:N],[10. 0.;0. 10.])
    qs = push!([[0.1; 0.1] for i in 1:N],[10. ; 10.])
    Ss = [[0.; 0.] for i in 1:N]
    Rs = [[1.;;] for i in 1:N]
    rs = [[0.1] for i in 1:N]

    return As, Bs, bs, Qs, qs, Rs, rs, Ss, x0
end

"""
  An example riccati recursion solver via the thin-wrapper interface
"""
function riccati(kkt::OcpStructuredLinearSystem)
    N = kkt.N
    nx = kkt.nx
    nu = kkt.nu
    RSQrq = kkt.RSQrq
    BAbt = kkt.BAbt
    ux = kkt.ux
    pi = kkt.pi
    work_mat = kkt.work_mat
    work_vec = kkt.work_vec
    L = kkt.L
    nx_end = kkt.nx[N+1]
    # Factorize last stage Q
    blasfeo_dpotrf_l_mn(nx_end+1,nx_end, RSQrq[N+1], 0, 0, L[N+1], 0, 0)

    for ii in N:-1:1
        blasfeo_dtrmm_rlnn(
            nu[ii]+nx[ii]+1,
            nx[ii+1],
            1.0, L[ii+1], nu[ii+1], nu[ii+1],
            BAbt[ii], 0, 0,
            work_mat, 0, 0,
        )
        blasfeo_dgead(
			      1, nx[ii+1],
			      1.0,
			      L[ii+1], nu[ii+1]+nx[ii+1], nu[ii+1],
			      work_mat, nu[ii]+nx[ii], 0,
			  )

        blasfeo_dsyrk_dpotrf_ln_mn(
			      nu[ii]+nx[ii]+1,
			      nu[ii]+nx[ii],
			      nx[ii+1],
			      work_mat, 0, 0,
			      work_mat, 0, 0,
			      RSQrq[ii], 0, 0,
			      L[ii], 0, 0
        )
    end

    blasfeo_drowex(nu[1]+nx[1], -1.0, L[1], nu[1]+nx[1], 0, ux[1], 0)
	  blasfeo_dtrsv_ltn(nu[1]+nx[1], L[1], 0, 0, ux[1], 0, ux[1], 0)
	  blasfeo_drowex(nx[2], 1.0, BAbt[1], nu[1]+nx[1], 0, ux[2], nu[2])
	  blasfeo_dgemv_t(nu[1]+nx[1], nx[2], 1.0, BAbt[1], 0, 0, ux[1], 0, 1.0, ux[2], nu[2], ux[2], nu[2])
	  blasfeo_dveccp(nx[2], ux[2], nu[2], pi[1], 0)
	  blasfeo_drowex(nx[2], 1.0, L[2], nu[2]+nx[2], nu[2], work_vec, 0)
	  blasfeo_dtrmv_ltn(nx[2], L[2], nu[2], nu[2], pi[1], 0, pi[1], 0)
	  blasfeo_daxpy(nx[2], 1.0, work_vec, 0, pi[1], 0, pi[1], 0)
	  blasfeo_dtrmv_lnn(nx[2], L[2], nu[2], nu[2], pi[1], 0, pi[1], 0)

    for ii in 2:N
        blasfeo_drowex(nu[ii], -1.0, L[ii], nu[ii]+nx[ii], 0, ux[ii], 0)
		    blasfeo_dtrsv_ltn_mn(nu[ii]+nx[ii], nu[ii], L[ii], 0, 0, ux[ii], 0, ux[ii], 0)
		    blasfeo_drowex(nx[ii+1], 1.0, BAbt[ii], nu[ii]+nx[ii], 0, ux[ii+1], nu[ii+1])
		    blasfeo_dgemv_t(nu[ii]+nx[ii], nx[ii+1], 1.0, BAbt[ii], 0, 0, ux[ii], 0, 1.0, ux[ii+1], nu[ii+1], ux[ii+1], nu[ii+1])
		    blasfeo_dveccp(nx[ii+1], ux[ii+1], nu[ii+1], pi[ii], 0)
		    blasfeo_drowex(nx[ii+1], 1.0, L[ii+1], nu[ii+1]+nx[ii+1], nu[ii+1], work_vec, 0)
		    blasfeo_dtrmv_ltn(nx[ii+1], L[ii+1], nu[ii+1], nu[ii+1], pi[ii], 0, pi[ii], 0)
		    blasfeo_daxpy(nx[ii+1], 1.0, work_vec, 0, pi[ii], 0, pi[ii], 0)
		    blasfeo_dtrmv_lnn(nx[ii+1], L[ii+1], nu[ii+1], nu[ii+1], pi[ii], 0, pi[ii], 0)
    end
end

function get_u(kkt::OcpStructuredLinearSystem)
    return [kkt.ux[ii][1:kkt.nu[ii]] for ii in 1:kkt.N]
end

function get_x(kkt::OcpStructuredLinearSystem)
    x = [kkt.ux[ii][kkt.nu[ii]+1:kkt.nu[ii]+kkt.nx[ii]] for ii in 1:kkt.N+1]
    x[1] = kkt.x0
    return x
end
