module LibBlasfeo

using blasfeo_jll
export blasfeo_jll

using CEnum: CEnum, @cenum

"""
    blasfeo_dmat

matrix structure
"""
struct blasfeo_dmat
    mem::Ptr{Cdouble}
    pA::Ptr{Cdouble}
    dA::Ptr{Cdouble}
    m::Cint
    n::Cint
    pm::Cint
    cn::Cint
    use_dA::Cint
    memsize::Cint
end

struct blasfeo_smat
    mem::Ptr{Cfloat}
    pA::Ptr{Cfloat}
    dA::Ptr{Cfloat}
    m::Cint
    n::Cint
    pm::Cint
    cn::Cint
    use_dA::Cint
    memsize::Cint
end

"""
    blasfeo_dvec

vector structure
"""
struct blasfeo_dvec
    mem::Ptr{Cdouble}
    pa::Ptr{Cdouble}
    m::Cint
    pm::Cint
    memsize::Cint
end

struct blasfeo_svec
    mem::Ptr{Cfloat}
    pa::Ptr{Cfloat}
    m::Cint
    pm::Cint
    memsize::Cint
end

"""
    blasfeo_pm_dmat

Explicitly panel-major matrix structure
"""
struct blasfeo_pm_dmat
    mem::Ptr{Cdouble}
    pA::Ptr{Cdouble}
    dA::Ptr{Cdouble}
    m::Cint
    n::Cint
    pm::Cint
    cn::Cint
    use_dA::Cint
    ps::Cint
    memsize::Cint
end

struct blasfeo_pm_smat
    mem::Ptr{Cfloat}
    pA::Ptr{Cfloat}
    dA::Ptr{Cfloat}
    m::Cint
    n::Cint
    pm::Cint
    cn::Cint
    use_dA::Cint
    ps::Cint
    memsize::Cint
end

mutable struct blasfeo_pm_dvec
    mem::Ptr{Cdouble}
    pa::Ptr{Cdouble}
    m::Cint
    pm::Cint
    memsize::Cint
    blasfeo_pm_dvec() = new()
end

mutable struct blasfeo_pm_svec
    mem::Ptr{Cfloat}
    pa::Ptr{Cfloat}
    m::Cint
    pm::Cint
    memsize::Cint
    blasfeo_pm_svec() = new()
end

"""
    blasfeo_cm_dmat

Explicitly column-major matrix structure
"""
struct blasfeo_cm_dmat
    mem::Ptr{Cdouble}
    pA::Ptr{Cdouble}
    dA::Ptr{Cdouble}
    m::Cint
    n::Cint
    use_dA::Cint
    memsize::Cint
end

struct blasfeo_cm_smat
    mem::Ptr{Cfloat}
    pA::Ptr{Cfloat}
    dA::Ptr{Cfloat}
    m::Cint
    n::Cint
    use_dA::Cint
    memsize::Cint
end

struct blasfeo_cm_dvec
    mem::Ptr{Cdouble}
    pa::Ptr{Cdouble}
    m::Cint
    memsize::Cint
end

struct blasfeo_cm_svec
    mem::Ptr{Cfloat}
    pa::Ptr{Cfloat}
    m::Cint
    memsize::Cint
end

"""
    blasfeo_daxpy(kmax, alpha, sx, xi, sy, yi, sz, zi)

z = y + alpha*x
z[zi:zi+n] = alpha*x[xi:xi+n] + y[yi:yi+n]
NB: Different arguments semantic compare to equivalent standard BLAS routine
"""
function blasfeo_daxpy(kmax, alpha, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_daxpy(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_daxpby(kmax, alpha, sx, xi, beta, sy, yi, sz, zi)

z = beta*y + alpha*x
"""
function blasfeo_daxpby(kmax, alpha, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_daxpby(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dvecmul(m, sx, xi, sy, yi, sz, zi)

z = x .* y
"""
function blasfeo_dvecmul(m, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dvecmul(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dvecmulacc(m, sx, xi, sy, yi, sz, zi)

z += x .* y
"""
function blasfeo_dvecmulacc(m, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dvecmulacc(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dvecmuldot(m, sx, xi, sy, yi, sz, zi)

z = x .* y, return sum(z) = x^T * y
"""
function blasfeo_dvecmuldot(m, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dvecmuldot(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cdouble
end

"""
    blasfeo_ddot(m, sx, xi, sy, yi)

return x^T * y
"""
function blasfeo_ddot(m, sx, xi, sy, yi)
    @ccall blasfeo.blasfeo_ddot(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint)::Cdouble
end

"""
    blasfeo_drotg(a, b, c, s)

construct givens plane rotation
"""
function blasfeo_drotg(a, b, c, s)
    @ccall blasfeo.blasfeo_drotg(a::Cdouble, b::Cdouble, c::Ptr{Cdouble}, s::Ptr{Cdouble})::Cvoid
end

"""
    blasfeo_dcolrot(m, sA, ai, aj0, aj1, c, s)

apply plane rotation [a b] [c -s; s; c] to the aj0 and aj1 columns of A at row index ai
"""
function blasfeo_dcolrot(m, sA, ai, aj0, aj1, c, s)
    @ccall blasfeo.blasfeo_dcolrot(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj0::Cint, aj1::Cint, c::Cdouble, s::Cdouble)::Cvoid
end

"""
    blasfeo_drowrot(m, sA, ai0, ai1, aj, c, s)

apply plane rotation [c s; -s c] [a; b] to the ai0 and ai1 rows of A at column index aj
"""
function blasfeo_drowrot(m, sA, ai0, ai1, aj, c, s)
    @ccall blasfeo.blasfeo_drowrot(m::Cint, sA::Ptr{blasfeo_dmat}, ai0::Cint, ai1::Cint, aj::Cint, c::Cdouble, s::Cdouble)::Cvoid
end

"""
    blasfeo_dgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x
"""
function blasfeo_dgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dgemv_n(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A^T * x
"""
function blasfeo_dgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dgemv_t(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_lnn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A ) * x, A (m)x(n)
"""
function blasfeo_dtrsv_lnn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_lnn_mn(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_ltn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A^T ) * x, A (m)x(n)
"""
function blasfeo_dtrsv_ltn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_ltn_mn(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_lnn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A ) * x, A (m)x(m) lower, not_transposed
"""
function blasfeo_dtrsv_lnn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_lnn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_lnu(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A ) * x, A (m)x(m) lower, not_transposed, assuming unit diagonal
"""
function blasfeo_dtrsv_lnu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_lnu(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_ltn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A^T ) * x, A (m)x(m) lower, transposed
"""
function blasfeo_dtrsv_ltn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_ltn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_ltu(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A^T ) * x, A (m)x(m) lower, transposed, assuming unit diagonal
"""
function blasfeo_dtrsv_ltu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_ltu(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_unn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A^T ) * x, A (m)x(m) upper, not_transposed
"""
function blasfeo_dtrsv_unn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_unn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrsv_utn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A^T ) * x, A (m)x(m) upper, transposed
"""
function blasfeo_dtrsv_utn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrsv_utn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_lnn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A lower triangular
"""
function blasfeo_dtrmv_lnn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_lnn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_lnu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A lower triangular, assuming unit diagonal
"""
function blasfeo_dtrmv_lnu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_lnu(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_ltn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A^T * x ; A lower triangular
"""
function blasfeo_dtrmv_ltn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_ltn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_ltu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A^T * x ; A lower triangular, assuming unit diagonal
"""
function blasfeo_dtrmv_ltu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_ltu(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_unn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A upper triangular
"""
function blasfeo_dtrmv_unn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_unn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_unu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A upper triangular, assuming unit diagonal
"""
function blasfeo_dtrmv_unu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_unu(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_utn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A^T * x ; A upper triangular
"""
function blasfeo_dtrmv_utn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_utn(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dtrmv_utu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A^T * x ; A upper triangular, assuming unit diagonal
"""
function blasfeo_dtrmv_utu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dtrmv_utu(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dgemv_nt(m, n, alpha_n, alpha_t, sA, ai, aj, sx_n, xi_n, sx_t, xi_t, beta_n, beta_t, sy_n, yi_n, sy_t, yi_t, sz_n, zi_n, sz_t, zi_t)

z_n <= beta_n * y_n + alpha_n * A  * x_n
z_t <= beta_t * y_t + alpha_t * A^T * x_t
"""
function blasfeo_dgemv_nt(m, n, alpha_n, alpha_t, sA, ai, aj, sx_n, xi_n, sx_t, xi_t, beta_n, beta_t, sy_n, yi_n, sy_t, yi_t, sz_n, zi_n, sz_t, zi_t)
    @ccall blasfeo.blasfeo_dgemv_nt(m::Cint, n::Cint, alpha_n::Cdouble, alpha_t::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx_n::Ptr{blasfeo_dvec}, xi_n::Cint, sx_t::Ptr{blasfeo_dvec}, xi_t::Cint, beta_n::Cdouble, beta_t::Cdouble, sy_n::Ptr{blasfeo_dvec}, yi_n::Cint, sy_t::Ptr{blasfeo_dvec}, yi_t::Cint, sz_n::Ptr{blasfeo_dvec}, zi_n::Cint, sz_t::Ptr{blasfeo_dvec}, zi_t::Cint)::Cvoid
end

"""
    blasfeo_dsymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x, where A is symmetric and only the lower triangular patr of A is accessed
"""
function blasfeo_dsymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dsymv_l(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

function blasfeo_dsymv_l_mn(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dsymv_l_mn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dsymv_u(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x, where A is symmetric and only the upper triangular patr of A is accessed
"""
function blasfeo_dsymv_u(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dsymv_u(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dger(m, n, alpha, sx, xi, sy, yi, sC, ci, cj, sD, di, dj)

D = C + alpha * x * y^T
"""
function blasfeo_dger(m, n, alpha, sx, xi, sy, yi, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dger(m::Cint, n::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgemv_d(m, alpha, sA, ai, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x, A diagonal
"""
function blasfeo_dgemv_d(m, alpha, sA, ai, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_dgemv_d(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dvec}, ai::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_dvec}, yi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B
"""
function blasfeo_dgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgemm_nn(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T
"""
function blasfeo_dgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgemm_nt(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B
"""
function blasfeo_dgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgemm_tn(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B^T
"""
function blasfeo_dgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgemm_tt(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T ; C, D lower triangular
"""
function blasfeo_dsyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_ln(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_dsyrk_ln_mn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_ln_mn(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B ; C, D lower triangular
"""
function blasfeo_dsyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_lt(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T ; C, D upper triangular
"""
function blasfeo_dsyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_un(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B ; C, D upper triangular
"""
function blasfeo_dsyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_ut(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A lower triangular
"""
function blasfeo_dtrmm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_llnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A lower triangular assuming unit diagonal
"""
function blasfeo_dtrmm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_llnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A lower triangular
"""
function blasfeo_dtrmm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_lltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A lower triangular assuming unit diagonal
"""
function blasfeo_dtrmm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_lltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A upper triangular
"""
function blasfeo_dtrmm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_lunn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A upper triangular assuming unit diagonal
"""
function blasfeo_dtrmm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_lunu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A upper triangular
"""
function blasfeo_dtrmm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_lutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A upper triangular assuming unit diagonal
"""
function blasfeo_dtrmm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_lutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A lower triangular
"""
function blasfeo_dtrmm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_rlnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A lower triangular assuming unit diagonal
"""
function blasfeo_dtrmm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_rlnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A lower triangular
"""
function blasfeo_dtrmm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_rltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A lower triangular assuming unit diagonal
"""
function blasfeo_dtrmm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_rltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A upper triangular
"""
function blasfeo_dtrmm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_runn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A upper triangular assuming unit diagonal
"""
function blasfeo_dtrmm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_runu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A upper triangular
"""
function blasfeo_dtrmm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_rutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrmm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A upper triangular assuming unit diagonal
"""
function blasfeo_dtrmm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrmm_rutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A lower triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_llnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A lower triangular assuming unit diagonal
"""
function blasfeo_dtrsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_llnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A lower triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_lltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A lower triangular assuming unit diagonal
"""
function blasfeo_dtrsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_lltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_lunn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A upper triangular assuming unit diagonal
"""
function blasfeo_dtrsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_lunu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_lutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A upper triangular assuming unit diagonal
"""
function blasfeo_dtrsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_lutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A lower triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_rlnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A lower triangular assuming unit diagonal
"""
function blasfeo_dtrsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_rlnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_rltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A lower triangular assuming unit diagonal
"""
function blasfeo_dtrsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_rltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_runn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A upper triangular assuming unit diagonal
"""
function blasfeo_dtrsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_runu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_dtrsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_rutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dtrsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A upper triangular assuming unit diagonal
"""
function blasfeo_dtrsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_dtrsm_rutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyr2k_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T + alpha * B * A^T; C, D lower triangular
"""
function blasfeo_dsyr2k_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyr2k_ln(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyr2k_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B + alpha * B^T * A; C, D lower triangular
"""
function blasfeo_dsyr2k_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyr2k_lt(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyr2k_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T + alpha * B * A^T; C, D upper triangular
"""
function blasfeo_dsyr2k_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyr2k_un(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyr2k_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B + alpha * B^T * A; C, D upper triangular
"""
function blasfeo_dsyr2k_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyr2k_ut(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    dgemm_diag_left_lib(m, n, alpha, dA, pB, sdb, beta, pC, sdc, pD, sdd)

D <= alpha * A * B + beta * C, with A diagonal (stored as strvec)
"""
function dgemm_diag_left_lib(m, n, alpha, dA, pB, sdb, beta, pC, sdc, pD, sdd)
    @ccall blasfeo.dgemm_diag_left_lib(m::Cint, n::Cint, alpha::Cdouble, dA::Ptr{Cdouble}, pB::Ptr{Cdouble}, sdb::Cint, beta::Cdouble, pC::Ptr{Cdouble}, sdc::Cint, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function blasfeo_dgemm_dn(m, n, alpha, sA, ai, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgemm_dn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dvec}, ai::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgemm_nd(m, n, alpha, sA, ai, aj, sB, bi, beta, sC, ci, cj, sD, di, dj)

D <= alpha * A * B + beta * C, with B diagonal (stored as strvec)
"""
function blasfeo_dgemm_nd(m, n, alpha, sA, ai, aj, sB, bi, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgemm_nd(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dvec}, bi::Cint, beta::Cdouble, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dpotrf_l(m, sC, ci, cj, sD, di, dj)

D <= chol( C ) ; C, D lower triangular
"""
function blasfeo_dpotrf_l(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dpotrf_l(m::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_dpotrf_l_mn(m, n, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dpotrf_l_mn(m::Cint, n::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dpotrf_u(m, sC, ci, cj, sD, di, dj)

D <= chol( C ) ; C, D upper triangular
"""
function blasfeo_dpotrf_u(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dpotrf_u(m::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dsyrk_dpotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj)

D <= chol( C + A * B' ) ; C, D lower triangular
D <= chol( C + A * B^T ) ; C, D lower triangular
"""
function blasfeo_dsyrk_dpotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_dpotrf_ln(m::Cint, k::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_dsyrk_dpotrf_ln_mn(m, n, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dsyrk_dpotrf_ln_mn(m::Cint, n::Cint, k::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgetrf_np(m, n, sC, ci, cj, sD, di, dj)

D <= lu( C ) ; no pivoting
"""
function blasfeo_dgetrf_np(m, n, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_dgetrf_np(m::Cint, n::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_dgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv)

D <= lu( C ) ; row pivoting
"""
function blasfeo_dgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv)
    @ccall blasfeo.blasfeo_dgetrf_rp(m::Cint, n::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint, ipiv::Ptr{Cint})::Cvoid
end

"""
    blasfeo_dgeqrf_worksize(m, n)

D <= qr( C )
"""
function blasfeo_dgeqrf_worksize(m, n)
    @ccall blasfeo.blasfeo_dgeqrf_worksize(m::Cint, n::Cint)::Cint
end

function blasfeo_dgeqrf(m, n, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_dgeqrf(m::Cint, n::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_dorglq_worksize(m, n, k)

D <= Q factor, where C is the output of the LQ factorization
"""
function blasfeo_dorglq_worksize(m, n, k)
    @ccall blasfeo.blasfeo_dorglq_worksize(m::Cint, n::Cint, k::Cint)::Cint
end

function blasfeo_dorglq(m, n, k, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_dorglq(m::Cint, n::Cint, k::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_dgelqf(m, n, sC, ci, cj, sD, di, dj, work)

D <= lq( C )
"""
function blasfeo_dgelqf(m, n, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_dgelqf(m::Cint, n::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

function blasfeo_dgelqf_worksize(m, n)
    @ccall blasfeo.blasfeo_dgelqf_worksize(m::Cint, n::Cint)::Cint
end

"""
    blasfeo_dgelqf_pd(m, n, sC, ci, cj, sD, di, dj, work)

D <= lq( C ), positive diagonal elements
"""
function blasfeo_dgelqf_pd(m, n, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_dgelqf_pd(m::Cint, n::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_dgelqf_pd_la(m, n1, sL, li, lj, sA, ai, aj, work)

[L, A] <= lq( [L, A] ), positive diagonal elements, array of matrices, with
L lower triangular, of size (m)x(m)
A full, of size (m)x(n1)
"""
function blasfeo_dgelqf_pd_la(m, n1, sL, li, lj, sA, ai, aj, work)
    @ccall blasfeo.blasfeo_dgelqf_pd_la(m::Cint, n1::Cint, sL::Ptr{blasfeo_dmat}, li::Cint, lj::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_dgelqf_pd_lla(m, n1, sL0, l0i, l0j, sL1, l1i, l1j, sA, ai, aj, work)

[L, L, A] <= lq( [L, L, A] ), positive diagonal elements, array of matrices, with:
L lower triangular, of size (m)x(m)
A full, of size (m)x(n1)
"""
function blasfeo_dgelqf_pd_lla(m, n1, sL0, l0i, l0j, sL1, l1i, l1j, sA, ai, aj, work)
    @ccall blasfeo.blasfeo_dgelqf_pd_lla(m::Cint, n1::Cint, sL0::Ptr{blasfeo_dmat}, l0i::Cint, l0j::Cint, sL1::Ptr{blasfeo_dmat}, l1i::Cint, l1j::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_cm_dgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

BLAS 3
"""
function blasfeo_cm_dgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dgemm_nn(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dgemm_nt(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dgemm_tn(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dgemm_tt(m::Cint, n::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk_ln(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk_lt(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk_un(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk_ut(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk3_ln(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk3_ln(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk3_lt(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk3_lt(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk3_un(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk3_un(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyrk3_ut(m, k, alpha, sA, ai, aj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyrk3_ut(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_llnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_llnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_lltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_lltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_lunn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_lunu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_lutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_lutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_rlnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_rlnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_rltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_rltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_runn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_runu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_rutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrsm_rutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_llnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_llnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_lltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_lltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_lunn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_lunu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_lutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_lutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_rlnn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_rlnu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_rltn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_rltu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_runn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_runu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_rutn(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dtrmm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dtrmm_rutu(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyr2k_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyr2k_ln(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyr2k_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyr2k_lt(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyr2k_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyr2k_un(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dsyr2k_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dsyr2k_ut(m::Cint, k::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint, beta::Cdouble, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_cm_dgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

BLAS 2
"""
function blasfeo_cm_dgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_cm_dgemv_n(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_cm_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_cm_dvec}, yi::Cint, sz::Ptr{blasfeo_cm_dvec}, zi::Cint)::Cvoid
end

function blasfeo_cm_dgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_cm_dgemv_t(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_cm_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_cm_dvec}, yi::Cint, sz::Ptr{blasfeo_cm_dvec}, zi::Cint)::Cvoid
end

function blasfeo_cm_dsymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_cm_dsymv_l(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_cm_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_cm_dvec}, yi::Cint, sz::Ptr{blasfeo_cm_dvec}, zi::Cint)::Cvoid
end

function blasfeo_cm_dsymv_u(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_cm_dsymv_u(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_cm_dvec}, xi::Cint, beta::Cdouble, sy::Ptr{blasfeo_cm_dvec}, yi::Cint, sz::Ptr{blasfeo_cm_dvec}, zi::Cint)::Cvoid
end

function blasfeo_cm_dger(m, n, alpha, sx, xi, sy, yi, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dger(m::Cint, n::Cint, alpha::Cdouble, sx::Ptr{blasfeo_cm_dvec}, xi::Cint, sy::Ptr{blasfeo_cm_dvec}, yi::Cint, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_cm_dpotrf_l(m, sC, ci, cj, sD, di, dj)

LAPACK
"""
function blasfeo_cm_dpotrf_l(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dpotrf_l(m::Cint, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dpotrf_u(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_dpotrf_u(m::Cint, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_dgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv)
    @ccall blasfeo.blasfeo_cm_dgetrf_rp(m::Cint, n::Cint, sC::Ptr{blasfeo_cm_dmat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_dmat}, di::Cint, dj::Cint, ipiv::Ptr{Cint})::Cvoid
end

"""
    blasfeo_saxpy(kmax, alpha, sx, xi, sy, yi, sz, zi)

z = y + alpha*x
"""
function blasfeo_saxpy(kmax, alpha, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_saxpy(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_saxpby(kmax, alpha, sx, xi, beta, sy, yi, sz, zi)

z = beta*y + alpha*x
"""
function blasfeo_saxpby(kmax, alpha, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_saxpby(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_svecmul(m, sx, xi, sy, yi, sz, zi)

z = x .* y
"""
function blasfeo_svecmul(m, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_svecmul(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_svecmulacc(m, sx, xi, sy, yi, sz, zi)

z += x .* y
"""
function blasfeo_svecmulacc(m, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_svecmulacc(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_svecmuldot(m, sx, xi, sy, yi, sz, zi)

z = x .* y, return sum(z) = x^T * y
"""
function blasfeo_svecmuldot(m, sx, xi, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_svecmuldot(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cfloat
end

"""
    blasfeo_sdot(m, sx, xi, sy, yi)

return x^T * y
"""
function blasfeo_sdot(m, sx, xi, sy, yi)
    @ccall blasfeo.blasfeo_sdot(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint)::Cfloat
end

"""
    blasfeo_srotg(a, b, c, s)

construct givens plane rotation
"""
function blasfeo_srotg(a, b, c, s)
    @ccall blasfeo.blasfeo_srotg(a::Cfloat, b::Cfloat, c::Ptr{Cfloat}, s::Ptr{Cfloat})::Cvoid
end

"""
    blasfeo_scolrot(m, sA, ai, aj0, aj1, c, s)

apply plane rotation [a b] [c -s; s; c] to the aj0 and aj1 columns of A at row index ai
"""
function blasfeo_scolrot(m, sA, ai, aj0, aj1, c, s)
    @ccall blasfeo.blasfeo_scolrot(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj0::Cint, aj1::Cint, c::Cfloat, s::Cfloat)::Cvoid
end

"""
    blasfeo_srowrot(m, sA, ai0, ai1, aj, c, s)

apply plane rotation [c s; -s c] [a; b] to the ai0 and ai1 rows of A at column index aj
"""
function blasfeo_srowrot(m, sA, ai0, ai1, aj, c, s)
    @ccall blasfeo.blasfeo_srowrot(m::Cint, sA::Ptr{blasfeo_smat}, ai0::Cint, ai1::Cint, aj::Cint, c::Cfloat, s::Cfloat)::Cvoid
end

"""
    blasfeo_sgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x
"""
function blasfeo_sgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_sgemv_n(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_sgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A' * x
"""
function blasfeo_sgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_sgemv_t(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_lnn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A ) * x, A (m)x(n)
"""
function blasfeo_strsv_lnn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_lnn_mn(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_ltn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A' ) * x, A (m)x(n)
"""
function blasfeo_strsv_ltn_mn(m, n, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_ltn_mn(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_lnn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A ) * x, A (m)x(m) lower, not_transposed
"""
function blasfeo_strsv_lnn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_lnn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_lnu(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A ) * x, A (m)x(m) lower, not_transposed, assuming unit diagonal
"""
function blasfeo_strsv_lnu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_lnu(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_ltn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A' ) * x, A (m)x(m) lower, transposed
"""
function blasfeo_strsv_ltn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_ltn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_ltu(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A' ) * x, A (m)x(m) lower, transposed, assuming unit diagonal
"""
function blasfeo_strsv_ltu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_ltu(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_unn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A' ) * x, A (m)x(m) upper, not_transposed
"""
function blasfeo_strsv_unn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_unn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strsv_utn(m, sA, ai, aj, sx, xi, sz, zi)

z <= inv( A' ) * x, A (m)x(m) upper, transposed
"""
function blasfeo_strsv_utn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strsv_utn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_lnn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A lower triangular
"""
function blasfeo_strmv_lnn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_lnn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_lnu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A lower triangular, assuming unit diagonal
"""
function blasfeo_strmv_lnu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_lnu(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_ltn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A' * x ; A lower triangular
"""
function blasfeo_strmv_ltn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_ltn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_ltu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A' * x ; A lower triangular, assuming unit diagonal
"""
function blasfeo_strmv_ltu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_ltu(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_unn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A upper triangular
"""
function blasfeo_strmv_unn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_unn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_unu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A * x ; A upper triangular, assuming unit diagonal
"""
function blasfeo_strmv_unu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_unu(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_utn(m, sA, ai, aj, sx, xi, sz, zi)

z <= A' * x ; A upper triangular
"""
function blasfeo_strmv_utn(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_utn(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_strmv_utu(m, sA, ai, aj, sx, xi, sz, zi)

z <= A' * x ; A upper triangular, assuming unit diagonal
"""
function blasfeo_strmv_utu(m, sA, ai, aj, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_strmv_utu(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_sgemv_nt(m, n, alpha_n, alpha_t, sA, ai, aj, sx_n, xi_n, sx_t, xi_t, beta_n, beta_t, sy_n, yi_n, sy_t, yi_t, sz_n, zi_n, sz_t, zi_t)

z_n <= beta_n * y_n + alpha_n * A  * x_n
z_t <= beta_t * y_t + alpha_t * A' * x_t
"""
function blasfeo_sgemv_nt(m, n, alpha_n, alpha_t, sA, ai, aj, sx_n, xi_n, sx_t, xi_t, beta_n, beta_t, sy_n, yi_n, sy_t, yi_t, sz_n, zi_n, sz_t, zi_t)
    @ccall blasfeo.blasfeo_sgemv_nt(m::Cint, n::Cint, alpha_n::Cfloat, alpha_t::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx_n::Ptr{blasfeo_svec}, xi_n::Cint, sx_t::Ptr{blasfeo_svec}, xi_t::Cint, beta_n::Cfloat, beta_t::Cfloat, sy_n::Ptr{blasfeo_svec}, yi_n::Cint, sy_t::Ptr{blasfeo_svec}, yi_t::Cint, sz_n::Ptr{blasfeo_svec}, zi_n::Cint, sz_t::Ptr{blasfeo_svec}, zi_t::Cint)::Cvoid
end

"""
    blasfeo_ssymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x, where A is symmetric and only the lower triangular patr of A is accessed
"""
function blasfeo_ssymv_l(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_ssymv_l(m::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

function blasfeo_ssymv_l_mn(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_ssymv_l_mn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_ssymv_u(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x, where A is symmetric and only the upper triangular patr of A is accessed
"""
function blasfeo_ssymv_u(m, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_ssymv_u(m::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_sger(m, n, alpha, sx, xi, sy, yi, sC, ci, cj, sD, di, dj)

D = C + alpha * x * y^T
"""
function blasfeo_sger(m, n, alpha, sx, xi, sy, yi, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sger(m::Cint, n::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgemv_d(m, alpha, sA, ai, sx, xi, beta, sy, yi, sz, zi)

z <= beta * y + alpha * A * x, A diagonal
"""
function blasfeo_sgemv_d(m, alpha, sA, ai, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_sgemv_d(m::Cint, alpha::Cfloat, sA::Ptr{blasfeo_svec}, ai::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_svec}, yi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_sgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B
"""
function blasfeo_sgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgemm_nn(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T
"""
function blasfeo_sgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgemm_nt(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B
"""
function blasfeo_sgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgemm_tn(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B
"""
function blasfeo_sgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgemm_tt(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T ; C, D lower triangular
"""
function blasfeo_ssyrk_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_ln(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_ssyrk_ln_mn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_ln_mn(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B ; C, D lower triangular
"""
function blasfeo_ssyrk_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_lt(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T ; C, D upper triangular
"""
function blasfeo_ssyrk_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_un(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B ; C, D upper triangular
"""
function blasfeo_ssyrk_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_ut(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A lower triangular
"""
function blasfeo_strmm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_llnn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A lower triangular assuming unit diagonal
"""
function blasfeo_strmm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_llnu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A lower triangular
"""
function blasfeo_strmm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_lltn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A lower triangular assuming unit diagonal
"""
function blasfeo_strmm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_lltu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A upper triangular
"""
function blasfeo_strmm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_lunn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A * B ; A upper triangular assuming unit diagonal
"""
function blasfeo_strmm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_lunu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A upper triangular
"""
function blasfeo_strmm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_lutn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^T * B ; A upper triangular assuming unit diagonal
"""
function blasfeo_strmm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_lutu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A lower triangular
"""
function blasfeo_strmm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_rlnn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A lower triangular assuming unit diagonal
"""
function blasfeo_strmm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_rlnu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A lower triangular
"""
function blasfeo_strmm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_rltn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A lower triangular assuming unit diagonal
"""
function blasfeo_strmm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_rltu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A upper triangular
"""
function blasfeo_strmm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_runn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A ; A upper triangular assuming unit diagonal
"""
function blasfeo_strmm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_runu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A upper triangular
"""
function blasfeo_strmm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_rutn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strmm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^T ; A upper triangular assuming unit diagonal
"""
function blasfeo_strmm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strmm_rutu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A lower triangular employint explicit inverse of diagonal
"""
function blasfeo_strsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_llnn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A lower triangular assuming unit diagonal
"""
function blasfeo_strsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_llnu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A lower triangular employint explicit inverse of diagonal
"""
function blasfeo_strsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_lltn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A lower triangular assuming unit diagonal
"""
function blasfeo_strsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_lltu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_strsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_lunn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-1} * B , with A upper triangular assuming unit diagonal
"""
function blasfeo_strsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_lunu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_strsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_lutn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * A^{-T} * B , with A upper triangular assuming unit diagonal
"""
function blasfeo_strsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_lutu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A lower triangular employing explicit inverse of diagonal
"""
function blasfeo_strsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_rlnn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A lower triangular assuming unit diagonal
"""
function blasfeo_strsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_rlnu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A lower triangular employing explicit inverse of diagonal
"""
function blasfeo_strsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_rltn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A lower triangular assuming unit diagonal
"""
function blasfeo_strsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_rltu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_strsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_runn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-1} , with A upper triangular assuming unit diagonal
"""
function blasfeo_strsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_runu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A upper triangular employing explicit inverse of diagonal
"""
function blasfeo_strsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_rutn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_strsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)

D <= alpha * B * A^{-T} , with A upper triangular assuming unit diagonal
"""
function blasfeo_strsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_strsm_rutu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyr2k_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T + alpha * B * A^T; C, D lower triangular
"""
function blasfeo_ssyr2k_ln(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyr2k_ln(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyr2k_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B + alpha * B^T * A; C, D lower triangular
"""
function blasfeo_ssyr2k_lt(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyr2k_lt(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyr2k_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A * B^T + alpha * B * A^T; C, D upper triangular
"""
function blasfeo_ssyr2k_un(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyr2k_un(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyr2k_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

D <= beta * C + alpha * A^T * B + alpha * B^T * A; C, D upper triangular
"""
function blasfeo_ssyr2k_ut(m, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyr2k_ut(m::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    sgemm_diag_left_ib(m, n, alpha, dA, pB, sdb, beta, pC, sdc, pD, sdd)

D <= alpha * A * B + beta * C, with A diagonal (stored as strvec)
"""
function sgemm_diag_left_ib(m, n, alpha, dA, pB, sdb, beta, pC, sdc, pD, sdd)
    @ccall blasfeo.sgemm_diag_left_ib(m::Cint, n::Cint, alpha::Cfloat, dA::Ptr{Cfloat}, pB::Ptr{Cfloat}, sdb::Cint, beta::Cfloat, pC::Ptr{Cfloat}, sdc::Cint, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function blasfeo_sgemm_dn(m, n, alpha, sA, ai, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgemm_dn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_svec}, ai::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgemm_nd(m, n, alpha, sA, ai, aj, sB, bi, beta, sC, ci, cj, sD, di, dj)

D <= alpha * A * B + beta * C, with B diagonal (stored as strvec)
"""
function blasfeo_sgemm_nd(m, n, alpha, sA, ai, aj, sB, bi, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgemm_nd(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_svec}, bi::Cint, beta::Cfloat, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_spotrf_l(m, sC, ci, cj, sD, di, dj)

D <= chol( C ) ; C, D lower triangular
"""
function blasfeo_spotrf_l(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_spotrf_l(m::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_spotrf_l_mn(m, n, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_spotrf_l_mn(m::Cint, n::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_spotrf_u(m, sC, ci, cj, sD, di, dj)

D <= chol( C ) ; C, D upper triangular
"""
function blasfeo_spotrf_u(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_spotrf_u(m::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ssyrk_spotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj)

D <= chol( C + A * B' ) ; C, D lower triangular
"""
function blasfeo_ssyrk_spotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_spotrf_ln(m::Cint, k::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_ssyrk_spotrf_ln_mn(m, n, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_ssyrk_spotrf_ln_mn(m::Cint, n::Cint, k::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_smat}, bi::Cint, bj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgetrf_np(m, n, sC, ci, cj, sD, di, dj)

D <= lu( C ) ; no pivoting
"""
function blasfeo_sgetrf_np(m, n, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_sgetrf_np(m::Cint, n::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_sgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv)

D <= lu( C ) ; row pivoting
"""
function blasfeo_sgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv)
    @ccall blasfeo.blasfeo_sgetrf_rp(m::Cint, n::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint, ipiv::Ptr{Cint})::Cvoid
end

"""
    blasfeo_sgeqrf(m, n, sC, ci, cj, sD, di, dj, work)

D <= qr( C )
"""
function blasfeo_sgeqrf(m, n, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_sgeqrf(m::Cint, n::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

function blasfeo_sgeqrf_worksize(m, n)
    @ccall blasfeo.blasfeo_sgeqrf_worksize(m::Cint, n::Cint)::Cint
end

"""
    blasfeo_sorglq_worksize(m, n, k)

D <= Q factor, where C is the output of the LQ factorization
"""
function blasfeo_sorglq_worksize(m, n, k)
    @ccall blasfeo.blasfeo_sorglq_worksize(m::Cint, n::Cint, k::Cint)::Cint
end

function blasfeo_sorglq(m, n, k, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_sorglq(m::Cint, n::Cint, k::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_sgelqf(m, n, sC, ci, cj, sD, di, dj, work)

D <= lq( C )
"""
function blasfeo_sgelqf(m, n, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_sgelqf(m::Cint, n::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

function blasfeo_sgelqf_worksize(m, n)
    @ccall blasfeo.blasfeo_sgelqf_worksize(m::Cint, n::Cint)::Cint
end

"""
    blasfeo_sgelqf_pd(m, n, sC, ci, cj, sD, di, dj, work)

D <= lq( C ), positive diagonal elements
"""
function blasfeo_sgelqf_pd(m, n, sC, ci, cj, sD, di, dj, work)
    @ccall blasfeo.blasfeo_sgelqf_pd(m::Cint, n::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_sgelqf_pd_la(m, n1, sL, li, lj, sA, ai, aj, work)

[L, A] <= lq( [L, A] ), positive diagonal elements, array of matrices, with
L lower triangular, of size (m)x(m)
A full, of size (m)x(n1)
"""
function blasfeo_sgelqf_pd_la(m, n1, sL, li, lj, sA, ai, aj, work)
    @ccall blasfeo.blasfeo_sgelqf_pd_la(m::Cint, n1::Cint, sL::Ptr{blasfeo_smat}, li::Cint, lj::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_sgelqf_pd_lla(m, n1, sL0, l0i, l0j, sL1, l1i, l1j, sA, ai, aj, work)

[L, L, A] <= lq( [L, L, A] ), positive diagonal elements, array of matrices, with:
L lower triangular, of size (m)x(m)
A full, of size (m)x(n1)
"""
function blasfeo_sgelqf_pd_lla(m, n1, sL0, l0i, l0j, sL1, l1i, l1j, sA, ai, aj, work)
    @ccall blasfeo.blasfeo_sgelqf_pd_lla(m::Cint, n1::Cint, sL0::Ptr{blasfeo_smat}, l0i::Cint, l0j::Cint, sL1::Ptr{blasfeo_smat}, l1i::Cint, l1j::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, work::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_cm_sgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)

BLAS 3
"""
function blasfeo_cm_sgemm_nn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_sgemm_nn(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_cm_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_sgemm_nt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_sgemm_nt(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_cm_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_sgemm_tn(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_sgemm_tn(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_cm_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_sgemm_tt(m, n, k, alpha, sA, ai, aj, sB, bi, bj, beta, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_sgemm_tt(m::Cint, n::Cint, k::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, beta::Cfloat, sC::Ptr{blasfeo_cm_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_llnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_llnn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_llnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_llnu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_lltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_lltn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_lltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_lltu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_lunn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_lunn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_lunu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_lunu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_lutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_lutn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_lutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_lutu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_rlnn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_rlnn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_rlnu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_rlnu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_rltn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_rltn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_rltu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_rltu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_runn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_runn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_runu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_runu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_rutn(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_rutn(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_strsm_rutu(m, n, alpha, sA, ai, aj, sB, bi, bj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_strsm_rutu(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_smat}, bi::Cint, bj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_cm_sgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)

BLAS 2
"""
function blasfeo_cm_sgemv_n(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_cm_sgemv_n(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_cm_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_cm_svec}, yi::Cint, sz::Ptr{blasfeo_cm_svec}, zi::Cint)::Cvoid
end

function blasfeo_cm_sgemv_t(m, n, alpha, sA, ai, aj, sx, xi, beta, sy, yi, sz, zi)
    @ccall blasfeo.blasfeo_cm_sgemv_t(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_cm_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_cm_svec}, xi::Cint, beta::Cfloat, sy::Ptr{blasfeo_cm_svec}, yi::Cint, sz::Ptr{blasfeo_cm_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_cm_spotrf_l(m, sC, ci, cj, sD, di, dj)

LAPACK
"""
function blasfeo_cm_spotrf_l(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_spotrf_l(m::Cint, sC::Ptr{blasfeo_cm_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_cm_spotrf_u(m, sC, ci, cj, sD, di, dj)
    @ccall blasfeo.blasfeo_cm_spotrf_u(m::Cint, sC::Ptr{blasfeo_cm_smat}, ci::Cint, cj::Cint, sD::Ptr{blasfeo_cm_smat}, di::Cint, dj::Cint)::Cvoid
end

function dtrcp_l_lib(m, alpha, offsetA, A, sda, offsetB, B, sdb)
    @ccall blasfeo.dtrcp_l_lib(m::Cint, alpha::Cdouble, offsetA::Cint, A::Ptr{Cdouble}, sda::Cint, offsetB::Cint, B::Ptr{Cdouble}, sdb::Cint)::Cvoid
end

function dgead_lib(m, n, alpha, offsetA, A, sda, offsetB, B, sdb)
    @ccall blasfeo.dgead_lib(m::Cint, n::Cint, alpha::Cdouble, offsetA::Cint, A::Ptr{Cdouble}, sda::Cint, offsetB::Cint, B::Ptr{Cdouble}, sdb::Cint)::Cvoid
end

"""
    ddiain_sqrt_lib(kmax, x, offset, pD, sdd)

TODO remove ???
"""
function ddiain_sqrt_lib(kmax, x, offset, pD, sdd)
    @ccall blasfeo.ddiain_sqrt_lib(kmax::Cint, x::Ptr{Cdouble}, offset::Cint, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

"""
    ddiareg_lib(kmax, reg, offset, pD, sdd)

TODO ddiaad1
"""
function ddiareg_lib(kmax, reg, offset, pD, sdd)
    @ccall blasfeo.ddiareg_lib(kmax::Cint, reg::Cdouble, offset::Cint, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function dgetr_lib(m, n, alpha, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.dgetr_lib(m::Cint, n::Cint, alpha::Cdouble, offsetA::Cint, pA::Ptr{Cdouble}, sda::Cint, offsetC::Cint, pC::Ptr{Cdouble}, sdc::Cint)::Cvoid
end

function dtrtr_l_lib(m, alpha, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.dtrtr_l_lib(m::Cint, alpha::Cdouble, offsetA::Cint, pA::Ptr{Cdouble}, sda::Cint, offsetC::Cint, pC::Ptr{Cdouble}, sdc::Cint)::Cvoid
end

function dtrtr_u_lib(m, alpha, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.dtrtr_u_lib(m::Cint, alpha::Cdouble, offsetA::Cint, pA::Ptr{Cdouble}, sda::Cint, offsetC::Cint, pC::Ptr{Cdouble}, sdc::Cint)::Cvoid
end

function ddiaex_lib(kmax, alpha, offset, pD, sdd, x)
    @ccall blasfeo.ddiaex_lib(kmax::Cint, alpha::Cdouble, offset::Cint, pD::Ptr{Cdouble}, sdd::Cint, x::Ptr{Cdouble})::Cvoid
end

function ddiaad_lib(kmax, alpha, x, offset, pD, sdd)
    @ccall blasfeo.ddiaad_lib(kmax::Cint, alpha::Cdouble, x::Ptr{Cdouble}, offset::Cint, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function ddiain_libsp(kmax, idx, alpha, x, pD, sdd)
    @ccall blasfeo.ddiain_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, x::Ptr{Cdouble}, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function ddiaex_libsp(kmax, idx, alpha, pD, sdd, x)
    @ccall blasfeo.ddiaex_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, pD::Ptr{Cdouble}, sdd::Cint, x::Ptr{Cdouble})::Cvoid
end

function ddiaad_libsp(kmax, idx, alpha, x, pD, sdd)
    @ccall blasfeo.ddiaad_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, x::Ptr{Cdouble}, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function ddiaadin_libsp(kmax, idx, alpha, x, y, pD, sdd)
    @ccall blasfeo.ddiaadin_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, x::Ptr{Cdouble}, y::Ptr{Cdouble}, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function drowin_lib(kmax, alpha, x, pD)
    @ccall blasfeo.drowin_lib(kmax::Cint, alpha::Cdouble, x::Ptr{Cdouble}, pD::Ptr{Cdouble})::Cvoid
end

function drowex_lib(kmax, alpha, pD, x)
    @ccall blasfeo.drowex_lib(kmax::Cint, alpha::Cdouble, pD::Ptr{Cdouble}, x::Ptr{Cdouble})::Cvoid
end

function drowad_lib(kmax, alpha, x, pD)
    @ccall blasfeo.drowad_lib(kmax::Cint, alpha::Cdouble, x::Ptr{Cdouble}, pD::Ptr{Cdouble})::Cvoid
end

function drowin_libsp(kmax, alpha, idx, x, pD)
    @ccall blasfeo.drowin_libsp(kmax::Cint, alpha::Cdouble, idx::Ptr{Cint}, x::Ptr{Cdouble}, pD::Ptr{Cdouble})::Cvoid
end

function drowad_libsp(kmax, idx, alpha, x, pD)
    @ccall blasfeo.drowad_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, x::Ptr{Cdouble}, pD::Ptr{Cdouble})::Cvoid
end

function drowadin_libsp(kmax, idx, alpha, x, y, pD)
    @ccall blasfeo.drowadin_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, x::Ptr{Cdouble}, y::Ptr{Cdouble}, pD::Ptr{Cdouble})::Cvoid
end

function dcolin_lib(kmax, x, offset, pD, sdd)
    @ccall blasfeo.dcolin_lib(kmax::Cint, x::Ptr{Cdouble}, offset::Cint, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function dcolad_lib(kmax, alpha, x, offset, pD, sdd)
    @ccall blasfeo.dcolad_lib(kmax::Cint, alpha::Cdouble, x::Ptr{Cdouble}, offset::Cint, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function dcolin_libsp(kmax, idx, x, pD, sdd)
    @ccall blasfeo.dcolin_libsp(kmax::Cint, idx::Ptr{Cint}, x::Ptr{Cdouble}, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function dcolad_libsp(kmax, alpha, idx, x, pD, sdd)
    @ccall blasfeo.dcolad_libsp(kmax::Cint, alpha::Cdouble, idx::Ptr{Cint}, x::Ptr{Cdouble}, pD::Ptr{Cdouble}, sdd::Cint)::Cvoid
end

function dcolsw_lib(kmax, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.dcolsw_lib(kmax::Cint, offsetA::Cint, pA::Ptr{Cdouble}, sda::Cint, offsetC::Cint, pC::Ptr{Cdouble}, sdc::Cint)::Cvoid
end

function dvecin_libsp(kmax, idx, x, y)
    @ccall blasfeo.dvecin_libsp(kmax::Cint, idx::Ptr{Cint}, x::Ptr{Cdouble}, y::Ptr{Cdouble})::Cvoid
end

function dvecad_libsp(kmax, idx, alpha, x, y)
    @ccall blasfeo.dvecad_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cdouble, x::Ptr{Cdouble}, y::Ptr{Cdouble})::Cvoid
end

"""
    blasfeo_memsize_dmat(m, n)

--- memory size calculations

returns the memory size (in bytes) needed for a dmat
"""
function blasfeo_memsize_dmat(m, n)
    @ccall blasfeo.blasfeo_memsize_dmat(m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_memsize_diag_dmat(m, n)

returns the memory size (in bytes) needed for the diagonal of a dmat
"""
function blasfeo_memsize_diag_dmat(m, n)
    @ccall blasfeo.blasfeo_memsize_diag_dmat(m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_memsize_dvec(m)

returns the memory size (in bytes) needed for a dvec
"""
function blasfeo_memsize_dvec(m)
    @ccall blasfeo.blasfeo_memsize_dvec(m::Cint)::Csize_t
end

"""
    blasfeo_create_dmat(m, n, sA, memory)

--- creation

create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_create_dmat(m, n, sA, memory)
    @ccall blasfeo.blasfeo_create_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_create_dvec(m, sA, memory)

create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_create_dvec(m, sA, memory)
    @ccall blasfeo.blasfeo_create_dvec(m::Cint, sA::Ptr{blasfeo_dvec}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_pack_dmat(m, n, A, lda, sB, bi, bj)

--- packing
pack the column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_dmat(m, n, A, lda, sB, bi, bj)
    @ccall blasfeo.blasfeo_pack_dmat(m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_pack_l_dmat(m, n, A, lda, sB, bi, bj)

pack the lower-triangular column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_l_dmat(m, n, A, lda, sB, bi, bj)
    @ccall blasfeo.blasfeo_pack_l_dmat(m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_pack_u_dmat(m, n, A, lda, sB, bi, bj)

pack the upper-triangular column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_u_dmat(m, n, A, lda, sB, bi, bj)
    @ccall blasfeo.blasfeo_pack_u_dmat(m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_pack_tran_dmat(m, n, A, lda, sB, bi, bj)

transpose and pack the column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_tran_dmat(m, n, A, lda, sB, bi, bj)
    @ccall blasfeo.blasfeo_pack_tran_dmat(m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_pack_dvec(m, x, incx, sy, yi)

pack the vector x (using increment incx) into the vector structure y (at offset yi)
"""
function blasfeo_pack_dvec(m, x, incx, sy, yi)
    @ccall blasfeo.blasfeo_pack_dvec(m::Cint, x::Ptr{Cdouble}, incx::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint)::Cvoid
end

"""
    blasfeo_unpack_dmat(m, n, sA, ai, aj, B, ldb)

unpack the matrix structure A (at row and col offsets ai and aj) into the column-major matrix B (with leading dimension ldb)
"""
function blasfeo_unpack_dmat(m, n, sA, ai, aj, B, ldb)
    @ccall blasfeo.blasfeo_unpack_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, B::Ptr{Cdouble}, ldb::Cint)::Cvoid
end

"""
    blasfeo_unpack_tran_dmat(m, n, sA, ai, aj, B, ldb)

transpose and unpack the matrix structure A (at row and col offsets ai and aj) into the column-major matrix B (with leading dimension ldb)
"""
function blasfeo_unpack_tran_dmat(m, n, sA, ai, aj, B, ldb)
    @ccall blasfeo.blasfeo_unpack_tran_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, B::Ptr{Cdouble}, ldb::Cint)::Cvoid
end

"""
    blasfeo_unpack_dvec(m, sx, xi, y, incy)

unpack the vector structure x (at offset xi) into the vector y (using increment incy)
"""
function blasfeo_unpack_dvec(m, sx, xi, y, incy)
    @ccall blasfeo.blasfeo_unpack_dvec(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, y::Ptr{Cdouble}, incy::Cint)::Cvoid
end

"""
    blasfeo_dgein1(a, sA, ai, aj)

ge
--- insert/extract

sA[ai, aj] <= a
"""
function blasfeo_dgein1(a, sA, ai, aj)
    @ccall blasfeo.blasfeo_dgein1(a::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_dgeex1(sA, ai, aj)

<= sA[ai, aj]
"""
function blasfeo_dgeex1(sA, ai, aj)
    @ccall blasfeo.blasfeo_dgeex1(sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cdouble
end

"""
    blasfeo_dgese(m, n, alpha, sA, ai, aj)

--- set
A <= alpha
"""
function blasfeo_dgese(m, n, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_dgese(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_dgecp(m, n, sA, ai, aj, sB, bi, bj)

--- copy / scale
B <= A
"""
function blasfeo_dgecp(m, n, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dgecp(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_dgesc(m, n, alpha, sA, ai, aj)

A <= alpha*A
"""
function blasfeo_dgesc(m, n, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_dgesc(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_dgecpsc(m, n, alpha, sA, ai, aj, sB, bi, bj)

B <= alpha*A
"""
function blasfeo_dgecpsc(m, n, alpha, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dgecpsc(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_dtrcp_l(m, sA, ai, aj, sB, bi, bj)

B <= A, A lower triangular
"""
function blasfeo_dtrcp_l(m, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dtrcp_l(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

function blasfeo_dtrcpsc_l(m, alpha, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dtrcpsc_l(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

function blasfeo_dtrsc_l(m, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_dtrsc_l(m::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_dgead(m, n, alpha, sA, ai, aj, sB, bi, bj)

--- sum
B <= B + alpha*A
"""
function blasfeo_dgead(m, n, alpha, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dgead(m::Cint, n::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_dvecad(m, alpha, sx, xi, sy, yi)

y <= y + alpha*x
"""
function blasfeo_dvecad(m, alpha, sx, xi, sy, yi)
    @ccall blasfeo.blasfeo_dvecad(m::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint)::Cvoid
end

"""
    blasfeo_dgetr(m, n, sA, ai, aj, sB, bi, bj)

--- traspositions
B <= A'
"""
function blasfeo_dgetr(m, n, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dgetr(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_dtrtr_l(m, sA, ai, aj, sB, bi, bj)

B <= A', A lower triangular
"""
function blasfeo_dtrtr_l(m, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dtrtr_l(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_dtrtr_u(m, sA, ai, aj, sB, bi, bj)

B <= A', A upper triangular
"""
function blasfeo_dtrtr_u(m, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_dtrtr_u(m::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    blasfeo_ddiare(kmax, alpha, sA, ai, aj)

dia
diag(A) += alpha
"""
function blasfeo_ddiare(kmax, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_ddiare(kmax::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_ddiain(kmax, alpha, sx, xi, sA, ai, aj)

diag(A) <= alpha*x
"""
function blasfeo_ddiain(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_ddiain(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_ddiain_sp(kmax, alpha, sx, xi, idx, sD, di, dj)

diag(A)[idx] <= alpha*x
"""
function blasfeo_ddiain_sp(kmax, alpha, sx, xi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_ddiain_sp(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ddiaex(kmax, alpha, sA, ai, aj, sx, xi)

x <= diag(A)
"""
function blasfeo_ddiaex(kmax, alpha, sA, ai, aj, sx, xi)
    @ccall blasfeo.blasfeo_ddiaex(kmax::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

"""
    blasfeo_ddiaex_sp(kmax, alpha, idx, sD, di, dj, sx, xi)

x <= diag(A)[idx]
"""
function blasfeo_ddiaex_sp(kmax, alpha, idx, sD, di, dj, sx, xi)
    @ccall blasfeo.blasfeo_ddiaex_sp(kmax::Cint, alpha::Cdouble, idx::Ptr{Cint}, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

"""
    blasfeo_ddiaad(kmax, alpha, sx, xi, sA, ai, aj)

diag(A) += alpha*x
"""
function blasfeo_ddiaad(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_ddiaad(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_ddiaad_sp(kmax, alpha, sx, xi, idx, sD, di, dj)

diag(A)[idx] += alpha*x
"""
function blasfeo_ddiaad_sp(kmax, alpha, sx, xi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_ddiaad_sp(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_ddiaadin_sp(kmax, alpha, sx, xi, sy, yi, idx, sD, di, dj)

diag(A)[idx] = y + alpha*x
"""
function blasfeo_ddiaadin_sp(kmax, alpha, sx, xi, sy, yi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_ddiaadin_sp(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_drowin(kmax, alpha, sx, xi, sA, ai, aj)

row
"""
function blasfeo_drowin(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_drowin(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_drowex(kmax, alpha, sA, ai, aj, sx, xi)
    @ccall blasfeo.blasfeo_drowex(kmax::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

function blasfeo_drowad(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_drowad(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_drowad_sp(kmax, alpha, sx, xi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_drowad_sp(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_dmat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_drowsw(kmax, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_drowsw(kmax::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_drowpe(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_drowpe(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_dmat})::Cvoid
end

function blasfeo_drowpei(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_drowpei(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_dmat})::Cvoid
end

"""
    blasfeo_dcolex(kmax, sA, ai, aj, sx, xi)

col
"""
function blasfeo_dcolex(kmax, sA, ai, aj, sx, xi)
    @ccall blasfeo.blasfeo_dcolex(kmax::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

function blasfeo_dcolin(kmax, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_dcolin(kmax::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_dcolad(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_dcolad(kmax::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_dcolsc(kmax, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_dcolsc(kmax::Cint, alpha::Cdouble, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_dcolsw(kmax, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_dcolsw(kmax::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_dmat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_dcolpe(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_dcolpe(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_dmat})::Cvoid
end

function blasfeo_dcolpei(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_dcolpei(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_dmat})::Cvoid
end

"""
    blasfeo_dvecse(m, alpha, sx, xi)

vec
a <= alpha
"""
function blasfeo_dvecse(m, alpha, sx, xi)
    @ccall blasfeo.blasfeo_dvecse(m::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

"""
    blasfeo_dvecin1(a, sx, xi)

sx[xi] <= a
"""
function blasfeo_dvecin1(a, sx, xi)
    @ccall blasfeo.blasfeo_dvecin1(a::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

"""
    blasfeo_dvecex1(sx, xi)

<= sx[xi]
"""
function blasfeo_dvecex1(sx, xi)
    @ccall blasfeo.blasfeo_dvecex1(sx::Ptr{blasfeo_dvec}, xi::Cint)::Cdouble
end

"""
    blasfeo_dveccp(m, sx, xi, sy, yi)

y <= x
"""
function blasfeo_dveccp(m, sx, xi, sy, yi)
    @ccall blasfeo.blasfeo_dveccp(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint)::Cvoid
end

"""
    blasfeo_dvecsc(m, alpha, sx, xi)

x <= alpha*x
"""
function blasfeo_dvecsc(m, alpha, sx, xi)
    @ccall blasfeo.blasfeo_dvecsc(m::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

"""
    blasfeo_dveccpsc(m, alpha, sx, xi, sy, yi)

y <= alpha*x
"""
function blasfeo_dveccpsc(m, alpha, sx, xi, sy, yi)
    @ccall blasfeo.blasfeo_dveccpsc(m::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, sy::Ptr{blasfeo_dvec}, yi::Cint)::Cvoid
end

"""
    blasfeo_dvecad_sp(m, alpha, sx, xi, idx, sz, zi)

z[idx] += alpha * x
"""
function blasfeo_dvecad_sp(m, alpha, sx, xi, idx, sz, zi)
    @ccall blasfeo.blasfeo_dvecad_sp(m::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, idx::Ptr{Cint}, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dvecin_sp(m, alpha, sx, xi, idx, sz, zi)

z[idx] <= alpha * x
"""
function blasfeo_dvecin_sp(m, alpha, sx, xi, idx, sz, zi)
    @ccall blasfeo.blasfeo_dvecin_sp(m::Cint, alpha::Cdouble, sx::Ptr{blasfeo_dvec}, xi::Cint, idx::Ptr{Cint}, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dvecex_sp(m, alpha, idx, sx, xi, sz, zi)

z <= alpha * x[idx]
"""
function blasfeo_dvecex_sp(m, alpha, idx, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dvecex_sp(m::Cint, alpha::Cdouble, idx::Ptr{Cint}, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

"""
    blasfeo_dvecexad_sp(m, alpha, idx, sx, xi, sz, zi)

z += alpha * x[idx]
"""
function blasfeo_dvecexad_sp(m, alpha, idx, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_dvecexad_sp(m::Cint, alpha::Cdouble, idx::Ptr{Cint}, sx::Ptr{blasfeo_dvec}, xi::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

function blasfeo_dveccl(m, sxm, xim, sx, xi, sxp, xip, sz, zi)
    @ccall blasfeo.blasfeo_dveccl(m::Cint, sxm::Ptr{blasfeo_dvec}, xim::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sxp::Ptr{blasfeo_dvec}, xip::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint)::Cvoid
end

function blasfeo_dveccl_mask(m, sxm, xim, sx, xi, sxp, xip, sz, zi, sm, mi)
    @ccall blasfeo.blasfeo_dveccl_mask(m::Cint, sxm::Ptr{blasfeo_dvec}, xim::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, sxp::Ptr{blasfeo_dvec}, xip::Cint, sz::Ptr{blasfeo_dvec}, zi::Cint, sm::Ptr{blasfeo_dvec}, mi::Cint)::Cvoid
end

function blasfeo_dvecze(m, sm, mi, sv, vi, se, ei)
    @ccall blasfeo.blasfeo_dvecze(m::Cint, sm::Ptr{blasfeo_dvec}, mi::Cint, sv::Ptr{blasfeo_dvec}, vi::Cint, se::Ptr{blasfeo_dvec}, ei::Cint)::Cvoid
end

function blasfeo_dvecnrm_inf(m, sx, xi, ptr_norm)
    @ccall blasfeo.blasfeo_dvecnrm_inf(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, ptr_norm::Ptr{Cdouble})::Cvoid
end

function blasfeo_dvecnrm_2(m, sx, xi, ptr_norm)
    @ccall blasfeo.blasfeo_dvecnrm_2(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, ptr_norm::Ptr{Cdouble})::Cvoid
end

function blasfeo_dvecnrm_1(m, sx, xi, ptr_norm)
    @ccall blasfeo.blasfeo_dvecnrm_1(m::Cint, sx::Ptr{blasfeo_dvec}, xi::Cint, ptr_norm::Ptr{Cdouble})::Cvoid
end

function blasfeo_dvecpe(kmax, ipiv, sx, xi)
    @ccall blasfeo.blasfeo_dvecpe(kmax::Cint, ipiv::Ptr{Cint}, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

function blasfeo_dvecpei(kmax, ipiv, sx, xi)
    @ccall blasfeo.blasfeo_dvecpei(kmax::Cint, ipiv::Ptr{Cint}, sx::Ptr{blasfeo_dvec}, xi::Cint)::Cvoid
end

"""
    blasfeo_pm_memsize_dmat(ps, m, n)

returns the memory size (in bytes) needed for a dmat
"""
function blasfeo_pm_memsize_dmat(ps, m, n)
    @ccall blasfeo.blasfeo_pm_memsize_dmat(ps::Cint, m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_pm_create_dmat(ps, m, n, sA, memory)

create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_pm_create_dmat(ps, m, n, sA, memory)
    @ccall blasfeo.blasfeo_pm_create_dmat(ps::Cint, m::Cint, n::Cint, sA::Ptr{blasfeo_pm_dmat}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_pm_print_dmat(m, n, sA, ai, aj)

print
"""
function blasfeo_pm_print_dmat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_pm_print_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_pm_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_cm_memsize_dmat(m, n)

returns the memory size (in bytes) needed for a dmat
"""
function blasfeo_cm_memsize_dmat(m, n)
    @ccall blasfeo.blasfeo_cm_memsize_dmat(m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_cm_create_dmat(m, n, sA, memory)

create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_cm_create_dmat(m, n, sA, memory)
    @ccall blasfeo.blasfeo_cm_create_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_pm_dmat}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_cm_dgetr(m, n, sA, ai, aj, sB, bi, bj)

aux
"""
function blasfeo_cm_dgetr(m, n, sA, ai, aj, sB, bi, bj)
    @ccall blasfeo.blasfeo_cm_dgetr(m::Cint, n::Cint, sA::Ptr{blasfeo_cm_dmat}, ai::Cint, aj::Cint, sB::Ptr{blasfeo_cm_dmat}, bi::Cint, bj::Cint)::Cvoid
end

"""
    strcp_l_lib(m, alpha, offsetA, A, sda, offsetB, B, sdb)

TODO remove
"""
function strcp_l_lib(m, alpha, offsetA, A, sda, offsetB, B, sdb)
    @ccall blasfeo.strcp_l_lib(m::Cint, alpha::Cfloat, offsetA::Cint, A::Ptr{Cfloat}, sda::Cint, offsetB::Cint, B::Ptr{Cfloat}, sdb::Cint)::Cvoid
end

function sgead_lib(m, n, alpha, offsetA, A, sda, offsetB, B, sdb)
    @ccall blasfeo.sgead_lib(m::Cint, n::Cint, alpha::Cfloat, offsetA::Cint, A::Ptr{Cfloat}, sda::Cint, offsetB::Cint, B::Ptr{Cfloat}, sdb::Cint)::Cvoid
end

function sgetr_lib(m, n, alpha, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.sgetr_lib(m::Cint, n::Cint, alpha::Cfloat, offsetA::Cint, pA::Ptr{Cfloat}, sda::Cint, offsetC::Cint, pC::Ptr{Cfloat}, sdc::Cint)::Cvoid
end

function strtr_l_lib(m, alpha, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.strtr_l_lib(m::Cint, alpha::Cfloat, offsetA::Cint, pA::Ptr{Cfloat}, sda::Cint, offsetC::Cint, pC::Ptr{Cfloat}, sdc::Cint)::Cvoid
end

function strtr_u_lib(m, alpha, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.strtr_u_lib(m::Cint, alpha::Cfloat, offsetA::Cint, pA::Ptr{Cfloat}, sda::Cint, offsetC::Cint, pC::Ptr{Cfloat}, sdc::Cint)::Cvoid
end

function sdiareg_lib(kmax, reg, offset, pD, sdd)
    @ccall blasfeo.sdiareg_lib(kmax::Cint, reg::Cfloat, offset::Cint, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function sdiain_sqrt_lib(kmax, x, offset, pD, sdd)
    @ccall blasfeo.sdiain_sqrt_lib(kmax::Cint, x::Ptr{Cfloat}, offset::Cint, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function sdiaex_lib(kmax, alpha, offset, pD, sdd, x)
    @ccall blasfeo.sdiaex_lib(kmax::Cint, alpha::Cfloat, offset::Cint, pD::Ptr{Cfloat}, sdd::Cint, x::Ptr{Cfloat})::Cvoid
end

function sdiaad_lib(kmax, alpha, x, offset, pD, sdd)
    @ccall blasfeo.sdiaad_lib(kmax::Cint, alpha::Cfloat, x::Ptr{Cfloat}, offset::Cint, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function sdiain_libsp(kmax, idx, alpha, x, pD, sdd)
    @ccall blasfeo.sdiain_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, x::Ptr{Cfloat}, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function sdiaex_libsp(kmax, idx, alpha, pD, sdd, x)
    @ccall blasfeo.sdiaex_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, pD::Ptr{Cfloat}, sdd::Cint, x::Ptr{Cfloat})::Cvoid
end

function sdiaad_libsp(kmax, idx, alpha, x, pD, sdd)
    @ccall blasfeo.sdiaad_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, x::Ptr{Cfloat}, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function sdiaadin_libsp(kmax, idx, alpha, x, y, pD, sdd)
    @ccall blasfeo.sdiaadin_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, x::Ptr{Cfloat}, y::Ptr{Cfloat}, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function srowin_lib(kmax, alpha, x, pD)
    @ccall blasfeo.srowin_lib(kmax::Cint, alpha::Cfloat, x::Ptr{Cfloat}, pD::Ptr{Cfloat})::Cvoid
end

function srowex_lib(kmax, alpha, pD, x)
    @ccall blasfeo.srowex_lib(kmax::Cint, alpha::Cfloat, pD::Ptr{Cfloat}, x::Ptr{Cfloat})::Cvoid
end

function srowad_lib(kmax, alpha, x, pD)
    @ccall blasfeo.srowad_lib(kmax::Cint, alpha::Cfloat, x::Ptr{Cfloat}, pD::Ptr{Cfloat})::Cvoid
end

function srowin_libsp(kmax, alpha, idx, x, pD)
    @ccall blasfeo.srowin_libsp(kmax::Cint, alpha::Cfloat, idx::Ptr{Cint}, x::Ptr{Cfloat}, pD::Ptr{Cfloat})::Cvoid
end

function srowad_libsp(kmax, idx, alpha, x, pD)
    @ccall blasfeo.srowad_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, x::Ptr{Cfloat}, pD::Ptr{Cfloat})::Cvoid
end

function srowadin_libsp(kmax, idx, alpha, x, y, pD)
    @ccall blasfeo.srowadin_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, x::Ptr{Cfloat}, y::Ptr{Cfloat}, pD::Ptr{Cfloat})::Cvoid
end

function srowsw_lib(kmax, pA, pC)
    @ccall blasfeo.srowsw_lib(kmax::Cint, pA::Ptr{Cfloat}, pC::Ptr{Cfloat})::Cvoid
end

function scolin_lib(kmax, x, offset, pD, sdd)
    @ccall blasfeo.scolin_lib(kmax::Cint, x::Ptr{Cfloat}, offset::Cint, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function scolad_lib(kmax, alpha, x, offset, pD, sdd)
    @ccall blasfeo.scolad_lib(kmax::Cint, alpha::Cfloat, x::Ptr{Cfloat}, offset::Cint, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function scolin_libsp(kmax, idx, x, pD, sdd)
    @ccall blasfeo.scolin_libsp(kmax::Cint, idx::Ptr{Cint}, x::Ptr{Cfloat}, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function scolad_libsp(kmax, alpha, idx, x, pD, sdd)
    @ccall blasfeo.scolad_libsp(kmax::Cint, alpha::Cfloat, idx::Ptr{Cint}, x::Ptr{Cfloat}, pD::Ptr{Cfloat}, sdd::Cint)::Cvoid
end

function scolsw_lib(kmax, offsetA, pA, sda, offsetC, pC, sdc)
    @ccall blasfeo.scolsw_lib(kmax::Cint, offsetA::Cint, pA::Ptr{Cfloat}, sda::Cint, offsetC::Cint, pC::Ptr{Cfloat}, sdc::Cint)::Cvoid
end

function svecin_libsp(kmax, idx, x, y)
    @ccall blasfeo.svecin_libsp(kmax::Cint, idx::Ptr{Cint}, x::Ptr{Cfloat}, y::Ptr{Cfloat})::Cvoid
end

function svecad_libsp(kmax, idx, alpha, x, y)
    @ccall blasfeo.svecad_libsp(kmax::Cint, idx::Ptr{Cint}, alpha::Cfloat, x::Ptr{Cfloat}, y::Ptr{Cfloat})::Cvoid
end

"""
    blasfeo_memsize_smat(m, n)

memory
returns the memory size (in bytes) needed for a smat
"""
function blasfeo_memsize_smat(m, n)
    @ccall blasfeo.blasfeo_memsize_smat(m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_memsize_diag_smat(m, n)

returns the memory size (in bytes) needed for the diagonal of a smat
"""
function blasfeo_memsize_diag_smat(m, n)
    @ccall blasfeo.blasfeo_memsize_diag_smat(m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_memsize_svec(m)

returns the memory size (in bytes) needed for a svec
"""
function blasfeo_memsize_svec(m)
    @ccall blasfeo.blasfeo_memsize_svec(m::Cint)::Csize_t
end

"""
    blasfeo_create_smat(m, n, sA, memory)

creation
create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_create_smat(m, n, sA, memory)
    @ccall blasfeo.blasfeo_create_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_create_svec(m, sA, memory)

create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_create_svec(m, sA, memory)
    @ccall blasfeo.blasfeo_create_svec(m::Cint, sA::Ptr{blasfeo_svec}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_pack_smat(m, n, A, lda, sA, ai, aj)

packing
pack the column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_smat(m, n, A, lda, sA, ai, aj)
    @ccall blasfeo.blasfeo_pack_smat(m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_pack_l_smat(m, n, A, lda, sA, ai, aj)

pack the lower-triangular column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_l_smat(m, n, A, lda, sA, ai, aj)
    @ccall blasfeo.blasfeo_pack_l_smat(m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_pack_u_smat(m, n, A, lda, sA, ai, aj)

pack the upper-triangular column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_u_smat(m, n, A, lda, sA, ai, aj)
    @ccall blasfeo.blasfeo_pack_u_smat(m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_pack_tran_smat(m, n, A, lda, sA, ai, aj)

transpose and pack the column-major matrix A (with leading dimension lda) into the matrix struct B (at row and col offsets bi and bj)
"""
function blasfeo_pack_tran_smat(m, n, A, lda, sA, ai, aj)
    @ccall blasfeo.blasfeo_pack_tran_smat(m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_pack_svec(m, x, xi, sa, ai)

pack the vector x (using increment incx) into the vector structure y (at offset yi)
"""
function blasfeo_pack_svec(m, x, xi, sa, ai)
    @ccall blasfeo.blasfeo_pack_svec(m::Cint, x::Ptr{Cfloat}, xi::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_unpack_smat(m, n, sA, ai, aj, A, lda)

unpack the matrix structure A (at row and col offsets ai and aj) into the column-major matrix B (with leading dimension ldb)
"""
function blasfeo_unpack_smat(m, n, sA, ai, aj, A, lda)
    @ccall blasfeo.blasfeo_unpack_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    blasfeo_unpack_tran_smat(m, n, sA, ai, aj, A, lda)

transpose and unpack the matrix structure A (at row and col offsets ai and aj) into the column-major matrix B (with leading dimension ldb)
"""
function blasfeo_unpack_tran_smat(m, n, sA, ai, aj, A, lda)
    @ccall blasfeo.blasfeo_unpack_tran_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    blasfeo_unpack_svec(m, sa, ai, x, xi)

unpack the vector structure x (at offset xi) into the vector y (using increment incy)
"""
function blasfeo_unpack_svec(m, sa, ai, x, xi)
    @ccall blasfeo.blasfeo_unpack_svec(m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint, x::Ptr{Cfloat}, xi::Cint)::Cvoid
end

"""
    blasfeo_sgese(m, n, alpha, sA, ai, aj)

ge
"""
function blasfeo_sgese(m, n, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_sgese(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_sgecpsc(m, n, alpha, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_sgecpsc(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_sgecp(m, n, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_sgecp(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_sgesc(m, n, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_sgesc(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_sgein1(a, sA, ai, aj)
    @ccall blasfeo.blasfeo_sgein1(a::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_sgeex1(sA, ai, aj)
    @ccall blasfeo.blasfeo_sgeex1(sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cfloat
end

function blasfeo_sgead(m, n, alpha, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_sgead(m::Cint, n::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_sgetr(m, n, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_sgetr(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

"""
    blasfeo_strcp_l(m, sA, ai, aj, sC, ci, cj)

tr
"""
function blasfeo_strcp_l(m, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_strcp_l(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_strtr_l(m, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_strtr_l(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_strtr_u(m, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_strtr_u(m::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

"""
    blasfeo_sdiare(kmax, alpha, sA, ai, aj)

dia
"""
function blasfeo_sdiare(kmax, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_sdiare(kmax::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_sdiaex(kmax, alpha, sA, ai, aj, sx, xi)
    @ccall blasfeo.blasfeo_sdiaex(kmax::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_sdiain(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_sdiain(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_sdiain_sp(kmax, alpha, sx, xi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_sdiain_sp(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_sdiaex_sp(kmax, alpha, idx, sD, di, dj, sx, xi)
    @ccall blasfeo.blasfeo_sdiaex_sp(kmax::Cint, alpha::Cfloat, idx::Ptr{Cint}, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_sdiaad(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_sdiaad(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_sdiaad_sp(kmax, alpha, sx, xi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_sdiaad_sp(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_sdiaadin_sp(kmax, alpha, sx, xi, sy, yi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_sdiaadin_sp(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sy::Ptr{blasfeo_svec}, yi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

"""
    blasfeo_srowin(kmax, alpha, sx, xi, sA, ai, aj)

row
"""
function blasfeo_srowin(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_srowin(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_srowex(kmax, alpha, sA, ai, aj, sx, xi)
    @ccall blasfeo.blasfeo_srowex(kmax::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_srowad(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_srowad(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_srowad_sp(kmax, alpha, sx, xi, idx, sD, di, dj)
    @ccall blasfeo.blasfeo_srowad_sp(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, idx::Ptr{Cint}, sD::Ptr{blasfeo_smat}, di::Cint, dj::Cint)::Cvoid
end

function blasfeo_srowsw(kmax, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_srowsw(kmax::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_srowpe(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_srowpe(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_smat})::Cvoid
end

function blasfeo_srowpei(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_srowpei(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_smat})::Cvoid
end

"""
    blasfeo_scolex(kmax, sA, ai, aj, sx, xi)

col
"""
function blasfeo_scolex(kmax, sA, ai, aj, sx, xi)
    @ccall blasfeo.blasfeo_scolex(kmax::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_scolin(kmax, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_scolin(kmax::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_scolad(kmax, alpha, sx, xi, sA, ai, aj)
    @ccall blasfeo.blasfeo_scolad(kmax::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_scolsc(kmax, alpha, sA, ai, aj)
    @ccall blasfeo.blasfeo_scolsc(kmax::Cint, alpha::Cfloat, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

function blasfeo_scolsw(kmax, sA, ai, aj, sC, ci, cj)
    @ccall blasfeo.blasfeo_scolsw(kmax::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint, sC::Ptr{blasfeo_smat}, ci::Cint, cj::Cint)::Cvoid
end

function blasfeo_scolpe(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_scolpe(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_smat})::Cvoid
end

function blasfeo_scolpei(kmax, ipiv, sA)
    @ccall blasfeo.blasfeo_scolpei(kmax::Cint, ipiv::Ptr{Cint}, sA::Ptr{blasfeo_smat})::Cvoid
end

"""
    blasfeo_svecse(m, alpha, sx, xi)

vec
"""
function blasfeo_svecse(m, alpha, sx, xi)
    @ccall blasfeo.blasfeo_svecse(m::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_sveccp(m, sa, ai, sc, ci)
    @ccall blasfeo.blasfeo_sveccp(m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint, sc::Ptr{blasfeo_svec}, ci::Cint)::Cvoid
end

function blasfeo_svecsc(m, alpha, sa, ai)
    @ccall blasfeo.blasfeo_svecsc(m::Cint, alpha::Cfloat, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

function blasfeo_sveccpsc(m, alpha, sa, ai, sc, ci)
    @ccall blasfeo.blasfeo_sveccpsc(m::Cint, alpha::Cfloat, sa::Ptr{blasfeo_svec}, ai::Cint, sc::Ptr{blasfeo_svec}, ci::Cint)::Cvoid
end

function blasfeo_svecad(m, alpha, sa, ai, sc, ci)
    @ccall blasfeo.blasfeo_svecad(m::Cint, alpha::Cfloat, sa::Ptr{blasfeo_svec}, ai::Cint, sc::Ptr{blasfeo_svec}, ci::Cint)::Cvoid
end

function blasfeo_svecin1(a, sx, xi)
    @ccall blasfeo.blasfeo_svecin1(a::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_svecex1(sx, xi)
    @ccall blasfeo.blasfeo_svecex1(sx::Ptr{blasfeo_svec}, xi::Cint)::Cfloat
end

function blasfeo_svecad_sp(m, alpha, sx, xi, idx, sz, zi)
    @ccall blasfeo.blasfeo_svecad_sp(m::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, idx::Ptr{Cint}, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

function blasfeo_svecin_sp(m, alpha, sx, xi, idx, sz, zi)
    @ccall blasfeo.blasfeo_svecin_sp(m::Cint, alpha::Cfloat, sx::Ptr{blasfeo_svec}, xi::Cint, idx::Ptr{Cint}, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

function blasfeo_svecex_sp(m, alpha, idx, sx, x, sz, zi)
    @ccall blasfeo.blasfeo_svecex_sp(m::Cint, alpha::Cfloat, idx::Ptr{Cint}, sx::Ptr{blasfeo_svec}, x::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

"""
    blasfeo_svecexad_sp(m, alpha, idx, sx, xi, sz, zi)

z += alpha * x[idx]
"""
function blasfeo_svecexad_sp(m, alpha, idx, sx, xi, sz, zi)
    @ccall blasfeo.blasfeo_svecexad_sp(m::Cint, alpha::Cdouble, idx::Ptr{Cint}, sx::Ptr{blasfeo_svec}, xi::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

function blasfeo_sveccl(m, sxm, xim, sx, xi, sxp, xip, sz, zi)
    @ccall blasfeo.blasfeo_sveccl(m::Cint, sxm::Ptr{blasfeo_svec}, xim::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sxp::Ptr{blasfeo_svec}, xip::Cint, sz::Ptr{blasfeo_svec}, zi::Cint)::Cvoid
end

function blasfeo_sveccl_mask(m, sxm, xim, sx, xi, sxp, xip, sz, zi, sm, mi)
    @ccall blasfeo.blasfeo_sveccl_mask(m::Cint, sxm::Ptr{blasfeo_svec}, xim::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, sxp::Ptr{blasfeo_svec}, xip::Cint, sz::Ptr{blasfeo_svec}, zi::Cint, sm::Ptr{blasfeo_svec}, mi::Cint)::Cvoid
end

function blasfeo_svecze(m, sm, mi, sv, vi, se, ei)
    @ccall blasfeo.blasfeo_svecze(m::Cint, sm::Ptr{blasfeo_svec}, mi::Cint, sv::Ptr{blasfeo_svec}, vi::Cint, se::Ptr{blasfeo_svec}, ei::Cint)::Cvoid
end

function blasfeo_svecnrm_inf(m, sx, xi, ptr_norm)
    @ccall blasfeo.blasfeo_svecnrm_inf(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, ptr_norm::Ptr{Cfloat})::Cvoid
end

function blasfeo_svecnrm_2(m, sx, xi, ptr_norm)
    @ccall blasfeo.blasfeo_svecnrm_2(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, ptr_norm::Ptr{Cfloat})::Cvoid
end

function blasfeo_svecnrm_1(m, sx, xi, ptr_norm)
    @ccall blasfeo.blasfeo_svecnrm_1(m::Cint, sx::Ptr{blasfeo_svec}, xi::Cint, ptr_norm::Ptr{Cfloat})::Cvoid
end

function blasfeo_svecpe(kmax, ipiv, sx, xi)
    @ccall blasfeo.blasfeo_svecpe(kmax::Cint, ipiv::Ptr{Cint}, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

function blasfeo_svecpei(kmax, ipiv, sx, xi)
    @ccall blasfeo.blasfeo_svecpei(kmax::Cint, ipiv::Ptr{Cint}, sx::Ptr{blasfeo_svec}, xi::Cint)::Cvoid
end

"""
    blasfeo_pm_memsize_smat(ps, m, n)

returns the memory size (in bytes) needed for a dmat
"""
function blasfeo_pm_memsize_smat(ps, m, n)
    @ccall blasfeo.blasfeo_pm_memsize_smat(ps::Cint, m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_pm_create_smat(ps, m, n, sA, memory)

create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_pm_create_smat(ps, m, n, sA, memory)
    @ccall blasfeo.blasfeo_pm_create_smat(ps::Cint, m::Cint, n::Cint, sA::Ptr{blasfeo_pm_smat}, memory::Ptr{Cvoid})::Cvoid
end

"""
    blasfeo_cm_memsize_smat(m, n)

returns the memory size (in bytes) needed for a dmat
"""
function blasfeo_cm_memsize_smat(m, n)
    @ccall blasfeo.blasfeo_cm_memsize_smat(m::Cint, n::Cint)::Csize_t
end

"""
    blasfeo_cm_create_smat(m, n, sA, memory)

create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
"""
function blasfeo_cm_create_smat(m, n, sA, memory)
    @ccall blasfeo.blasfeo_cm_create_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_pm_smat}, memory::Ptr{Cvoid})::Cvoid
end

"""
    d_zeros(pA, row, col)

dynamically allocate row*col doubles of memory and set accordingly a pointer to double; set allocated memory to zero
"""
function d_zeros(pA, row, col)
    @ccall blasfeo.d_zeros(pA::Ptr{Ptr{Cdouble}}, row::Cint, col::Cint)::Cvoid
end

"""
    d_zeros_align(pA, row, col)

dynamically allocate row*col doubles of memory aligned to 64-byte boundaries and set accordingly a pointer to double; set allocated memory to zero
"""
function d_zeros_align(pA, row, col)
    @ccall blasfeo.d_zeros_align(pA::Ptr{Ptr{Cdouble}}, row::Cint, col::Cint)::Cvoid
end

"""
    d_zeros_align_bytes(pA, size)

dynamically allocate size bytes of memory aligned to 64-byte boundaries and set accordingly a pointer to double; set allocated memory to zero
"""
function d_zeros_align_bytes(pA, size)
    @ccall blasfeo.d_zeros_align_bytes(pA::Ptr{Ptr{Cdouble}}, size::Cint)::Cvoid
end

"""
    d_free(pA)

free the memory allocated by d_zeros
"""
function d_free(pA)
    @ccall blasfeo.d_free(pA::Ptr{Cdouble})::Cvoid
end

"""
    d_free_align(pA)

free the memory allocated by d_zeros_align or d_zeros_align_bytes
"""
function d_free_align(pA)
    @ccall blasfeo.d_free_align(pA::Ptr{Cdouble})::Cvoid
end

"""
    d_print_mat(m, n, A, lda)

print a column-major matrix
"""
function d_print_mat(m, n, A, lda)
    @ccall blasfeo.d_print_mat(m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_exp_mat(m, n, A, lda)

print in exponential notation a column-major matrix
"""
function d_print_exp_mat(m, n, A, lda)
    @ccall blasfeo.d_print_exp_mat(m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_tran_mat(row, col, A, lda)

print the transposed of a column-major matrix
"""
function d_print_tran_mat(row, col, A, lda)
    @ccall blasfeo.d_print_tran_mat(row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_exp_tran_mat(row, col, A, lda)

print in exponential notation the transposed of a column-major matrix
"""
function d_print_exp_tran_mat(row, col, A, lda)
    @ccall blasfeo.d_print_exp_tran_mat(row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_to_file_mat(file, row, col, A, lda)

print to file a column-major matrix
"""
function d_print_to_file_mat(file, row, col, A, lda)
    @ccall blasfeo.d_print_to_file_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_to_file_exp_mat(file, row, col, A, lda)

print to file a column-major matrix in exponential format
"""
function d_print_to_file_exp_mat(file, row, col, A, lda)
    @ccall blasfeo.d_print_to_file_exp_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_tran_to_file_mat(file, row, col, A, lda)

print to file the transposed of a column-major matrix
"""
function d_print_tran_to_file_mat(file, row, col, A, lda)
    @ccall blasfeo.d_print_tran_to_file_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_tran_to_file_exp_mat(file, row, col, A, lda)

print to file the transposed of a column-major matrix in exponential format
"""
function d_print_tran_to_file_exp_mat(file, row, col, A, lda)
    @ccall blasfeo.d_print_tran_to_file_exp_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    d_print_to_string_mat(buf_out, row, col, A, lda)

print to string a column-major matrix
"""
function d_print_to_string_mat(buf_out, row, col, A, lda)
    @ccall blasfeo.d_print_to_string_mat(buf_out::Ptr{Ptr{Cchar}}, row::Cint, col::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

"""
    blasfeo_allocate_dmat(m, n, sA)

create a strmat for a matrix of size m*n by dynamically allocating memory
"""
function blasfeo_allocate_dmat(m, n, sA)
    @ccall blasfeo.blasfeo_allocate_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat})::Cvoid
end

"""
    blasfeo_allocate_dvec(m, sa)

create a strvec for a vector of size m by dynamically allocating memory
"""
function blasfeo_allocate_dvec(m, sa)
    @ccall blasfeo.blasfeo_allocate_dvec(m::Cint, sa::Ptr{blasfeo_dvec})::Cvoid
end

"""
    blasfeo_free_dmat(sA)

free the memory allocated by blasfeo_allocate_dmat
"""
function blasfeo_free_dmat(sA)
    @ccall blasfeo.blasfeo_free_dmat(sA::Ptr{blasfeo_dmat})::Cvoid
end

"""
    blasfeo_free_dvec(sa)

free the memory allocated by blasfeo_allocate_dvec
"""
function blasfeo_free_dvec(sa)
    @ccall blasfeo.blasfeo_free_dvec(sa::Ptr{blasfeo_dvec})::Cvoid
end

"""
    blasfeo_print_dmat(m, n, sA, ai, aj)

print a strmat
"""
function blasfeo_print_dmat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_exp_dmat(m, n, sA, ai, aj)

print in exponential notation a strmat
"""
function blasfeo_print_exp_dmat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_exp_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_tran_dmat(m, n, sA, ai, aj)

print the transposed of a strmat
"""
function blasfeo_print_tran_dmat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_tran_dmat(m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_dvec(m, sa, ai)

print a strvec
"""
function blasfeo_print_dvec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_dvec(m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_exp_dvec(m, sa, ai)

print in exponential notation a strvec
"""
function blasfeo_print_exp_dvec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_exp_dvec(m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_tran_dvec(m, sa, ai)

print the transposed of a strvec
"""
function blasfeo_print_tran_dvec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_tran_dvec(m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_exp_tran_dvec(m, sa, ai)

print in exponential notation the transposed of a strvec
"""
function blasfeo_print_exp_tran_dvec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_exp_tran_dvec(m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_dmat(file, m, n, sA, ai, aj)

print to file a strmat
"""
function blasfeo_print_to_file_dmat(file, m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_to_file_dmat(file::Ptr{Libc.FILE}, m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_exp_dmat(file, m, n, sA, ai, aj)

print to file a strmat in exponential format
"""
function blasfeo_print_to_file_exp_dmat(file, m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_to_file_exp_dmat(file::Ptr{Libc.FILE}, m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_dvec(file, m, sa, ai)

print to file a strvec
"""
function blasfeo_print_to_file_dvec(file, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_file_dvec(file::Ptr{Libc.FILE}, m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_tran_dvec(file, m, sa, ai)

print to file the transposed of a strvec
"""
function blasfeo_print_to_file_tran_dvec(file, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_file_tran_dvec(file::Ptr{Libc.FILE}, m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_string_dvec(buf_out, m, sa, ai)

print to string a strvec
"""
function blasfeo_print_to_string_dvec(buf_out, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_string_dvec(buf_out::Ptr{Ptr{Cchar}}, m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_string_dmat(buf_out, m, n, sA, ai, aj)

print to string a strmat
"""
function blasfeo_print_to_string_dmat(buf_out, m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_to_string_dmat(buf_out::Ptr{Ptr{Cchar}}, m::Cint, n::Cint, sA::Ptr{blasfeo_dmat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_to_string_tran_dvec(buf_out, m, sa, ai)

print to string the transposed of a strvec
"""
function blasfeo_print_to_string_tran_dvec(buf_out, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_string_tran_dvec(buf_out::Ptr{Ptr{Cchar}}, m::Cint, sa::Ptr{blasfeo_dvec}, ai::Cint)::Cvoid
end

"""
    s_zeros(pA, row, col)

dynamically allocate row*col floats of memory and set accordingly a pointer to float; set allocated memory to zero
"""
function s_zeros(pA, row, col)
    @ccall blasfeo.s_zeros(pA::Ptr{Ptr{Cfloat}}, row::Cint, col::Cint)::Cvoid
end

"""
    s_zeros_align(pA, row, col)

dynamically allocate row*col floats of memory aligned to 64-byte boundaries and set accordingly a pointer to float; set allocated memory to zero
"""
function s_zeros_align(pA, row, col)
    @ccall blasfeo.s_zeros_align(pA::Ptr{Ptr{Cfloat}}, row::Cint, col::Cint)::Cvoid
end

"""
    s_zeros_align_bytes(pA, size)

dynamically allocate size bytes of memory aligned to 64-byte boundaries and set accordingly a pointer to float; set allocated memory to zero
"""
function s_zeros_align_bytes(pA, size)
    @ccall blasfeo.s_zeros_align_bytes(pA::Ptr{Ptr{Cfloat}}, size::Cint)::Cvoid
end

"""
    s_free(pA)

free the memory allocated by d_zeros
"""
function s_free(pA)
    @ccall blasfeo.s_free(pA::Ptr{Cfloat})::Cvoid
end

"""
    s_free_align(pA)

free the memory allocated by d_zeros_align or d_zeros_align_bytes
"""
function s_free_align(pA)
    @ccall blasfeo.s_free_align(pA::Ptr{Cfloat})::Cvoid
end

"""
    s_print_mat(m, n, A, lda)

print a column-major matrix
"""
function s_print_mat(m, n, A, lda)
    @ccall blasfeo.s_print_mat(m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_exp_mat(m, n, A, lda)

print in exponential notation a column-major matrix
"""
function s_print_exp_mat(m, n, A, lda)
    @ccall blasfeo.s_print_exp_mat(m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_tran_mat(row, col, A, lda)

print the transposed of a column-major matrix
"""
function s_print_tran_mat(row, col, A, lda)
    @ccall blasfeo.s_print_tran_mat(row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_exp_tran_mat(row, col, A, lda)

print in exponential notation the transposed of a column-major matrix
"""
function s_print_exp_tran_mat(row, col, A, lda)
    @ccall blasfeo.s_print_exp_tran_mat(row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_to_file_mat(file, row, col, A, lda)

print to file a column-major matrix
"""
function s_print_to_file_mat(file, row, col, A, lda)
    @ccall blasfeo.s_print_to_file_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_to_file_exp_mat(file, row, col, A, lda)

print to file a column-major matrix in exponential format
"""
function s_print_to_file_exp_mat(file, row, col, A, lda)
    @ccall blasfeo.s_print_to_file_exp_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_tran_to_file_mat(file, row, col, A, lda)

print to file the transposed of a column-major matrix
"""
function s_print_tran_to_file_mat(file, row, col, A, lda)
    @ccall blasfeo.s_print_tran_to_file_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_tran_to_file_exp_mat(file, row, col, A, lda)

print to file the transposed of a column-major matrix in exponential format
"""
function s_print_tran_to_file_exp_mat(file, row, col, A, lda)
    @ccall blasfeo.s_print_tran_to_file_exp_mat(file::Ptr{Libc.FILE}, row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    s_print_to_string_mat(buf_out, row, col, A, lda)

print to string a column-major matrix
"""
function s_print_to_string_mat(buf_out, row, col, A, lda)
    @ccall blasfeo.s_print_to_string_mat(buf_out::Ptr{Ptr{Cchar}}, row::Cint, col::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

"""
    blasfeo_allocate_smat(m, n, sA)

create a strmat for a matrix of size m*n by dynamically allocating memory
"""
function blasfeo_allocate_smat(m, n, sA)
    @ccall blasfeo.blasfeo_allocate_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat})::Cvoid
end

"""
    blasfeo_allocate_svec(m, sa)

create a strvec for a vector of size m by dynamically allocating memory
"""
function blasfeo_allocate_svec(m, sa)
    @ccall blasfeo.blasfeo_allocate_svec(m::Cint, sa::Ptr{blasfeo_svec})::Cvoid
end

"""
    blasfeo_free_smat(sA)

free the memory allocated by blasfeo_allocate_dmat
"""
function blasfeo_free_smat(sA)
    @ccall blasfeo.blasfeo_free_smat(sA::Ptr{blasfeo_smat})::Cvoid
end

"""
    blasfeo_free_svec(sa)

free the memory allocated by blasfeo_allocate_dvec
"""
function blasfeo_free_svec(sa)
    @ccall blasfeo.blasfeo_free_svec(sa::Ptr{blasfeo_svec})::Cvoid
end

"""
    blasfeo_print_smat(m, n, sA, ai, aj)

print a strmat
"""
function blasfeo_print_smat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_exp_smat(m, n, sA, ai, aj)

print in exponential notation a strmat
"""
function blasfeo_print_exp_smat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_exp_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_tran_smat(m, n, sA, ai, aj)

print the transpose of a strmat
"""
function blasfeo_print_tran_smat(m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_tran_smat(m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_svec(m, sa, ai)

print a strvec
"""
function blasfeo_print_svec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_svec(m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_exp_svec(m, sa, ai)

print in exponential notation a strvec
"""
function blasfeo_print_exp_svec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_exp_svec(m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_tran_svec(m, sa, ai)

print the transposed of a strvec
"""
function blasfeo_print_tran_svec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_tran_svec(m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_exp_tran_svec(m, sa, ai)

print in exponential notation the transposed of a strvec
"""
function blasfeo_print_exp_tran_svec(m, sa, ai)
    @ccall blasfeo.blasfeo_print_exp_tran_svec(m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_smat(file, m, n, sA, ai, aj)

print to file a strmat
"""
function blasfeo_print_to_file_smat(file, m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_to_file_smat(file::Ptr{Libc.FILE}, m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_exp_smat(file, m, n, sA, ai, aj)

print to file a strmat in exponential format
"""
function blasfeo_print_to_file_exp_smat(file, m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_to_file_exp_smat(file::Ptr{Libc.FILE}, m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_svec(file, m, sa, ai)

print to file a strvec
"""
function blasfeo_print_to_file_svec(file, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_file_svec(file::Ptr{Libc.FILE}, m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_file_tran_svec(file, m, sa, ai)

print to file the transposed of a strvec
"""
function blasfeo_print_to_file_tran_svec(file, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_file_tran_svec(file::Ptr{Libc.FILE}, m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_string_smat(buf_out, m, n, sA, ai, aj)

print to string a strmat
"""
function blasfeo_print_to_string_smat(buf_out, m, n, sA, ai, aj)
    @ccall blasfeo.blasfeo_print_to_string_smat(buf_out::Ptr{Ptr{Cchar}}, m::Cint, n::Cint, sA::Ptr{blasfeo_smat}, ai::Cint, aj::Cint)::Cvoid
end

"""
    blasfeo_print_to_string_svec(buf_out, m, sa, ai)

print to string a strvec
"""
function blasfeo_print_to_string_svec(buf_out, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_string_svec(buf_out::Ptr{Ptr{Cchar}}, m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

"""
    blasfeo_print_to_string_tran_svec(buf_out, m, sa, ai)

print to string the transposed of a strvec
"""
function blasfeo_print_to_string_tran_svec(buf_out, m, sa, ai)
    @ccall blasfeo.blasfeo_print_to_string_tran_svec(buf_out::Ptr{Ptr{Cchar}}, m::Cint, sa::Ptr{blasfeo_svec}, ai::Cint)::Cvoid
end

const D_EL_SIZE = 8

const S_EL_SIZE = 4

const CACHE_LINE_SIZE = 64

const L1_CACHE_SIZE = 32 * 1024

const L2_CACHE_SIZE = 256 * 1024

const LLC_CACHE_SIZE = 6 * 1024 * 1024

const D_PS = 4

const D_PLD = 4

const D_M_KERNEL = 12

const D_N_KERNEL = 8

const D_KC = 256

const D_NC = 64

const D_MC = 1500

const S_PS = 8

const S_PLD = 4

const S_M_KERNEL = 24

const S_N_KERNEL = 8

const S_KC = 256

const S_NC = 144

const S_MC = 3000

const D_CACHE_LINE_EL = CACHE_LINE_SIZE ÷ D_EL_SIZE

const D_L1_CACHE_EL = L1_CACHE_SIZE ÷ D_EL_SIZE

const D_L2_CACHE_EL = L2_CACHE_SIZE ÷ D_EL_SIZE

const D_LLC_CACHE_EL = LLC_CACHE_SIZE ÷ D_EL_SIZE

const S_CACHE_LINE_EL = CACHE_LINE_SIZE ÷ S_EL_SIZE

const S_L1_CACHE_EL = L1_CACHE_SIZE ÷ S_EL_SIZE

const S_L2_CACHE_EL = L2_CACHE_SIZE ÷ S_EL_SIZE

const S_LLC_CACHE_EL = LLC_CACHE_SIZE ÷ S_EL_SIZE

# exports
const PREFIXES = ["blasfeo_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
