#pragma once

#include "cg_solver.hpp"
#include "dwf_operator.hpp"
#include "indexing.hpp"
#include "parameters.hpp"
#include "spin.hpp"
#include "types.hpp"

#include <stdexcept>
#include <vector>

// Even-odd (checkerboard) preconditioned solver for D psi = b.
//
// The domain-wall operator splits as D = A + H: H is the spacetime
// hopping, which only connects sites of opposite (t + x) parity, and A
// collects the (3 - M5) diagonal plus every fifth-dimension coupling.
// A is spacetime-local and gauge-independent: per spin component it is
// one constant N5 x N5 bidiagonal matrix with an mf corner, so both A
// and A^-1 apply in O(N5) per site (substitution plus a Sherman-
// Morrison corner correction). Eliminating the odd sites leaves the
// Schur system on the evens,
//
//   K x_e = b_e - H_eo A^-1 b_o,   K = A - H_eo A^-1 H_oe,
//   x_o = A^-1 (b_o - H_oe x_e),
//
// solved with the same CGNR scheme as the unpreconditioned path. K is
// markedly better conditioned than D and one K application costs about
// half a D^dag D iteration, which is where the speedup comes from.

// The spin-0 block is lower bidiagonal, b on the diagonal, -1 below,
// +mf in the (0, N5-1) corner; the spin-1 block is its transpose. A
// dagger therefore swaps the two spin structures.

inline const std::vector<double>& fifth_inverse_powers()
{
    // binv[s] = (3 - M5)^-(s+1), the T^-1 e_corner column used by the
    // Sherman-Morrison correction.
    static const std::vector<double> powers = []
    {
        std::vector<double> result(N5);
        const double inverse = 1.0 / (3.0 - M5);
        double value = inverse;
        for (int s = 0; s < N5; ++s)
        {
            result[s] = value;
            value *= inverse;
        }
        return result;
    }();
    return powers;
}

// out(sites of `parity`) = A in (forward = true) or A^-1 in, acting on
// the s index; `lower_for_spin0` selects which spin gets the lower-
// bidiagonal block (swapped for the daggered operator).
inline void apply_fifth_parity(
    const VectorC& in,
    VectorC& out,
    int parity,
    bool forward,
    bool lower_for_spin0)
{
    const double diagonal = 3.0 - M5;
    const std::vector<double>& binv = fifth_inverse_powers();
    const double corner_denominator = 1.0 + mf * binv[N5 - 1];
    const int stride = Nt * Nx * Ns;

#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < Nt; ++t)
    {
        for (int half = 0; half < Nx / 2; ++half)
        {
            const int x = 2 * half + (parity + t) % 2;
            for (int spin = 0; spin < Ns; ++spin)
            {
                const int base = fermion_index(0, t, x, spin);
                const bool lower = (spin == 0) == lower_for_spin0;

                if (forward && lower)
                {
                    out[base] = diagonal * in[base]
                        + mf * in[base + (N5 - 1) * stride];
                    for (int s = 1; s < N5; ++s)
                        out[base + s * stride] =
                            diagonal * in[base + s * stride]
                            - in[base + (s - 1) * stride];
                }
                else if (forward)
                {
                    for (int s = 0; s < N5 - 1; ++s)
                        out[base + s * stride] =
                            diagonal * in[base + s * stride]
                            - in[base + (s + 1) * stride];
                    out[base + (N5 - 1) * stride] =
                        diagonal * in[base + (N5 - 1) * stride]
                        + mf * in[base];
                }
                else if (lower)
                {
                    // Forward substitution against the bidiagonal part,
                    // then remove the mf-corner via Sherman-Morrison.
                    Complex previous =
                        in[base] / diagonal;
                    out[base] = previous;
                    for (int s = 1; s < N5; ++s)
                    {
                        previous = (in[base + s * stride] + previous)
                            / diagonal;
                        out[base + s * stride] = previous;
                    }
                    const Complex alpha =
                        mf * out[base + (N5 - 1) * stride]
                        / corner_denominator;
                    for (int s = 0; s < N5; ++s)
                        out[base + s * stride] -= alpha * binv[s];
                }
                else
                {
                    Complex previous =
                        in[base + (N5 - 1) * stride] / diagonal;
                    out[base + (N5 - 1) * stride] = previous;
                    for (int s = N5 - 2; s >= 0; --s)
                    {
                        previous = (in[base + s * stride] + previous)
                            / diagonal;
                        out[base + s * stride] = previous;
                    }
                    const Complex alpha =
                        mf * out[base] / corner_denominator;
                    for (int s = 0; s < N5; ++s)
                        out[base + s * stride] -=
                            alpha * binv[N5 - 1 - s];
                }
            }
        }
    }
}

// out(sites of out_parity) = [H in](same site): the spacetime hopping of
// apply_D (or apply_D_dagger) restricted to one output parity, with the
// diagonal and fifth-dimension terms excluded.
inline void hop_parity(
    const std::vector<Complex>& links,
    const VectorC& in,
    VectorC& out,
    int out_parity,
    bool dagger)
{
    const Eigen::Matrix2cd I = identity2();
    const Eigen::Matrix2cd s1 = sigma1();
    const Eigen::Matrix2cd s2 = sigma2();
    // Matching apply_D / apply_D_dagger term for term.
    const double flip = dagger ? -1.0 : 1.0;
    const Eigen::Matrix2cd proj_tp = I - flip * s1;
    const Eigen::Matrix2cd proj_tm = I + flip * s1;
    const Eigen::Matrix2cd proj_xp = I - flip * s2;
    const Eigen::Matrix2cd proj_xm = I + flip * s2;

#pragma omp parallel for collapse(2) schedule(static)
    for (int s = 0; s < N5; ++s)
    {
        for (int t = 0; t < Nt; ++t)
        {
            const int tp = plus_periodic(t, Nt);
            const int tm = minus_periodic(t, Nt);
            const double sign_tp = (t == Nt - 1) ? -1.0 : 1.0;
            const double sign_tm = (t == 0) ? -1.0 : 1.0;

            for (int half = 0; half < Nx / 2; ++half)
            {
                const int x = 2 * half + (out_parity + t) % 2;
                const int xp = plus_periodic(x, Nx);
                const int xm = minus_periodic(x, Nx);

                Eigen::Vector2cd v_tp;
                Eigen::Vector2cd v_tm;
                Eigen::Vector2cd v_xp;
                Eigen::Vector2cd v_xm;
                for (int a = 0; a < Ns; ++a)
                {
                    v_tp[a] = sign_tp * in[fermion_index(s, tp, x, a)];
                    v_tm[a] = sign_tm * in[fermion_index(s, tm, x, a)];
                    v_xp[a] = in[fermion_index(s, t, xp, a)];
                    v_xm[a] = in[fermion_index(s, t, xm, a)];
                }

                const Eigen::Vector2cd term_tp = proj_tp * v_tp;
                const Eigen::Vector2cd term_tm = proj_tm * v_tm;
                const Eigen::Vector2cd term_xp = proj_xp * v_xp;
                const Eigen::Vector2cd term_xm = proj_xm * v_xm;

                const Complex Ut = links[gauge_index(0, t, x)];
                const Complex Ut_dag =
                    std::conj(links[gauge_index(0, tm, x)]);
                const Complex Ux = links[gauge_index(1, t, x)];
                const Complex Ux_dag =
                    std::conj(links[gauge_index(1, t, xm)]);

                for (int a = 0; a < Ns; ++a)
                {
                    const int row = fermion_index(s, t, x, a);
                    out[row] = -0.5 * (Ut * term_tp[a]
                                       + Ut_dag * term_tm[a]
                                       + Ux * term_xp[a]
                                       + Ux_dag * term_xm[a]);
                }
            }
        }
    }
}

// Scratch vectors reused across all Schur applications of one solve.
// Their stale entries are never read: every stage assigns the parity
// sites it is responsible for before anything reads them.
struct EoWorkspace
{
    VectorC odd = VectorC(Ndof);
    VectorC odd_solved = VectorC(Ndof);
    VectorC hopped = VectorC(Ndof);
};

inline VectorC apply_schur(
    const std::vector<Complex>& links,
    const VectorC& even,
    EoWorkspace& workspace,
    bool dagger)
{
    // K = A - H_eo A^-1 H_oe (daggered factors when dagger is set;
    // the A blocks' dagger swaps the two spin structures).
    hop_parity(links, even, workspace.odd, 1, dagger);
    apply_fifth_parity(
        workspace.odd, workspace.odd_solved, 1, false, !dagger);
    hop_parity(links, workspace.odd_solved, workspace.hopped, 0, dagger);

    VectorC out = VectorC::Zero(Ndof);
    apply_fifth_parity(even, out, 0, true, !dagger);
    for (int t = 0; t < Nt; ++t)
        for (int x = t % 2; x < Nx; x += 2)
            for (int s = 0; s < N5; ++s)
                for (int a = 0; a < Ns; ++a)
                {
                    const int i = fermion_index(s, t, x, a);
                    out[i] -= workspace.hopped[i];
                }
    return out;
}

inline VectorC restrict_parity(const VectorC& v, int parity)
{
    VectorC out = VectorC::Zero(Ndof);
    for (int t = 0; t < Nt; ++t)
        for (int x = (parity + t) % 2; x < Nx; x += 2)
            for (int s = 0; s < N5; ++s)
                for (int a = 0; a < Ns; ++a)
                {
                    const int i = fermion_index(s, t, x, a);
                    out[i] = v[i];
                }
    return out;
}

// Full solve of D x = b through the even-odd Schur complement, with the
// same CGNR stopping rule (normal-equation relative residual) as the
// unpreconditioned path.
inline CgResult solve_dirac_eo(
    const Gauge& theta,
    const VectorC& b,
    double rtol,
    int maxiter)
{
    if (Nt % 2 != 0 || Nx % 2 != 0)
        throw std::runtime_error(
            "Even-odd preconditioning requires even Nt and Nx");

    const std::vector<Complex> links = link_table(theta);
    EoWorkspace workspace;

    const VectorC b_even = restrict_parity(b, 0);
    const VectorC b_odd = restrict_parity(b, 1);

    apply_fifth_parity(b_odd, workspace.odd_solved, 1, false, true);
    hop_parity(links, workspace.odd_solved, workspace.hopped, 0, false);
    VectorC reduced = b_even;
    for (int t = 0; t < Nt; ++t)
        for (int x = t % 2; x < Nx; x += 2)
            for (int s = 0; s < N5; ++s)
                for (int a = 0; a < Ns; ++a)
                {
                    const int i = fermion_index(s, t, x, a);
                    reduced[i] -= workspace.hopped[i];
                }

    const auto apply_normal = [&](const VectorC& v)
    {
        const VectorC forward =
            apply_schur(links, v, workspace, false);
        return apply_schur(links, forward, workspace, true);
    };
    const CgResult inner = conjugate_gradient(
        apply_normal, apply_schur(links, reduced, workspace, true),
        nullptr, rtol, maxiter);

    // Reconstruct the odd half: x_o = A^-1 (b_o - H_oe x_e).
    hop_parity(links, inner.x, workspace.hopped, 1, false);
    VectorC remainder = b_odd;
    for (int t = 0; t < Nt; ++t)
        for (int x = (1 + t) % 2; x < Nx; x += 2)
            for (int s = 0; s < N5; ++s)
                for (int a = 0; a < Ns; ++a)
                {
                    const int i = fermion_index(s, t, x, a);
                    remainder[i] -= workspace.hopped[i];
                }
    VectorC x = inner.x;
    apply_fifth_parity(remainder, x, 1, false, true);

    return {std::move(x), inner.iterations,
            inner.relative_residual, inner.converged};
}
