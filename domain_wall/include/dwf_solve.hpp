#pragma once

#include "cg_solver.hpp"
#include "dwf_eo.hpp"
#include "dwf_operator.hpp"
#include "parameters.hpp"
#include "types.hpp"

// Single entry point for measurement solves of D x = source. Uses the
// even-odd preconditioned path (dwf.even_odd = 1, the default) when the
// lattice supports a checkerboard, otherwise the original CGNR on D^dag D.
inline CgResult propagator_solve(
    const Gauge& theta,
    const VectorC& source,
    double rtol,
    int maxiter)
{
    if (even_odd_precond && Nt % 2 == 0 && Nx % 2 == 0)
        return solve_dirac_eo(theta, source, rtol, maxiter);

    const auto apply_M = [&](const VectorC& v)
    {
        return apply_D_dagger_D(theta, v);
    };
    const VectorC rhs = apply_D_dagger(theta, source);
    return conjugate_gradient(apply_M, rhs, nullptr, rtol, maxiter);
}
