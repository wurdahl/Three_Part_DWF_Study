#pragma once

#include "types.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>

struct CgResult
{
    VectorC x;
    int iterations;
    double relative_residual;
    bool converged;
};

inline CgResult conjugate_gradient(
    const std::function<VectorC(const VectorC&)>& apply_A,
    const VectorC& b,
    const VectorC* initial_guess,
    double rtol,
    int maxiter)
{
    VectorC x =
        initial_guess ? *initial_guess : VectorC::Zero(b.size());

    VectorC r = b - apply_A(x);
    VectorC p = r;

    const double bnorm = b.norm();
    if (bnorm == 0.0)
        return {x, 0, 0.0, true};

    double rr_old = std::real(r.dot(r));
    double relative_residual = std::sqrt(rr_old) / bnorm;

    if (relative_residual <= rtol)
        return {x, 0, relative_residual, true};

    for (int iteration = 1; iteration <= maxiter; ++iteration)
    {
        const VectorC Ap = apply_A(p);
        const Complex denominator = p.dot(Ap);

        if (std::abs(denominator) == 0.0)
            throw std::runtime_error("CG encountered a zero denominator");

        const double alpha = rr_old / std::real(denominator);
        x += alpha * p;
        r -= alpha * Ap;

        const double rr_new = std::real(r.dot(r));
        relative_residual = std::sqrt(rr_new) / bnorm;

        if (relative_residual <= rtol)
            return {x, iteration, relative_residual, true};

        const double beta_cg = rr_new / rr_old;
        p = r + beta_cg * p;
        rr_old = rr_new;
    }

    return {x, maxiter, relative_residual, false};
}
