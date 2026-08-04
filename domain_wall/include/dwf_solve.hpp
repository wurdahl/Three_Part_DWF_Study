#pragma once

#include "cg_solver.hpp"
#include "dwf_eo.hpp"
#include "dwf_operator.hpp"
#include "parameters.hpp"
#include "types.hpp"

#ifdef DWF_GPU_ANALYSIS
#include "gpu_propagator.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#endif

#ifdef DWF_GPU_ANALYSIS
// One solver, and so one CUDA stream, per OpenMP thread: the analyzer
// parallelizes over configurations, and independent solves then overlap
// on the device instead of serializing behind a single stream. The gauge
// field is uploaded only when it actually changes, so the 16 (or 32)
// solves of one configuration share a single link build.
inline CgResult gpu_propagator_solve(
    const Gauge& theta,
    const VectorC& source,
    double rtol,
    int maxiter)
{
    static thread_local std::unique_ptr<GpuPropagator> solver;
    static thread_local Gauge cached;

    if (!solver)
    {
        solver.reset(new GpuPropagator(Nt, Nx, N5, M5, mf));
        cached.assign(
            Ngauge, std::numeric_limits<double>::quiet_NaN());
    }
    if (!std::equal(theta.begin(), theta.end(), cached.begin()))
    {
        solver->set_gauge(theta.data());
        cached = theta;
    }

    VectorC solution(Ndof);
    const GpuPropagator::Result result =
        solver->solve(source.data(), solution.data(), rtol, maxiter);
    return {std::move(solution), result.iterations,
            result.relative_residual, result.converged};
}
#endif

// Single entry point for measurement solves of D x = source. Uses the
// even-odd preconditioned path (dwf.even_odd = 1, the default) when the
// lattice supports a checkerboard, otherwise the original CGNR on D^dag D.
inline CgResult propagator_solve(
    const Gauge& theta,
    const VectorC& source,
    double rtol,
    int maxiter)
{
#ifdef DWF_GPU_ANALYSIS
    if (gpu_analysis)
        return gpu_propagator_solve(theta, source, rtol, maxiter);
#endif

    if (even_odd_precond && Nt % 2 == 0 && Nx % 2 == 0)
        return solve_dirac_eo(theta, source, rtol, maxiter);

    const auto apply_M = [&](const VectorC& v)
    {
        return apply_D_dagger_D(theta, v);
    };
    const VectorC rhs = apply_D_dagger(theta, source);
    return conjugate_gradient(apply_M, rhs, nullptr, rtol, maxiter);
}
