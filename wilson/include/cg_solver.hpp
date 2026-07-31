#pragma once

#include "types.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

struct SolveResult
{
    VectorC x;
    int iterations;
    double normal_residual;
    double dirac_residual;
};

inline SolveResult solve_CGNR(const SparseC& D,
                              const VectorC& source,
                              double tolerance,
                              int max_iterations)
{
    SparseC normal=D.adjoint()*D;
    VectorC rhs=D.adjoint()*source;

    Eigen::ConjugateGradient<SparseC,Eigen::Lower|Eigen::Upper> cg;
    cg.setTolerance(tolerance);
    cg.setMaxIterations(max_iterations);
    cg.compute(normal);
    VectorC x=cg.solve(rhs);

    double normal_residual=(normal*x-rhs).norm()
        /std::max(rhs.norm(),1.0e-300);
    double dirac_residual=(D*x-source).norm()
        /std::max(source.norm(),1.0e-300);

    if(!std::isfinite(normal_residual) ||
       normal_residual>20.0*tolerance)
    {
        std::ostringstream msg;
        msg.setf(std::ios::scientific);
        msg<<"CGNR failed: iterations="<<cg.iterations()
           <<", normal residual="<<normal_residual
           <<", D residual="<<dirac_residual;
        throw std::runtime_error(msg.str());
    }

    return {x,static_cast<int>(cg.iterations()),
            normal_residual,dirac_residual};
}

inline VectorC solve_DDdag(const SparseC& D,
                           const VectorC& phi,
                           double tolerance,
                           int max_iterations)
{
    SparseC A=D*D.adjoint();

    Eigen::ConjugateGradient<SparseC,Eigen::Lower|Eigen::Upper> cg;
    cg.setTolerance(tolerance);
    cg.setMaxIterations(max_iterations);
    cg.compute(A);
    VectorC eta=cg.solve(phi);

    double residual=(A*eta-phi).norm()
        /std::max(phi.norm(),1.0e-300);

    if(!std::isfinite(residual)||residual>20.0*tolerance)
    {
        std::ostringstream msg;
        msg.setf(std::ios::scientific);
        msg<<"pseudofermion CG failed: iterations="<<cg.iterations()
           <<", residual="<<residual;
        throw std::runtime_error(msg.str());
    }
    return eta;
}
