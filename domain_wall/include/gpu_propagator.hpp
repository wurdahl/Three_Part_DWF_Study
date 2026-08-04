#pragma once

#include <complex>
#include <memory>

// GPU measurement solver for D psi = source, the analyzer/perambulator
// counterpart of GpuHmc. Solves the normal equations D^dag D psi =
// D^dag source with the same CGNR semantics as cg_solver.hpp.
//
// Consumer Blackwell runs fp64 at 1/64 the fp32 rate, so a pure fp64
// solve wastes the card. Instead the inner CG runs in fp32 and an outer
// fp64 defect-correction loop restores full accuracy: each refinement
// multiplies the residual by the inner tolerance, so two or three passes
// reach 1e-8 while nearly all arithmetic stays on the fast path.
//
// One instance owns one CUDA stream and its own device buffers, so the
// analyzer can keep an instance per OpenMP thread and let independent
// configurations overlap on the device.
class GpuPropagator
{
public:
    struct Result
    {
        int iterations;             // summed over refinement passes
        double relative_residual;   // fp64, on the normal equations
        bool converged;
    };

    GpuPropagator(int Nt, int Nx, int N5, double M5, double mf);
    ~GpuPropagator();

    // theta holds 2*Nt*Nx gauge angles; rebuilds the device link tables.
    void set_gauge(const double* theta);

    // source and solution each hold N5*Nt*Nx*2 complex entries.
    Result solve(const std::complex<double>* source,
                 std::complex<double>* solution,
                 double rtol, int maxiter);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
