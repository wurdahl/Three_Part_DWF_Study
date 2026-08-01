#pragma once

#include <complex>
#include <memory>
#include <string>

// Fully device-resident HMC trajectory engine for the domain-wall action.
// The gauge field, momentum, links, CG state, and forces all live on the
// GPU; per trajectory the host only uploads chi and the momentum draw,
// downloads a handful of Hamiltonian scalars, and downloads the accepted
// configuration. CG runs in fixed-size CUDA-graph chunks with device-side
// alpha/beta, so the only per-solve host synchronization is one 8-byte
// residual read per chunk.
//
// MD force solves run in fp32; the initial and final solves of each
// trajectory (whose actions enter the Metropolis test) run in fp64. A
// stalled fp32 solve is retried once in fp64.
class GpuHmc
{
public:
    struct TrajectoryStats
    {
        double H0;
        double H;
        long long cg_iterations;   // upper bound: counted in chunk units
    };

    GpuHmc(int Nt, int Nx, int N5,
           double M5, double mf, double beta_gauge);
    ~GpuHmc();

    // Uploads the cold-start (or restored) gauge field into theta0.
    void set_theta0(const double* theta0);

    // Uploads chi, builds phi = D(theta0) chi on the device, and clears
    // the CG warm-start guess. Call once per trajectory before momentum.
    void begin_trajectory(const std::complex<double>* chi);

    void upload_momentum(const double* momentum);

    // Runs the initial fp64 action solve, the full leapfrog, and the
    // final fp64 action solve without downloading any field data.
    TrajectoryStats run_leapfrog(
        int steps, double epsilon, double rtol, int maxiter, bool fp64_md);

    // Applies the Metropolis outcome and downloads theta0 (2*Nt*Nx).
    void finish_trajectory(bool accepted, double* theta0_out);

    std::string device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
