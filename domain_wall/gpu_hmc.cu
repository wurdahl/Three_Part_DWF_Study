// Fully device-resident CUDA implementation of the domain-wall HMC
// trajectory. Mirrors dwf_operator.hpp, cg_solver.hpp, pseudofermion.hpp,
// and gauge.hpp. Three latency optimizations over a straightforward port:
//   1. CG alpha/beta live in device memory, so no per-iteration host sync.
//   2. The whole leapfrog (gauge field, momentum, links, forces) stays on
//      the GPU; the host sees only Hamiltonian scalars per trajectory.
//   3. CG iterations run in fixed 8-iteration CUDA-graph chunks; the only
//      per-chunk host traffic is one 8-byte residual read.

#include "include/gpu_hmc.hpp"

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
using C = cuda::std::complex<T>;

constexpr int kBlock = 256;
constexpr int kReduceBlocks = 64;

// CG iterations per graph launch. Larger chunks amortize launch latency
// but overshoot convergence by up to kChunk-1 iterations (a slightly
// more-converged eta; tolerance-level shifts in dH). Build with
// -DDWF_CG_CHUNK=1 to reproduce the CPU solver's exact stopping point.
#ifndef DWF_CG_CHUNK
#define DWF_CG_CHUNK 8
#endif
constexpr int kChunk = DWF_CG_CHUNK;

// Device scalar slots.
enum Slot
{
    RR_OLD = 0,
    RR_NEW = 1,
    PAP = 2,
    ALPHA = 3,
    BETA = 4,
    BNORM2 = 5,
    PF_ACTION = 6,
    G_ACTION = 7,
    KINETIC = 8,
    NSLOTS = 9
};

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
}

struct Dims
{
    int Nt;
    int Nx;
    int N5;
};

__device__ __forceinline__ int fidx(Dims d, int s, int t, int x, int a)
{
    return (((s * d.Nt + t) * d.Nx + x) << 1) + a;
}

__device__ __forceinline__ int gidx(Dims d, int mu, int t, int x)
{
    return (mu * d.Nt + t) * d.Nx + x;
}

// ---------------------------------------------------------------------
// Domain-wall operator kernels (identical math to dwf_operator.hpp).
// Spin algebra, with s1 = sigma1 and s2 = sigma2:
//   (I - s1) v = (v0 - v1, v1 - v0)     (I + s1) v = (v0 + v1, v0 + v1)
//   (I - s2) v = (v0 + i v1, v1 - i v0) (I + s2) v = (v0 - i v1, v1 + i v0)
// ---------------------------------------------------------------------
template <typename T>
__global__ void k_apply_D(
    const C<T>* __restrict__ psi,
    C<T>* __restrict__ out,
    const C<T>* __restrict__ links,
    Dims d, T M5, T mf)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= d.N5 * d.Nt * d.Nx)
        return;

    const int x = site % d.Nx;
    const int t = (site / d.Nx) % d.Nt;
    const int s = site / (d.Nx * d.Nt);

    const int tp = (t + 1 == d.Nt) ? 0 : t + 1;
    const int tm = (t == 0) ? d.Nt - 1 : t - 1;
    const int xp = (x + 1 == d.Nx) ? 0 : x + 1;
    const int xm = (x == 0) ? d.Nx - 1 : x - 1;
    const T sign_tp = (t == d.Nt - 1) ? T(-1) : T(1);
    const T sign_tm = (t == 0) ? T(-1) : T(1);

    const C<T> Ut = links[gidx(d, 0, t, x)];
    const C<T> Ut_dag = conj(links[gidx(d, 0, tm, x)]);
    const C<T> Ux = links[gidx(d, 1, t, x)];
    const C<T> Ux_dag = conj(links[gidx(d, 1, t, xm)]);

    const C<T> tp0 = sign_tp * psi[fidx(d, s, tp, x, 0)];
    const C<T> tp1 = sign_tp * psi[fidx(d, s, tp, x, 1)];
    const C<T> tm0 = sign_tm * psi[fidx(d, s, tm, x, 0)];
    const C<T> tm1 = sign_tm * psi[fidx(d, s, tm, x, 1)];
    const C<T> xp0 = psi[fidx(d, s, t, xp, 0)];
    const C<T> xp1 = psi[fidx(d, s, t, xp, 1)];
    const C<T> xm0 = psi[fidx(d, s, t, xm, 0)];
    const C<T> xm1 = psi[fidx(d, s, t, xm, 1)];

    const C<T> iu(T(0), T(1));
    const T half = T(0.5);

    C<T> out0 = (T(3) - M5) * psi[fidx(d, s, t, x, 0)];
    C<T> out1 = (T(3) - M5) * psi[fidx(d, s, t, x, 1)];

    out0 -= half * (Ut * (tp0 - tp1) + Ut_dag * (tm0 + tm1)
                    + Ux * (xp0 + iu * xp1) + Ux_dag * (xm0 - iu * xm1));
    out1 -= half * (Ut * (tp1 - tp0) + Ut_dag * (tm0 + tm1)
                    + Ux * (xp1 - iu * xp0) + Ux_dag * (xm1 + iu * xm0));

    out1 += (s < d.N5 - 1) ? -psi[fidx(d, s + 1, t, x, 1)]
                           : mf * psi[fidx(d, 0, t, x, 1)];
    out0 += (s > 0) ? -psi[fidx(d, s - 1, t, x, 0)]
                    : mf * psi[fidx(d, d.N5 - 1, t, x, 0)];

    out[fidx(d, s, t, x, 0)] = out0;
    out[fidx(d, s, t, x, 1)] = out1;
}

template <typename T>
__global__ void k_apply_D_dagger(
    const C<T>* __restrict__ psi,
    C<T>* __restrict__ out,
    const C<T>* __restrict__ links,
    Dims d, T M5, T mf)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= d.N5 * d.Nt * d.Nx)
        return;

    const int x = site % d.Nx;
    const int t = (site / d.Nx) % d.Nt;
    const int s = site / (d.Nx * d.Nt);

    const int tp = (t + 1 == d.Nt) ? 0 : t + 1;
    const int tm = (t == 0) ? d.Nt - 1 : t - 1;
    const int xp = (x + 1 == d.Nx) ? 0 : x + 1;
    const int xm = (x == 0) ? d.Nx - 1 : x - 1;
    const T sign_tp = (t == d.Nt - 1) ? T(-1) : T(1);
    const T sign_tm = (t == 0) ? T(-1) : T(1);

    const C<T> Ut = links[gidx(d, 0, t, x)];
    const C<T> Ut_dag = conj(links[gidx(d, 0, tm, x)]);
    const C<T> Ux = links[gidx(d, 1, t, x)];
    const C<T> Ux_dag = conj(links[gidx(d, 1, t, xm)]);

    const C<T> tp0 = sign_tp * psi[fidx(d, s, tp, x, 0)];
    const C<T> tp1 = sign_tp * psi[fidx(d, s, tp, x, 1)];
    const C<T> tm0 = sign_tm * psi[fidx(d, s, tm, x, 0)];
    const C<T> tm1 = sign_tm * psi[fidx(d, s, tm, x, 1)];
    const C<T> xp0 = psi[fidx(d, s, t, xp, 0)];
    const C<T> xp1 = psi[fidx(d, s, t, xp, 1)];
    const C<T> xm0 = psi[fidx(d, s, t, xm, 0)];
    const C<T> xm1 = psi[fidx(d, s, t, xm, 1)];

    const C<T> iu(T(0), T(1));
    const T half = T(0.5);

    C<T> out0 = (T(3) - M5) * psi[fidx(d, s, t, x, 0)];
    C<T> out1 = (T(3) - M5) * psi[fidx(d, s, t, x, 1)];

    out0 -= half * (Ut_dag * (tm0 - tm1) + Ut * (tp0 + tp1)
                    + Ux_dag * (xm0 + iu * xm1) + Ux * (xp0 - iu * xp1));
    out1 -= half * (Ut_dag * (tm1 - tm0) + Ut * (tp0 + tp1)
                    + Ux_dag * (xm1 - iu * xm0) + Ux * (xp1 + iu * xp0));

    out1 += (s > 0) ? -psi[fidx(d, s - 1, t, x, 1)]
                    : mf * psi[fidx(d, d.N5 - 1, t, x, 1)];
    out0 += (s < d.N5 - 1) ? -psi[fidx(d, s + 1, t, x, 0)]
                           : mf * psi[fidx(d, 0, t, x, 0)];

    out[fidx(d, s, t, x, 0)] = out0;
    out[fidx(d, s, t, x, 1)] = out1;
}

// ---------------------------------------------------------------------
// Reductions: deterministic two-stage, fp64 accumulation, result written
// into a device scalar slot (no host round trip).
// ---------------------------------------------------------------------
template <typename T>
__global__ void k_cdot_partial(
    const C<T>* __restrict__ a, const C<T>* __restrict__ b,
    double* __restrict__ partial, int n)
{
    __shared__ double sh[kBlock];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
    {
        // real part of conj(a) . b; every consumer needs only this.
        sum += static_cast<double>(a[i].real())
                 * static_cast<double>(b[i].real())
             + static_cast<double>(a[i].imag())
                 * static_cast<double>(b[i].imag());
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sh[0];
}

__global__ void k_rdot_partial(
    const double* __restrict__ a, const double* __restrict__ b,
    double* __restrict__ partial, int n)
{
    __shared__ double sh[kBlock];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        sum += a[i] * b[i];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sh[0];
}

__global__ void k_rsum_partial(
    const double* __restrict__ a, double* __restrict__ partial, int n)
{
    __shared__ double sh[kBlock];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        sum += a[i];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sh[0];
}

__global__ void k_reduce_final(
    const double* __restrict__ partial, double* __restrict__ out,
    double scale)
{
    __shared__ double sh[kReduceBlocks];
    sh[threadIdx.x] = partial[threadIdx.x];
    __syncthreads();
    for (int stride = kReduceBlocks / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = scale * sh[0];
}

// ---------------------------------------------------------------------
// CG vector updates with device-resident scalars.
// ---------------------------------------------------------------------
__global__ void k_set_alpha(double* __restrict__ scal)
{
    scal[ALPHA] = scal[RR_OLD] / scal[PAP];
}

__global__ void k_set_beta(double* __restrict__ scal)
{
    scal[BETA] = scal[RR_NEW] / scal[RR_OLD];
    scal[RR_OLD] = scal[RR_NEW];
}

template <typename T>
__global__ void k_cg_update(
    C<T>* __restrict__ x, C<T>* __restrict__ r,
    const C<T>* __restrict__ p, const C<T>* __restrict__ Ap,
    const double* __restrict__ scal, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const T alpha = static_cast<T>(scal[ALPHA]);
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

template <typename T>
__global__ void k_xpay(
    C<T>* __restrict__ p, const C<T>* __restrict__ r,
    const double* __restrict__ scal, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = r[i] + static_cast<T>(scal[BETA]) * p[i];
}

template <typename T>
__global__ void k_residual(
    C<T>* __restrict__ r, const C<T>* __restrict__ b,
    const C<T>* __restrict__ Ax, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        r[i] = b[i] - Ax[i];
}

__global__ void k_f64_to_f32(
    C<float>* __restrict__ dst, const C<double>* __restrict__ src, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = C<float>(
            static_cast<float>(src[i].real()),
            static_cast<float>(src[i].imag()));
}

__global__ void k_f32_to_f64(
    C<double>* __restrict__ dst, const C<float>* __restrict__ src, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = C<double>(src[i].real(), src[i].imag());
}

// ---------------------------------------------------------------------
// Pseudofermion force (identical math to pseudofermion.hpp).
// ---------------------------------------------------------------------
template <typename T>
__global__ void k_force_accum(
    const C<T>* __restrict__ eta, const C<T>* __restrict__ xi,
    const C<T>* __restrict__ links,
    C<T>* __restrict__ t_fwd, C<T>* __restrict__ t_bwd,
    C<T>* __restrict__ x_fwd, C<T>* __restrict__ x_bwd,
    Dims d)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= d.Nt * d.Nx)
        return;

    const int x = site % d.Nx;
    const int t = site / d.Nx;

    const int tp = (t + 1 == d.Nt) ? 0 : t + 1;
    const int tm = (t == 0) ? d.Nt - 1 : t - 1;
    const int xp = (x + 1 == d.Nx) ? 0 : x + 1;
    const int xm = (x == 0) ? d.Nx - 1 : x - 1;
    const T sign_tp = (t == d.Nt - 1) ? T(-1) : T(1);
    const T sign_tm = (t == 0) ? T(-1) : T(1);

    const C<T> Ut = links[gidx(d, 0, t, x)];
    const C<T> Ut_dag = conj(links[gidx(d, 0, tm, x)]);
    const C<T> Ux = links[gidx(d, 1, t, x)];
    const C<T> Ux_dag = conj(links[gidx(d, 1, t, xm)]);

    const C<T> iu(T(0), T(1));
    const T half = T(0.5);

    C<T> tf(T(0), T(0));
    C<T> tb(T(0), T(0));
    C<T> xf(T(0), T(0));
    C<T> xb(T(0), T(0));

    for (int s = 0; s < d.N5; ++s)
    {
        const C<T> e0 = conj(eta[fidx(d, s, t, x, 0)]);
        const C<T> e1 = conj(eta[fidx(d, s, t, x, 1)]);

        const C<T> xtp0 = sign_tp * xi[fidx(d, s, tp, x, 0)];
        const C<T> xtp1 = sign_tp * xi[fidx(d, s, tp, x, 1)];
        const C<T> xtm0 = sign_tm * xi[fidx(d, s, tm, x, 0)];
        const C<T> xtm1 = sign_tm * xi[fidx(d, s, tm, x, 1)];
        const C<T> xxp0 = xi[fidx(d, s, t, xp, 0)];
        const C<T> xxp1 = xi[fidx(d, s, t, xp, 1)];
        const C<T> xxm0 = xi[fidx(d, s, t, xm, 0)];
        const C<T> xxm1 = xi[fidx(d, s, t, xm, 1)];

        tf += -half * iu * Ut * (e0 * (xtp0 - xtp1) + e1 * (xtp1 - xtp0));
        tb += half * iu * Ut_dag * (e0 * (xtm0 + xtm1)
                                    + e1 * (xtm0 + xtm1));
        xf += -half * iu * Ux * (e0 * (xxp0 + iu * xxp1)
                                 + e1 * (xxp1 - iu * xxp0));
        xb += half * iu * Ux_dag * (e0 * (xxm0 - iu * xxm1)
                                    + e1 * (xxm1 + iu * xxm0));
    }

    t_fwd[site] = tf;
    t_bwd[site] = tb;
    x_fwd[site] = xf;
    x_bwd[site] = xb;
}

template <typename T>
__global__ void k_force_combine(
    const C<T>* __restrict__ t_fwd, const C<T>* __restrict__ t_bwd,
    const C<T>* __restrict__ x_fwd, const C<T>* __restrict__ x_bwd,
    double* __restrict__ force, Dims d)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= d.Nt * d.Nx)
        return;

    const int x = site % d.Nx;
    const int t = site / d.Nx;
    const int tp = (t + 1 == d.Nt) ? 0 : t + 1;
    const int xp = (x + 1 == d.Nx) ? 0 : x + 1;

    force[gidx(d, 0, t, x)] = -2.0 * static_cast<double>(
        (t_fwd[t * d.Nx + x] + t_bwd[tp * d.Nx + x]).real());
    force[gidx(d, 1, t, x)] = -2.0 * static_cast<double>(
        (x_fwd[t * d.Nx + x] + x_bwd[t * d.Nx + xp]).real());
}

// ---------------------------------------------------------------------
// Gauge sector (identical math to gauge.hpp).
// ---------------------------------------------------------------------
__global__ void k_build_links(
    const double* __restrict__ theta,
    C<double>* __restrict__ links, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double s;
        double c;
        sincos(theta[i], &s, &c);
        links[i] = C<double>(c, s);
    }
}

__global__ void k_gauge_plaq(
    const double* __restrict__ theta,
    double* __restrict__ sin_p, double* __restrict__ act_site,
    Dims d, double beta_gauge)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= d.Nt * d.Nx)
        return;

    const int x = site % d.Nx;
    const int t = site / d.Nx;
    const int tp = (t + 1 == d.Nt) ? 0 : t + 1;
    const int xp = (x + 1 == d.Nx) ? 0 : x + 1;

    const double p = theta[gidx(d, 0, t, x)]
                   + theta[gidx(d, 1, tp, x)]
                   - theta[gidx(d, 0, t, xp)]
                   - theta[gidx(d, 1, t, x)];
    sin_p[site] = sin(p);
    act_site[site] = beta_gauge * (1.0 - cos(p));
}

__global__ void k_gauge_force(
    const double* __restrict__ sin_p, double* __restrict__ force,
    Dims d, double beta_gauge)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= d.Nt * d.Nx)
        return;

    const int x = site % d.Nx;
    const int t = site / d.Nx;
    const int tm = (t == 0) ? d.Nt - 1 : t - 1;
    const int xm = (x == 0) ? d.Nx - 1 : x - 1;

    const double here = sin_p[t * d.Nx + x];
    force[gidx(d, 0, t, x)] = beta_gauge * (here - sin_p[t * d.Nx + xm]);
    force[gidx(d, 1, t, x)] = beta_gauge * (sin_p[tm * d.Nx + x] - here);
}

__global__ void k_theta_update(
    double* __restrict__ theta, const double* __restrict__ mom,
    double epsilon, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        theta[i] += epsilon * mom[i];
}

__global__ void k_mom_update(
    double* __restrict__ mom, const double* __restrict__ gauge_force,
    const double* __restrict__ pf_force, double coeff, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mom[i] -= coeff * (gauge_force[i] + pf_force[i]);
}

int blocks_for(int n)
{
    return (n + kBlock - 1) / kBlock;
}

template <typename T>
struct DeviceFields
{
    C<T>* x = nullptr;
    C<T>* r = nullptr;
    C<T>* p = nullptr;
    C<T>* Ap = nullptr;
    C<T>* tmp = nullptr;
    C<T>* b = nullptr;
    C<T>* links = nullptr;
    C<T>* t_fwd = nullptr;
    C<T>* t_bwd = nullptr;
    C<T>* x_fwd = nullptr;
    C<T>* x_bwd = nullptr;
    cudaGraphExec_t chunk = nullptr;
};

} // namespace

struct GpuHmc::Impl
{
    Dims dims;
    double M5;
    double mf;
    double beta_gauge;
    int ndof;
    int nsites;
    int ngauge;
    int nplane;

    cudaStream_t stream = nullptr;

    DeviceFields<double> f64;
    DeviceFields<float> f32;
    double* theta0 = nullptr;
    double* theta = nullptr;
    double* mom = nullptr;
    double* gforce = nullptr;
    double* pfforce = nullptr;
    double* sin_p = nullptr;
    double* act_site = nullptr;
    double* scal = nullptr;
    double* partial = nullptr;

    enum class Guess { none, fp32, fp64 };
    Guess guess = Guess::none;
    long long chunk_iterations = 0;

    template <typename T>
    void alloc(DeviceFields<T>& f)
    {
        auto grab = [](C<T>*& ptr, std::size_t count)
        {
            check(cudaMalloc(&ptr, count * sizeof(C<T>)), "cudaMalloc");
        };
        grab(f.x, ndof);
        grab(f.r, ndof);
        grab(f.p, ndof);
        grab(f.Ap, ndof);
        grab(f.tmp, ndof);
        grab(f.b, ndof);
        grab(f.links, ngauge);
        grab(f.t_fwd, nplane);
        grab(f.t_bwd, nplane);
        grab(f.x_fwd, nplane);
        grab(f.x_bwd, nplane);
    }

    template <typename T>
    void release(DeviceFields<T>& f)
    {
        for (void* ptr : {static_cast<void*>(f.x), static_cast<void*>(f.r),
                          static_cast<void*>(f.p), static_cast<void*>(f.Ap),
                          static_cast<void*>(f.tmp), static_cast<void*>(f.b),
                          static_cast<void*>(f.links),
                          static_cast<void*>(f.t_fwd),
                          static_cast<void*>(f.t_bwd),
                          static_cast<void*>(f.x_fwd),
                          static_cast<void*>(f.x_bwd)})
            cudaFree(ptr);
        if (f.chunk)
            cudaGraphExecDestroy(f.chunk);
    }

    template <typename T>
    void cdot(const C<T>* a, const C<T>* b, Slot slot, double scale = 1.0)
    {
        k_cdot_partial<T><<<kReduceBlocks, kBlock, 0, stream>>>(
            a, b, partial, ndof);
        k_reduce_final<<<1, kReduceBlocks, 0, stream>>>(
            partial, scal + slot, scale);
    }

    void rdot(const double* a, const double* b, int n, Slot slot,
              double scale)
    {
        k_rdot_partial<<<kReduceBlocks, kBlock, 0, stream>>>(
            a, b, partial, n);
        k_reduce_final<<<1, kReduceBlocks, 0, stream>>>(
            partial, scal + slot, scale);
    }

    void rsum(const double* a, int n, Slot slot)
    {
        k_rsum_partial<<<kReduceBlocks, kBlock, 0, stream>>>(a, partial, n);
        k_reduce_final<<<1, kReduceBlocks, 0, stream>>>(
            partial, scal + slot, 1.0);
    }

    double read_scalar(Slot slot)
    {
        double value = 0.0;
        check(cudaMemcpyAsync(&value, scal + slot, sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              "scalar read");
        check(cudaStreamSynchronize(stream), "scalar sync");
        return value;
    }

    template <typename T>
    void apply_A(const C<T>* in, C<T>* out, C<T>* scratch,
                 const C<T>* links)
    {
        const T tM5 = static_cast<T>(M5);
        const T tmf = static_cast<T>(mf);
        k_apply_D_dagger<T><<<blocks_for(nsites), kBlock, 0, stream>>>(
            in, scratch, links, dims, tM5, tmf);
        k_apply_D<T><<<blocks_for(nsites), kBlock, 0, stream>>>(
            scratch, out, links, dims, tM5, tmf);
    }

    // Records kChunk CG iterations as an executable graph. All pointers
    // and grid shapes are fixed for the lifetime of the object.
    template <typename T>
    void capture_chunk(DeviceFields<T>& f)
    {
        const int nb = blocks_for(ndof);
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
              "begin capture");
        for (int i = 0; i < kChunk; ++i)
        {
            apply_A<T>(f.p, f.Ap, f.tmp, f.links);
            cdot<T>(f.p, f.Ap, PAP);
            k_set_alpha<<<1, 1, 0, stream>>>(scal);
            k_cg_update<T><<<nb, kBlock, 0, stream>>>(
                f.x, f.r, f.p, f.Ap, scal, ndof);
            cdot<T>(f.r, f.r, RR_NEW);
            k_set_beta<<<1, 1, 0, stream>>>(scal);
            k_xpay<T><<<nb, kBlock, 0, stream>>>(f.p, f.r, scal, ndof);
        }
        cudaGraph_t graph = nullptr;
        check(cudaStreamEndCapture(stream, &graph), "end capture");
        check(cudaGraphInstantiate(&f.chunk, graph, 0), "instantiate");
        cudaGraphDestroy(graph);
    }

    // CG on (D D^dagger) x = b, semantics matching cg_solver.hpp except
    // that convergence is only tested every kChunk iterations.
    template <typename T>
    bool cg(DeviceFields<T>& f, bool have_guess, double rtol, int maxiter)
    {
        const int nb = blocks_for(ndof);

        if (!have_guess)
        {
            check(cudaMemsetAsync(f.x, 0, ndof * sizeof(C<T>), stream),
                  "memset x");
            check(cudaMemcpyAsync(f.r, f.b, ndof * sizeof(C<T>),
                                  cudaMemcpyDeviceToDevice, stream),
                  "copy r=b");
        }
        else
        {
            apply_A<T>(f.x, f.Ap, f.tmp, f.links);
            k_residual<T><<<nb, kBlock, 0, stream>>>(f.r, f.b, f.Ap, ndof);
        }
        check(cudaMemcpyAsync(f.p, f.r, ndof * sizeof(C<T>),
                              cudaMemcpyDeviceToDevice, stream),
              "copy p=r");

        cdot<T>(f.b, f.b, BNORM2);
        cdot<T>(f.r, f.r, RR_OLD);

        const double bnorm2 = read_scalar(BNORM2);
        if (bnorm2 == 0.0)
            return true;
        const double bnorm = std::sqrt(bnorm2);

        double rr = read_scalar(RR_OLD);
        if (std::sqrt(rr) / bnorm <= rtol)
            return true;

        int iterations = 0;
        while (iterations < maxiter)
        {
            check(cudaGraphLaunch(f.chunk, stream), "chunk launch");
            iterations += kChunk;
            ++chunk_iterations;
            rr = read_scalar(RR_NEW);
            if (!std::isfinite(rr))
                throw std::runtime_error(
                    "GPU CG produced a non-finite residual");
            if (std::sqrt(rr) / bnorm <= rtol)
                return true;
        }
        return false;
    }

    // Solve, then leave the pseudofermion force in pfforce and (for fp64
    // solves) the action in scal[PF_ACTION]. Falls back to fp64 if an
    // fp32 solve stalls.
    void solve_pf(bool fp64_solve, double rtol, int maxiter)
    {
        const int nb = blocks_for(ndof);
        bool run_fp64 = fp64_solve;

        if (!run_fp64)
        {
            if (guess == Guess::fp64)
                k_f64_to_f32<<<nb, kBlock, 0, stream>>>(
                    f32.x, f64.x, ndof);
            const bool ok =
                cg(f32, guess != Guess::none, rtol, maxiter);
            if (ok)
            {
                guess = Guess::fp32;
                pf_force(f32);
                return;
            }
            run_fp64 = true;
        }

        if (guess == Guess::fp32)
            k_f32_to_f64<<<nb, kBlock, 0, stream>>>(f64.x, f32.x, ndof);
        if (!cg(f64, guess != Guess::none, rtol, maxiter))
            throw std::runtime_error(
                "Pseudofermion CG failed to converge");
        guess = Guess::fp64;
        cdot<double>(f64.b, f64.x, PF_ACTION);
        pf_force(f64);
    }

    template <typename T>
    void pf_force(DeviceFields<T>& f)
    {
        const T tM5 = static_cast<T>(M5);
        const T tmf = static_cast<T>(mf);
        k_apply_D_dagger<T><<<blocks_for(nsites), kBlock, 0, stream>>>(
            f.x, f.tmp, f.links, dims, tM5, tmf);
        k_force_accum<T><<<blocks_for(nplane), kBlock, 0, stream>>>(
            f.x, f.tmp, f.links, f.t_fwd, f.t_bwd, f.x_fwd, f.x_bwd, dims);
        k_force_combine<T><<<blocks_for(nplane), kBlock, 0, stream>>>(
            f.t_fwd, f.t_bwd, f.x_fwd, f.x_bwd, pfforce, dims);
    }

    void build_links(const double* theta_field)
    {
        k_build_links<<<blocks_for(ngauge), kBlock, 0, stream>>>(
            theta_field, f64.links, ngauge);
        k_f64_to_f32<<<blocks_for(ngauge), kBlock, 0, stream>>>(
            f32.links, f64.links, ngauge);
    }

    // Gauge plaquette pass at the current theta: fills sin_p and
    // act_site, then the force. The action reduction is launched only
    // when the Hamiltonian is needed.
    void gauge_pass(bool need_action)
    {
        k_gauge_plaq<<<blocks_for(nplane), kBlock, 0, stream>>>(
            theta, sin_p, act_site, dims, beta_gauge);
        if (need_action)
            rsum(act_site, nplane, G_ACTION);
        k_gauge_force<<<blocks_for(nplane), kBlock, 0, stream>>>(
            sin_p, gforce, dims, beta_gauge);
    }
};

GpuHmc::GpuHmc(int Nt, int Nx, int N5,
               double M5, double mf, double beta_gauge)
    : impl_(new Impl)
{
    Impl& im = *impl_;
    im.dims = {Nt, Nx, N5};
    im.M5 = M5;
    im.mf = mf;
    im.beta_gauge = beta_gauge;
    im.nsites = N5 * Nt * Nx;
    im.ndof = im.nsites * 2;
    im.ngauge = 2 * Nt * Nx;
    im.nplane = Nt * Nx;

    check(cudaStreamCreate(&im.stream), "stream create");
    im.alloc(im.f64);
    im.alloc(im.f32);

    auto grab = [](double*& ptr, std::size_t count)
    {
        check(cudaMalloc(&ptr, count * sizeof(double)), "cudaMalloc");
    };
    grab(im.theta0, im.ngauge);
    grab(im.theta, im.ngauge);
    grab(im.mom, im.ngauge);
    grab(im.gforce, im.ngauge);
    grab(im.pfforce, im.ngauge);
    grab(im.sin_p, im.nplane);
    grab(im.act_site, im.nplane);
    grab(im.scal, NSLOTS);
    grab(im.partial, kReduceBlocks);

    im.capture_chunk(im.f64);
    im.capture_chunk(im.f32);
}

GpuHmc::~GpuHmc()
{
    Impl& im = *impl_;
    im.release(im.f64);
    im.release(im.f32);
    for (double* ptr : {im.theta0, im.theta, im.mom, im.gforce,
                        im.pfforce, im.sin_p, im.act_site, im.scal,
                        im.partial})
        cudaFree(ptr);
    cudaStreamDestroy(im.stream);
}

void GpuHmc::set_theta0(const double* theta0)
{
    Impl& im = *impl_;
    check(cudaMemcpyAsync(im.theta0, theta0,
                          im.ngauge * sizeof(double),
                          cudaMemcpyHostToDevice, im.stream),
          "theta0 upload");
    check(cudaStreamSynchronize(im.stream), "theta0 sync");
}

void GpuHmc::begin_trajectory(const std::complex<double>* chi)
{
    Impl& im = *impl_;
    check(cudaMemcpyAsync(im.f64.tmp, chi,
                          im.ndof * sizeof(C<double>),
                          cudaMemcpyHostToDevice, im.stream),
          "chi upload");
    im.build_links(im.theta0);
    k_apply_D<double><<<blocks_for(im.nsites), kBlock, 0, im.stream>>>(
        im.f64.tmp, im.f64.b, im.f64.links, im.dims, im.M5, im.mf);
    k_f64_to_f32<<<blocks_for(im.ndof), kBlock, 0, im.stream>>>(
        im.f32.b, im.f64.b, im.ndof);
    im.guess = Impl::Guess::none;
}

void GpuHmc::upload_momentum(const double* momentum)
{
    Impl& im = *impl_;
    check(cudaMemcpyAsync(im.mom, momentum,
                          im.ngauge * sizeof(double),
                          cudaMemcpyHostToDevice, im.stream),
          "momentum upload");
    check(cudaStreamSynchronize(im.stream), "momentum sync");
}

GpuHmc::TrajectoryStats GpuHmc::run_leapfrog(
    int steps, double epsilon, double rtol, int maxiter, bool fp64_md)
{
    Impl& im = *impl_;
    const int nb = blocks_for(im.ngauge);
    im.chunk_iterations = 0;

    // Initial Hamiltonian at theta0 (links already built there).
    check(cudaMemcpyAsync(im.theta, im.theta0,
                          im.ngauge * sizeof(double),
                          cudaMemcpyDeviceToDevice, im.stream),
          "theta copy");
    im.gauge_pass(true);
    im.solve_pf(true, rtol, maxiter);
    im.rdot(im.mom, im.mom, im.ngauge, KINETIC, 0.5);

    TrajectoryStats stats{};
    stats.H0 = im.read_scalar(KINETIC)
             + im.read_scalar(G_ACTION)
             + im.read_scalar(PF_ACTION);

    k_mom_update<<<nb, kBlock, 0, im.stream>>>(
        im.mom, im.gforce, im.pfforce, 0.5 * epsilon, im.ngauge);

    for (int step = 0; step < steps; ++step)
    {
        const bool last = step == steps - 1;
        k_theta_update<<<nb, kBlock, 0, im.stream>>>(
            im.theta, im.mom, epsilon, im.ngauge);
        im.build_links(im.theta);
        im.gauge_pass(last);
        // Only the final step's action enters the Metropolis test.
        im.solve_pf(fp64_md || last, rtol, maxiter);
        if (!last)
            k_mom_update<<<nb, kBlock, 0, im.stream>>>(
                im.mom, im.gforce, im.pfforce, epsilon, im.ngauge);
    }

    k_mom_update<<<nb, kBlock, 0, im.stream>>>(
        im.mom, im.gforce, im.pfforce, 0.5 * epsilon, im.ngauge);
    im.rdot(im.mom, im.mom, im.ngauge, KINETIC, 0.5);

    stats.H = im.read_scalar(KINETIC)
            + im.read_scalar(G_ACTION)
            + im.read_scalar(PF_ACTION);
    stats.cg_iterations = im.chunk_iterations * kChunk;
    return stats;
}

void GpuHmc::finish_trajectory(bool accepted, double* theta0_out)
{
    Impl& im = *impl_;
    if (accepted)
        check(cudaMemcpyAsync(im.theta0, im.theta,
                              im.ngauge * sizeof(double),
                              cudaMemcpyDeviceToDevice, im.stream),
              "accept copy");
    check(cudaMemcpyAsync(theta0_out, im.theta0,
                          im.ngauge * sizeof(double),
                          cudaMemcpyDeviceToHost, im.stream),
          "theta0 download");
    check(cudaStreamSynchronize(im.stream), "finish sync");
}

std::string GpuHmc::device_name() const
{
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, device),
          "cudaGetDeviceProperties");
    return properties.name;
}
