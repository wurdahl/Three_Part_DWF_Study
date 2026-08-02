#pragma once

#include "cg_solver.hpp"
#include "dwf_operator.hpp"
#include "indexing.hpp"
#include "parameters.hpp"
#include "spin.hpp"
#include "types.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Hadspec-style distillation for the boundary quark field. The smearing
// basis is the lowest eigenvectors of the gauge-covariant spatial Laplacian
// on each timeslice; perambulators carry the propagator between bases and
// elementals carry the spatial structure of interpolators, so Wick
// contractions reduce to small dense traces done downstream in Python.

struct DistillationBasis
{
    // eigenvectors[t] is Nx x n, one Laplacian eigenvector per column,
    // ordered by ascending eigenvalue.
    std::vector<Eigen::MatrixXcd> eigenvectors;
    std::vector<Eigen::VectorXd> eigenvalues;
};

inline int distillation_vector_count()
{
    const int n = distillation_vectors > 0
        ? distillation_vectors
        : std::min(8, Nx);
    if (n < 1 || n > Nx)
        throw std::runtime_error(
            "distillation.n_vectors must be in 1..Nx");
    return n;
}

inline std::vector<int> distillation_t_sources()
{
    std::vector<int> sources;
    if (distillation_t_sources_text.empty())
    {
        sources = {0, Nt / 2};
    }
    else
    {
        std::stringstream stream(distillation_t_sources_text);
        std::string item;
        while (std::getline(stream, item, ','))
            sources.push_back(std::stoi(item));
    }
    for (int t : sources)
        if (t < 0 || t >= Nt)
            throw std::runtime_error(
                "distillation.t_sources entries must be in 0..Nt-1");
    return sources;
}

inline DistillationBasis build_distillation_basis(
    const Gauge& theta, int n_vectors)
{
    DistillationBasis basis;
    basis.eigenvectors.resize(Nt);
    basis.eigenvalues.resize(Nt);

    for (int t = 0; t < Nt; ++t)
    {
        Eigen::MatrixXcd laplacian = Eigen::MatrixXcd::Zero(Nx, Nx);
        for (int x = 0; x < Nx; ++x)
        {
            const int xp = plus_periodic(x, Nx);
            const int xm = minus_periodic(x, Nx);
            laplacian(x, x) += 2.0;
            laplacian(x, xp) -= link(theta, 1, t, x);
            laplacian(x, xm) -= std::conj(link(theta, 1, t, xm));
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(laplacian);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Laplacian eigensolve failed");
        basis.eigenvectors[t] = solver.eigenvectors().leftCols(n_vectors);
        basis.eigenvalues[t] = solver.eigenvalues().head(n_vectors);
    }
    return basis;
}

struct Elementals
{
    // momentum[n][t](k, l) = sum_x v_k*(x,t) e^{i p_n x} v_l(x,t) and the
    // symmetric-covariant-derivative analogue; interpolator spin matrices
    // are applied at contraction time.
    std::vector<std::vector<Eigen::MatrixXcd>> momentum;
    std::vector<std::vector<Eigen::MatrixXcd>> derivative;
};

inline Elementals build_elementals(
    const Gauge& theta,
    const DistillationBasis& basis,
    int momentum_count)
{
    Elementals elementals;
    elementals.momentum.assign(
        momentum_count, std::vector<Eigen::MatrixXcd>(Nt));
    elementals.derivative.assign(
        momentum_count, std::vector<Eigen::MatrixXcd>(Nt));

    for (int t = 0; t < Nt; ++t)
    {
        const Eigen::MatrixXcd& V = basis.eigenvectors[t];
        Eigen::MatrixXcd derived(Nx, V.cols());
        for (int x = 0; x < Nx; ++x)
        {
            const int xp = plus_periodic(x, Nx);
            const int xm = minus_periodic(x, Nx);
            derived.row(x) =
                0.5 * (link(theta, 1, t, x) * V.row(xp)
                       - std::conj(link(theta, 1, t, xm)) * V.row(xm));
        }
        for (int n = 0; n < momentum_count; ++n)
        {
            const double p = 2.0 * pi * n / Nx;
            Eigen::VectorXcd phase(Nx);
            for (int x = 0; x < Nx; ++x)
                phase[x] = std::exp(Complex(0.0, p * x));
            elementals.momentum[n][t] =
                V.adjoint() * phase.asDiagonal() * V;
            elementals.derivative[n][t] =
                V.adjoint() * phase.asDiagonal() * derived;
        }
    }
    return elementals;
}

// tau[t] is (Ns*n) x (Ns*n) with row index (spin a)*n + k at sink time t
// and column index (spin b)*n + l at the source: the boundary quark
// propagator projected into the distillation basis at both ends.
inline std::vector<Eigen::MatrixXcd> compute_perambulator(
    const Gauge& theta,
    const DistillationBasis& basis,
    int t_src)
{
    const int n = static_cast<int>(basis.eigenvectors[0].cols());
    std::vector<Eigen::MatrixXcd> tau(
        Nt, Eigen::MatrixXcd::Zero(Ns * n, Ns * n));

    const Eigen::Matrix2cd PL = projector_L();
    const Eigen::Matrix2cd PR = projector_R();
    const auto apply_M = [&](const VectorC& v)
    {
        return apply_D_dagger_D(theta, v);
    };

    for (int l = 0; l < n; ++l)
    {
        for (int b = 0; b < Ns; ++b)
        {
            VectorC source = VectorC::Zero(Ndof);
            for (int x = 0; x < Nx; ++x)
            {
                const Complex w = basis.eigenvectors[t_src](x, l);
                for (int spin = 0; spin < Ns; ++spin)
                {
                    source[fermion_index(0, t_src, x, spin)]
                        += PL(spin, b) * w;
                    source[fermion_index(N5 - 1, t_src, x, spin)]
                        += PR(spin, b) * w;
                }
            }

            const VectorC rhs = apply_D_dagger(theta, source);
            const CgResult solve = conjugate_gradient(
                apply_M, rhs, nullptr,
                propagator_rtol, propagator_maxiter);
            if (!solve.converged)
                throw std::runtime_error(
                    "Perambulator CGNR failed to converge");

            for (int t = 0; t < Nt; ++t)
            {
                Eigen::MatrixXcd quark(Nx, Ns);
                for (int x = 0; x < Nx; ++x)
                {
                    Eigen::Vector2cd psi_left;
                    Eigen::Vector2cd psi_right;
                    for (int spin = 0; spin < Ns; ++spin)
                    {
                        psi_left[spin] =
                            solve.x[fermion_index(0, t, x, spin)];
                        psi_right[spin] =
                            solve.x[fermion_index(N5 - 1, t, x, spin)];
                    }
                    const Eigen::Vector2cd q =
                        PL * psi_left + PR * psi_right;
                    for (int a = 0; a < Ns; ++a)
                        quark(x, a) = q[a];
                }
                const Eigen::MatrixXcd projected =
                    basis.eigenvectors[t].adjoint() * quark;
                for (int a = 0; a < Ns; ++a)
                    for (int k = 0; k < n; ++k)
                        tau[t](a * n + k, b * n + l) = projected(k, a);
            }
        }
    }
    return tau;
}
