#pragma once

#include "cg_solver.hpp"
#include "dwf_operator.hpp"
#include "dwf_solve.hpp"
#include "indexing.hpp"
#include "parameters.hpp"
#include "spin.hpp"
#include "types.hpp"

#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

struct CorrelatorPair
{
    Corr CPP;
    Corr CJ5;
};

using ComplexCorr = std::vector<Complex>;

// Stochastic estimator of L(t)=sum_x Tr[gamma5 G_q(x,t;x,t)] for the
// physical boundary quark field. Complex Z4 noise has E[eta eta^dagger]=1.
inline ComplexCorr estimate_pseudoscalar_loop_density(
    const Gauge& theta,
    unsigned long long seed)
{
    if (eta_noise_vectors < 1)
        throw std::runtime_error("eta.noise_vectors must be positive");
    if (eta_time_dilution < 1 || Nt % eta_time_dilution != 0)
        throw std::runtime_error(
            "eta.time_dilution must be positive and divide dwf.Nt");
    if (eta_spin_dilution != 1 && eta_spin_dilution != Ns)
        throw std::runtime_error("eta.spin_dilution must be 1 or 2");

    ComplexCorr loop(Nt * Nx, Complex(0.0, 0.0));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> z4(0, 3);
    const Complex phases[4] = {
        Complex(1.0, 0.0), Complex(0.0, 1.0),
        Complex(-1.0, 0.0), Complex(0.0, -1.0)};
    const Eigen::Matrix2cd PL = projector_L();
    const Eigen::Matrix2cd PR = projector_R();
    const Eigen::Matrix2cd gamma5 = sigma3();

    for (int noise = 0; noise < eta_noise_vectors; ++noise)
    {
        std::vector<Complex> eta(
            static_cast<std::size_t>(Nt * Nx * Ns));

        for (int t = 0; t < Nt; ++t)
            for (int x = 0; x < Nx; ++x)
                for (int spin = 0; spin < Ns; ++spin)
                    eta[(t * Nx + x) * Ns + spin] = phases[z4(rng)];

        // Dilution: the noise vector is split into subsets that are
        // solved separately, so the only surviving off-diagonal noise at
        // a site comes from within its own subset. Time dilution is
        // interlaced (t = t_class mod eta_time_dilution), which is what
        // matters for a time-separated loop correlator. With both
        // dilutions set to 1 this is the original single-solve estimator.
        for (int t_class = 0; t_class < eta_time_dilution; ++t_class)
        {
            for (int spin_class = 0;
                 spin_class < eta_spin_dilution;
                 ++spin_class)
            {
                const auto in_subset = [&](int spin)
                {
                    return eta_spin_dilution == 1 || spin == spin_class;
                };

                VectorC source = VectorC::Zero(Ndof);
                for (int t = t_class; t < Nt; t += eta_time_dilution)
                {
                    for (int x = 0; x < Nx; ++x)
                    {
                        Eigen::Vector2cd value =
                            Eigen::Vector2cd::Zero();
                        for (int spin = 0; spin < Ns; ++spin)
                            if (in_subset(spin))
                                value[spin] =
                                    eta[(t * Nx + x) * Ns + spin];
                        const Eigen::Vector2cd left = PL * value;
                        const Eigen::Vector2cd right = PR * value;
                        for (int spin = 0; spin < Ns; ++spin)
                        {
                            source[fermion_index(0, t, x, spin)]
                                += left[spin];
                            source[fermion_index(N5 - 1, t, x, spin)]
                                += right[spin];
                        }
                    }
                }

                const CgResult solve = propagator_solve(
                    theta, source, propagator_rtol, propagator_maxiter);
                if (!solve.converged)
                    throw std::runtime_error(
                        "Eta loop CGNR failed to converge");

                for (int t = t_class; t < Nt; t += eta_time_dilution)
                {
                    for (int x = 0; x < Nx; ++x)
                    {
                        Eigen::Vector2cd psi_left;
                        Eigen::Vector2cd psi_right;
                        Eigen::Vector2cd noise_value =
                            Eigen::Vector2cd::Zero();
                        for (int spin = 0; spin < Ns; ++spin)
                        {
                            psi_left[spin] =
                                solve.x[fermion_index(0, t, x, spin)];
                            psi_right[spin] =
                                solve.x[fermion_index(N5 - 1, t, x, spin)];
                            if (in_subset(spin))
                                noise_value[spin] =
                                    eta[(t * Nx + x) * Ns + spin];
                        }
                        const Eigen::Vector2cd q =
                            PL * psi_left + PR * psi_right;
                        loop[t * Nx + x] +=
                            noise_value.dot(gamma5 * q);
                    }
                }
            }
        }
    }

    for (Complex& value : loop)
        value /= static_cast<double>(eta_noise_vectors);
    return loop;
}

inline std::vector<Corr> compute_momentum_CPP(const Gauge& theta)
{
    if (max_momentum < 0 || max_momentum > Nx / 2)
        throw std::runtime_error("analysis.max_momentum is outside 0..Nx/2");

    std::vector<Corr> result(
        max_momentum + 1, Corr(Nt, 0.0));
    const Eigen::Matrix2cd PL = projector_L();
    const Eigen::Matrix2cd PR = projector_R();
    const std::array<int, 2> source_times = {0, Nt / 2};

    for (int t_src : source_times)
    {
        std::array<VectorC, Ns> propagators;
        for (int source_spin = 0; source_spin < Ns; ++source_spin)
        {
            VectorC source = VectorC::Zero(Ndof);
            for (int spin = 0; spin < Ns; ++spin)
            {
                source[fermion_index(0, t_src, 0, spin)]
                    += PL(spin, source_spin);
                source[fermion_index(N5 - 1, t_src, 0, spin)]
                    += PR(spin, source_spin);
            }
            const CgResult solve = propagator_solve(
                theta, source, propagator_rtol, propagator_maxiter);
            if (!solve.converged)
                throw std::runtime_error(
                    "Momentum propagator CGNR failed to converge");
            propagators[source_spin] = solve.x;
        }

        for (int t = 0; t < Nt; ++t)
        {
            const int dt = (t - t_src + Nt) % Nt;
            for (int x = 0; x < Nx; ++x)
            {
                double local = 0.0;
                for (int source_spin = 0; source_spin < Ns; ++source_spin)
                {
                    Eigen::Vector2cd left;
                    Eigen::Vector2cd right;
                    for (int spin = 0; spin < Ns; ++spin)
                    {
                        left[spin] = propagators[source_spin][
                            fermion_index(0, t, x, spin)];
                        right[spin] = propagators[source_spin][
                            fermion_index(N5 - 1, t, x, spin)];
                    }
                    local += (PL * left + PR * right).squaredNorm();
                }
                for (int n = 0; n <= max_momentum; ++n)
                {
                    const double p = 2.0 * pi * n / Nx;
                    result[n][dt] += std::cos(p * x) * local;
                }
            }
        }
    }

    for (Corr& channel : result)
        for (double& value : channel)
            value /= static_cast<double>(source_times.size());
    return result;
}

inline std::array<VectorC, 2> boundary_wall_source_propagator(
    const Gauge& theta,
    int t_src)
{
    const Eigen::Matrix2cd PL = projector_L();
    const Eigen::Matrix2cd PR = projector_R();

    std::array<VectorC, 2> propagators;

    for (int source_spin = 0; source_spin < Ns; ++source_spin)
    {
        VectorC source = VectorC::Zero(Ndof);

        for (int x = 0; x < Nx; ++x)
        {
            for (int spin = 0; spin < Ns; ++spin)
            {
                source[fermion_index(0, t_src, x, spin)]
                    += PL(spin, source_spin);

                source[fermion_index(N5 - 1, t_src, x, spin)]
                    += PR(spin, source_spin);
            }
        }

        const CgResult solve = propagator_solve(
            theta, source, propagator_rtol, propagator_maxiter);

        if (!solve.converged)
            std::cout << "    CGNR info = nonzero\n";

        const double dirac_residual =
            (apply_D(theta, solve.x) - source).norm()
            / source.norm();

        std::cout
            << "    t_src=" << t_src
            << " spin=" << source_spin
            << " iterations=" << solve.iterations
            << " normal residual=" << solve.relative_residual
            << " D residual=" << dirac_residual
            << '\n';

        propagators[source_spin] = solve.x;
    }

    return propagators;
}

inline CorrelatorPair compute_CPP_CJ5(const Gauge& theta)
{
    Corr CPP(Nt, 0.0);
    Corr CJ5(Nt, 0.0);

    const Eigen::Matrix2cd PL = projector_L();
    const Eigen::Matrix2cd PR = projector_R();

    const std::array<int, 2> source_times = {0, Nt / 2};

    for (int t_src : source_times)
    {
        const auto propagators =
            boundary_wall_source_propagator(theta, t_src);

        for (int t = 0; t < Nt; ++t)
        {
            const int dt = (t - t_src + Nt) % Nt;

            for (int x = 0; x < Nx; ++x)
            {
                for (int source_spin = 0;
                     source_spin < Ns;
                     ++source_spin)
                {
                    Eigen::Vector2cd psi_left;
                    Eigen::Vector2cd psi_right;
                    Eigen::Vector2cd psi_mid;

                    for (int spin = 0; spin < Ns; ++spin)
                    {
                        psi_left[spin] =
                            propagators[source_spin][
                                fermion_index(0, t, x, spin)];

                        psi_right[spin] =
                            propagators[source_spin][
                                fermion_index(N5 - 1, t, x, spin)];

                        psi_mid[spin] =
                            propagators[source_spin][
                                fermion_index(N5 / 2, t, x, spin)];
                    }

                    const Eigen::Vector2cd quark =
                        PL * psi_left + PR * psi_right;

                    CPP[dt] += quark.squaredNorm();

                    for (int spin = 0; spin < Ns; ++spin)
                    {
                        CJ5[dt] += std::real(
                            std::conj(psi_mid[spin])
                            * quark[spin]);
                    }
                }
            }
        }
    }

    for (int t = 0; t < Nt; ++t)
    {
        CPP[t] /= static_cast<double>(source_times.size());
        CJ5[t] /= static_cast<double>(source_times.size());
    }

    return {std::move(CPP), std::move(CJ5)};
}
