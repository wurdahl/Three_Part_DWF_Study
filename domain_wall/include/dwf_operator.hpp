#pragma once

#include "indexing.hpp"
#include "parameters.hpp"
#include "spin.hpp"
#include "types.hpp"

#include <cmath>

inline Complex link(const Gauge& theta, int mu, int t, int x)
{
    return std::exp(Complex(0.0, theta[gauge_index(mu, t, x)]));
}

inline VectorC apply_D(const Gauge& theta, const VectorC& psi)
{
    VectorC out = (3.0 - M5) * psi;

    const Eigen::Matrix2cd I = identity2();
    const Eigen::Matrix2cd s1 = sigma1();
    const Eigen::Matrix2cd s2 = sigma2();

    for (int s = 0; s < N5; ++s)
    {
        for (int t = 0; t < Nt; ++t)
        {
            const int tp = plus_periodic(t, Nt);
            const int tm = minus_periodic(t, Nt);
            const double sign_tp = (t == Nt - 1) ? -1.0 : 1.0;
            const double sign_tm = (t == 0) ? -1.0 : 1.0;

            for (int x = 0; x < Nx; ++x)
            {
                const int xp = plus_periodic(x, Nx);
                const int xm = minus_periodic(x, Nx);

                Eigen::Vector2cd v_tp;
                Eigen::Vector2cd v_tm;
                Eigen::Vector2cd v_xp;
                Eigen::Vector2cd v_xm;

                for (int a = 0; a < Ns; ++a)
                {
                    v_tp[a] = sign_tp * psi[fermion_index(s, tp, x, a)];
                    v_tm[a] = sign_tm * psi[fermion_index(s, tm, x, a)];
                    v_xp[a] = psi[fermion_index(s, t, xp, a)];
                    v_xm[a] = psi[fermion_index(s, t, xm, a)];
                }

                const Eigen::Vector2cd term_tp = (I - s1) * v_tp;
                const Eigen::Vector2cd term_tm = (I + s1) * v_tm;
                const Eigen::Vector2cd term_xp = (I - s2) * v_xp;
                const Eigen::Vector2cd term_xm = (I + s2) * v_xm;

                const Complex Ut = link(theta, 0, t, x);
                const Complex Ut_dag =
                    std::conj(link(theta, 0, tm, x));
                const Complex Ux = link(theta, 1, t, x);
                const Complex Ux_dag =
                    std::conj(link(theta, 1, t, xm));

                for (int a = 0; a < Ns; ++a)
                {
                    const int row = fermion_index(s, t, x, a);
                    out[row] += -0.5 * Ut * term_tp[a];
                    out[row] += -0.5 * Ut_dag * term_tm[a];
                    out[row] += -0.5 * Ux * term_xp[a];
                    out[row] += -0.5 * Ux_dag * term_xm[a];
                }

                // Exact componentwise fifth-direction terms from D_DWF.
                const int row_spin1 = fermion_index(s, t, x, 1);
                if (s < N5 - 1)
                    out[row_spin1] +=
                        -psi[fermion_index(s + 1, t, x, 1)];
                else
                    out[row_spin1] +=
                        mf * psi[fermion_index(0, t, x, 1)];

                const int row_spin0 = fermion_index(s, t, x, 0);
                if (s > 0)
                    out[row_spin0] +=
                        -psi[fermion_index(s - 1, t, x, 0)];
                else
                    out[row_spin0] +=
                        mf * psi[fermion_index(N5 - 1, t, x, 0)];
            }
        }
    }

    return out;
}

inline VectorC apply_D_dagger(const Gauge& theta, const VectorC& psi)
{
    VectorC out = (3.0 - M5) * psi;

    const Eigen::Matrix2cd I = identity2();
    const Eigen::Matrix2cd s1 = sigma1();
    const Eigen::Matrix2cd s2 = sigma2();

    for (int s = 0; s < N5; ++s)
    {
        for (int t = 0; t < Nt; ++t)
        {
            const int tp = plus_periodic(t, Nt);
            const int tm = minus_periodic(t, Nt);
            const double sign_tp = (t == Nt - 1) ? -1.0 : 1.0;
            const double sign_tm = (t == 0) ? -1.0 : 1.0;

            for (int x = 0; x < Nx; ++x)
            {
                const int xp = plus_periodic(x, Nx);
                const int xm = minus_periodic(x, Nx);

                Eigen::Vector2cd v_tm;
                Eigen::Vector2cd v_tp;
                Eigen::Vector2cd v_xm;
                Eigen::Vector2cd v_xp;

                for (int a = 0; a < Ns; ++a)
                {
                    v_tm[a] = sign_tm * psi[fermion_index(s, tm, x, a)];
                    v_tp[a] = sign_tp * psi[fermion_index(s, tp, x, a)];
                    v_xm[a] = psi[fermion_index(s, t, xm, a)];
                    v_xp[a] = psi[fermion_index(s, t, xp, a)];
                }

                const Eigen::Vector2cd term_tm = (I - s1) * v_tm;
                const Eigen::Vector2cd term_tp = (I + s1) * v_tp;
                const Eigen::Vector2cd term_xm = (I - s2) * v_xm;
                const Eigen::Vector2cd term_xp = (I + s2) * v_xp;

                const Complex Ut_dag =
                    std::conj(link(theta, 0, tm, x));
                const Complex Ut = link(theta, 0, t, x);
                const Complex Ux_dag =
                    std::conj(link(theta, 1, t, xm));
                const Complex Ux = link(theta, 1, t, x);

                for (int a = 0; a < Ns; ++a)
                {
                    const int row = fermion_index(s, t, x, a);
                    out[row] += -0.5 * Ut_dag * term_tm[a];
                    out[row] += -0.5 * Ut * term_tp[a];
                    out[row] += -0.5 * Ux_dag * term_xm[a];
                    out[row] += -0.5 * Ux * term_xp[a];
                }

                // Exact componentwise fifth-direction terms from D_DWF_dagger.
                const int row_spin1 = fermion_index(s, t, x, 1);
                if (s > 0)
                    out[row_spin1] +=
                        -psi[fermion_index(s - 1, t, x, 1)];
                else
                    out[row_spin1] +=
                        mf * psi[fermion_index(N5 - 1, t, x, 1)];

                const int row_spin0 = fermion_index(s, t, x, 0);
                if (s < N5 - 1)
                    out[row_spin0] +=
                        -psi[fermion_index(s + 1, t, x, 0)];
                else
                    out[row_spin0] +=
                        mf * psi[fermion_index(0, t, x, 0)];
            }
        }
    }

    return out;
}

inline VectorC apply_DD_dagger(const Gauge& theta, const VectorC& psi)
{
    return apply_D(theta, apply_D_dagger(theta, psi));
}

inline VectorC apply_D_dagger_D(const Gauge& theta, const VectorC& psi)
{
    return apply_D_dagger(theta, apply_D(theta, psi));
}
