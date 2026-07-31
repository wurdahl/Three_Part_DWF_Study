#pragma once

#include "cg_solver.hpp"
#include "dwf_operator.hpp"
#include "indexing.hpp"
#include "parameters.hpp"
#include "spin.hpp"
#include "types.hpp"

#include <cmath>
#include <stdexcept>

struct PseudofermionActionForce
{
    double action;
    Gauge force;
    VectorC eta;
    int iterations;
    double relative_residual;
};

inline PseudofermionActionForce pseudofermion_action_force(
    const Gauge& theta,
    const VectorC& phi,
    const VectorC* eta_guess)
{
    const auto apply_M = [&](const VectorC& v)
    {
        return apply_DD_dagger(theta, v);
    };

    const CgResult solve = conjugate_gradient(
        apply_M,
        phi,
        eta_guess,
        pseudofermion_rtol,
        pseudofermion_maxiter);

    if (!solve.converged)
        throw std::runtime_error(
            "Pseudofermion CG failed to converge");

    const VectorC& eta = solve.x;
    const double action = std::real(phi.dot(eta));
    const VectorC xi = apply_D_dagger(theta, eta);

    Gauge force(Ngauge, 0.0);

    const Eigen::Matrix2cd I = identity2();
    const Eigen::Matrix2cd s1 = sigma1();
    const Eigen::Matrix2cd s2 = sigma2();

    std::vector<Complex> t_fwd(Nt * Nx, Complex(0.0, 0.0));
    std::vector<Complex> t_bwd_out(Nt * Nx, Complex(0.0, 0.0));
    std::vector<Complex> x_fwd(Nt * Nx, Complex(0.0, 0.0));
    std::vector<Complex> x_bwd_out(Nt * Nx, Complex(0.0, 0.0));

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

            Complex tf = 0.0;
            Complex tb = 0.0;
            Complex xf = 0.0;
            Complex xb = 0.0;

            for (int s = 0; s < N5; ++s)
            {
                Eigen::Vector2cd xi_tp;
                Eigen::Vector2cd xi_tm;
                Eigen::Vector2cd xi_xp;
                Eigen::Vector2cd xi_xm;
                Eigen::Vector2cd eta_here;

                for (int a = 0; a < Ns; ++a)
                {
                    xi_tp[a] =
                        sign_tp * xi[fermion_index(s, tp, x, a)];
                    xi_tm[a] =
                        sign_tm * xi[fermion_index(s, tm, x, a)];
                    xi_xp[a] =
                        xi[fermion_index(s, t, xp, a)];
                    xi_xm[a] =
                        xi[fermion_index(s, t, xm, a)];
                    eta_here[a] =
                        eta[fermion_index(s, t, x, a)];
                }

                const Eigen::Vector2cd term_tf = (I - s1) * xi_tp;
                const Eigen::Vector2cd term_tb = (I + s1) * xi_tm;
                const Eigen::Vector2cd term_xf = (I - s2) * xi_xp;
                const Eigen::Vector2cd term_xb = (I + s2) * xi_xm;

                const Complex Ut =
                    std::exp(Complex(0.0, theta[gauge_index(0, t, x)]));
                const Complex Ut_dag =
                    std::conj(std::exp(
                        Complex(0.0, theta[gauge_index(0, tm, x)])));

                const Complex Ux =
                    std::exp(Complex(0.0, theta[gauge_index(1, t, x)]));
                const Complex Ux_dag =
                    std::conj(std::exp(
                        Complex(0.0, theta[gauge_index(1, t, xm)])));

                for (int a = 0; a < Ns; ++a)
                {
                    const Complex eta_star = std::conj(eta_here[a]);

                    tf += eta_star
                        * (Complex(0.0, -0.5) * Ut * term_tf[a]);

                    tb += eta_star
                        * (Complex(0.0, 0.5) * Ut_dag * term_tb[a]);

                    xf += eta_star
                        * (Complex(0.0, -0.5) * Ux * term_xf[a]);

                    xb += eta_star
                        * (Complex(0.0, 0.5) * Ux_dag * term_xb[a]);
                }
            }

            t_fwd[t * Nx + x] = tf;
            t_bwd_out[t * Nx + x] = tb;
            x_fwd[t * Nx + x] = xf;
            x_bwd_out[t * Nx + x] = xb;
        }
    }

    // Exact np.roll(..., shift=-1) shifts from the Python force.
    for (int t = 0; t < Nt; ++t)
    {
        const int tp = plus_periodic(t, Nt);

        for (int x = 0; x < Nx; ++x)
        {
            const int xp = plus_periodic(x, Nx);

            const Complex t_bwd = t_bwd_out[tp * Nx + x];
            const Complex x_bwd = x_bwd_out[t * Nx + xp];

            force[gauge_index(0, t, x)] =
                -2.0 * std::real(t_fwd[t * Nx + x] + t_bwd);

            force[gauge_index(1, t, x)] =
                -2.0 * std::real(x_fwd[t * Nx + x] + x_bwd);
        }
    }

    return {
        action,
        std::move(force),
        eta,
        solve.iterations,
        solve.relative_residual
    };
}
