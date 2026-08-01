#include "dwf_operator.hpp"
#include "gauge.hpp"
#include "npy_io.hpp"
#include "parameters.hpp"
#include "pseudofermion.hpp"
#include "svg_plot.hpp"
#include "types.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <malloc.h>
#include <omp.h>

int main()
{
    try
    {
        omp_set_num_threads(hmc_threads > 0 ? hmc_threads : 1);

        // Field-sized vectors are allocated and freed once per operator
        // application; keeping them below the mmap threshold lets glibc
        // recycle the same arena block instead of remapping pages.
        mallopt(M_MMAP_THRESHOLD, 512 * 1024 * 1024);
        mallopt(M_TRIM_THRESHOLD, 512 * 1024 * 1024);

        std::filesystem::create_directories("output/domain_wall");
        std::mt19937_64 rng(random_seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Exact Python cold start: np.ones((2, Nt, Nx)).
        Gauge theta0(Ngauge, 1.0);

        std::vector<double> configs;
        configs.reserve(
            static_cast<std::size_t>(trajectories) * Ngauge);

        std::vector<double> analysis_configs;
        analysis_configs.reserve(
            static_cast<std::size_t>(
                (trajectories - thermalization_cut + thin - 1) / thin)
            * Ngauge);

        std::vector<double> plaquettes(trajectories, 0.0);
        std::vector<int> accepts(trajectories, 0);
        std::vector<double> delta_h_values(trajectories, 0.0);
        std::vector<double> acceptance_probabilities(
            trajectories, 0.0);

        int accepted_count = 0;

        for (int trajectory = 0;
             trajectory < trajectories;
             ++trajectory)
        {
            // eta_guess = None at the start of every trajectory.
            bool have_eta_guess = false;
            VectorC eta_guess;

            VectorC chi(Ndof);

            for (int i = 0; i < Ndof; ++i)
            {
                chi[i] = Complex(normal(rng), normal(rng))
                       / std::sqrt(2.0);
            }

            const VectorC phi = apply_D(theta0, chi);

            Gauge momentum(Ngauge);

            for (double& p : momentum)
                p = normal(rng);

            const GaugeActionForce gauge_initial =
                gauge_action_force(theta0);

            const PseudofermionActionForce pf_initial =
                pseudofermion_action_force(
                    theta0,
                    phi,
                    nullptr);

            eta_guess = pf_initial.eta;
            have_eta_guess = true;

            double kinetic_initial = 0.0;

            for (double p : momentum)
                kinetic_initial += 0.5 * p * p;

            const double H0 =
                kinetic_initial
                + gauge_initial.action
                + pf_initial.action;

            Gauge theta = theta0;
            Gauge force(Ngauge);

            for (int i = 0; i < Ngauge; ++i)
            {
                force[i] =
                    gauge_initial.force[i]
                    + pf_initial.force[i];
            }

            for (int i = 0; i < Ngauge; ++i)
                momentum[i] -= 0.5 * epsilon * force[i];

            double gauge_action = gauge_initial.action;
            double pseudofermion_action = pf_initial.action;

            for (int step = 0;
                 step < leapfrog_steps;
                 ++step)
            {
                for (int i = 0; i < Ngauge; ++i)
                    theta[i] += epsilon * momentum[i];

                const GaugeActionForce gauge_current =
                    gauge_action_force(theta);

                const PseudofermionActionForce pf_current =
                    pseudofermion_action_force(
                        theta,
                        phi,
                        have_eta_guess ? &eta_guess : nullptr);

                eta_guess = pf_current.eta;
                have_eta_guess = true;

                gauge_action = gauge_current.action;
                pseudofermion_action = pf_current.action;

                for (int i = 0; i < Ngauge; ++i)
                {
                    force[i] =
                        gauge_current.force[i]
                        + pf_current.force[i];
                }

                if (step != leapfrog_steps - 1)
                {
                    for (int i = 0; i < Ngauge; ++i)
                        momentum[i] -= epsilon * force[i];
                }
            }

            for (int i = 0; i < Ngauge; ++i)
                momentum[i] -= 0.5 * epsilon * force[i];

            double kinetic_final = 0.0;

            for (double p : momentum)
                kinetic_final += 0.5 * p * p;

            const double H =
                kinetic_final
                + gauge_action
                + pseudofermion_action;

            const double delta_h = H - H0;
            const double acceptance_probability =
                std::min(std::exp(-delta_h), 1.0);

            const bool accepted =
                uniform(rng) < acceptance_probability;

            if (accepted)
            {
                theta0 = theta;
                ++accepted_count;
            }

            const double avg_plaquette =
                average_plaquette(theta0);

            std::cout
                << "traj " << std::setw(4) << trajectory
                << "  accepted=" << accepted
                << "  acc_prob=" << std::fixed
                << std::setprecision(3)
                << acceptance_probability
                << "  dH=" << std::scientific
                << std::setprecision(3)
                << delta_h
                << "  plaq=" << std::fixed
                << std::setprecision(5)
                << avg_plaquette
                << '\n';

            configs.insert(
                configs.end(),
                theta0.begin(),
                theta0.end());

            if (trajectory >= thermalization_cut
                && (trajectory - thermalization_cut) % thin == 0)
            {
                analysis_configs.insert(
                    analysis_configs.end(),
                    theta0.begin(),
                    theta0.end());
            }

            plaquettes[trajectory] = avg_plaquette;
            accepts[trajectory] = accepted ? 1 : 0;
            delta_h_values[trajectory] = delta_h;
            acceptance_probabilities[trajectory] =
                acceptance_probability;
        }

        const std::size_t analysis_count =
            analysis_configs.size() / Ngauge;

        save_npy_float64(
            configs_filename,
            {
                static_cast<std::size_t>(trajectories),
                2,
                static_cast<std::size_t>(Nt),
                static_cast<std::size_t>(Nx)
            },
            configs);

        save_npy_float64(
            analysis_configs_filename,
            {
                analysis_count,
                2,
                static_cast<std::size_t>(Nt),
                static_cast<std::size_t>(Nx)
            },
            analysis_configs);

        std::ofstream history(history_filename);
        history
            << "trajectory,accepted,acceptance_probability,"
            << "delta_h,plaquette\n";

        for (int i = 0; i < trajectories; ++i)
        {
            history
                << i << ","
                << accepts[i] << ","
                << acceptance_probabilities[i] << ","
                << delta_h_values[i] << ","
                << plaquettes[i] << "\n";
        }

        write_single_series_svg(
            "output/domain_wall/gauge_history.svg",
            "Gauge history",
            "average plaquette",
            plaquettes,
            std::numeric_limits<double>::quiet_NaN(),
            thermalization_cut);

        std::cout
            << "\nacceptance rate = "
            << static_cast<double>(accepted_count)
               / trajectories
            << "\nall configs = ("
            << trajectories << ", 2, "
            << Nt << ", " << Nx << ")"
            << "\nanalysis configs = ("
            << analysis_count << ", 2, "
            << Nt << ", " << Nx << ")\n"
            << "saved configs.npy\n"
            << "saved analysis_configs.npy\n"
            << "saved run_info_history.csv\n"
            << "saved output/domain_wall/gauge_history.svg\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
