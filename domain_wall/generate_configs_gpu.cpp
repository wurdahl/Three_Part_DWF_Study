// GPU generator with a fully device-resident trajectory. The RNG stream
// (chi, momentum, Metropolis draw) is identical to generate_configs.cpp,
// so DWF_GPU_FP64_MD=1 reproduces the CPU chain up to last-bit link-phase
// rounding. Per trajectory the host uploads chi and momentum, downloads
// three Hamiltonian scalars and the accepted configuration; everything
// else stays on the GPU.

#include "gauge.hpp"
#include "gpu_hmc.hpp"
#include "npy_io.hpp"
#include "parameters.hpp"
#include "svg_plot.hpp"
#include "types.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main()
{
    try
    {
        std::filesystem::create_directories("output/domain_wall");
        std::mt19937_64 rng(random_seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        const bool fp64_md =
            std::getenv("DWF_GPU_FP64_MD") != nullptr
            && std::string(std::getenv("DWF_GPU_FP64_MD")) == "1";

        GpuHmc gpu(Nt, Nx, N5, M5, mf, beta);
        std::cout << "GPU: " << gpu.device_name()
                  << (fp64_md ? " (fp64 MD)" : " (fp32 MD)") << "\n";

        Gauge theta0(Ngauge, 1.0);
        gpu.set_theta0(theta0.data());

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

        std::vector<Complex> chi(Ndof);
        Gauge momentum(Ngauge);

        for (int trajectory = 0;
             trajectory < trajectories;
             ++trajectory)
        {
            for (int i = 0; i < Ndof; ++i)
            {
                chi[i] = Complex(normal(rng), normal(rng))
                       / std::sqrt(2.0);
            }
            gpu.begin_trajectory(chi.data());

            for (double& p : momentum)
                p = normal(rng);
            gpu.upload_momentum(momentum.data());

            const GpuHmc::TrajectoryStats stats = gpu.run_leapfrog(
                leapfrog_steps, epsilon,
                pseudofermion_rtol, pseudofermion_maxiter, fp64_md);

            const double delta_h = stats.H - stats.H0;
            const double acceptance_probability =
                std::min(std::exp(-delta_h), 1.0);

            const bool accepted =
                uniform(rng) < acceptance_probability;

            if (accepted)
                ++accepted_count;

            gpu.finish_trajectory(accepted, theta0.data());

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
