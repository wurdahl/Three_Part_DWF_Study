#include "correlator.hpp"
#include "npy_io.hpp"
#include "parameters.hpp"
#include "svg_plot.hpp"
#include "types.hpp"

#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>

int main()
{
    try
    {
        const NpyArray array =
            load_npy_float64(analysis_configs_filename);

        if (array.shape.size() != 4
            || array.shape[1] != 2
            || array.shape[2] != static_cast<std::size_t>(Nt)
            || array.shape[3] != static_cast<std::size_t>(Nx))
        {
            throw std::runtime_error(
                "analysis_configs.npy has the wrong shape");
        }

        const int configuration_count =
            static_cast<int>(array.shape[0]);

        if (analysis_threads > 0)
            omp_set_num_threads(analysis_threads);

        std::vector<Corr> CPP_all(configuration_count);
        std::vector<Corr> CJ5_all(configuration_count);
        std::vector<ComplexCorr> loops_all(configuration_count);
        std::vector<std::vector<Corr>> momentum_CPP_all(
            configuration_count);

#pragma omp parallel for schedule(dynamic)
        for (int config = 0;
             config < configuration_count;
             ++config)
        {
#pragma omp critical(progress_output)
            std::cout
                << "measuring config "
                << config + 1
                << "/" << configuration_count
                << '\n';

            Gauge theta(Ngauge);

            const std::size_t offset =
                static_cast<std::size_t>(config) * Ngauge;

            for (int i = 0; i < Ngauge; ++i)
                theta[i] = array.data[offset + i];

            CorrelatorPair pair = compute_CPP_CJ5(theta);
            ComplexCorr loop = estimate_pseudoscalar_loop_density(
                theta, eta_random_seed
                    + static_cast<unsigned long long>(config) * 104729ULL);
            std::vector<Corr> momentum_CPP =
                compute_momentum_CPP(theta);

            CPP_all[config] = std::move(pair.CPP);
            CJ5_all[config] = std::move(pair.CJ5);
            loops_all[config] = std::move(loop);
            momentum_CPP_all[config] = std::move(momentum_CPP);
        }

        Corr CPP_average(Nt, 0.0);
        Corr CJ5_average(Nt, 0.0);

        for (int config = 0;
             config < configuration_count;
             ++config)
        {
            for (int t = 0; t < Nt; ++t)
            {
                CPP_average[t] += CPP_all[config][t];
                CJ5_average[t] += CJ5_all[config][t];
            }
        }

        for (int t = 0; t < Nt; ++t)
        {
            CPP_average[t] /= configuration_count;
            CJ5_average[t] /= configuration_count;
        }

        Corr residual_mass(Nt, 0.0);

        for (int t = 0; t < Nt; ++t)
            residual_mass[t] =
                CJ5_average[t] / CPP_average[t];

        double residual_mass_average = 0.0;
        int residual_mass_count = 0;

        for (int t = 3; t < Nt - 3; ++t)
        {
            residual_mass_average += residual_mass[t];
            ++residual_mass_count;
        }

        residual_mass_average /= residual_mass_count;

        // Vacuum-subtracted disconnected Wick contraction, averaged over all
        // time origins. Subtraction is performed across gauge configurations.
        ComplexCorr loop_mean(Nt * Nx, Complex(0.0, 0.0));
        for (const ComplexCorr& loop : loops_all)
            for (int site = 0; site < Nt * Nx; ++site)
                loop_mean[site] +=
                    loop[site] / static_cast<double>(configuration_count);

        std::vector<Corr> disconnected_all(
            configuration_count, Corr(Nt, 0.0));
        std::vector<Corr> eta_all(
            configuration_count, Corr(Nt, 0.0));
        Corr disconnected_average(Nt, 0.0);
        Corr eta_average(Nt, 0.0);

        for (int config = 0; config < configuration_count; ++config)
        {
            for (int dt = 0; dt < Nt; ++dt)
            {
                double value = 0.0;
                for (int t0 = 0; t0 < Nt; ++t0)
                {
                    const int t1 = (t0 + dt) % Nt;
                    Complex loop1 = 0.0;
                    Complex loop0 = 0.0;
                    for (int x = 0; x < Nx; ++x)
                    {
                        loop1 += loops_all[config][t1 * Nx + x]
                               - loop_mean[t1 * Nx + x];
                        loop0 += loops_all[config][t0 * Nx + x]
                               - loop_mean[t0 * Nx + x];
                    }
                    value += std::real(loop1 * std::conj(loop0));
                }
                value /= static_cast<double>(Nt);
                disconnected_all[config][dt] = value;
                eta_all[config][dt] =
                    momentum_CPP_all[config][0][dt]
                    - eta_flavors * value;
                disconnected_average[dt] +=
                    value / static_cast<double>(configuration_count);
                eta_average[dt] +=
                    eta_all[config][dt]
                    / static_cast<double>(configuration_count);
            }
        }

        write_two_series_svg(
            "output/domain_wall/residual_mass_correlators.svg",
            "Residual mass correlators",
            "correlator magnitude",
            CPP_average,
            CJ5_average,
            "|C_PP(t)|",
            "|C_J5(t)|",
            true);

        std::ofstream momentum_csv(
            "output/domain_wall/momentum_correlators.csv");
        momentum_csv
            << "momentum_index,p,p_squared,t,pion,disconnected,eta\n";

        for (int n = 0; n <= max_momentum; ++n)
        {
            const double p = 2.0 * pi * n / Nx;
            std::vector<ComplexCorr> loop_p(
                configuration_count,
                ComplexCorr(Nt, Complex(0.0, 0.0)));
            ComplexCorr loop_p_mean(Nt, Complex(0.0, 0.0));

            for (int config = 0; config < configuration_count; ++config)
            {
                for (int t = 0; t < Nt; ++t)
                {
                    for (int x = 0; x < Nx; ++x)
                    {
                        const Complex phase =
                            std::exp(Complex(0.0, -p * x));
                        loop_p[config][t] +=
                            phase * loops_all[config][t * Nx + x];
                    }
                    loop_p_mean[t] += loop_p[config][t]
                        / static_cast<double>(configuration_count);
                }
            }

            for (int dt = 0; dt < Nt; ++dt)
            {
                double pion = 0.0;
                double disconnected = 0.0;
                for (int config = 0; config < configuration_count; ++config)
                {
                    pion += momentum_CPP_all[config][n][dt]
                        / static_cast<double>(configuration_count);
                    double one_config = 0.0;
                    for (int t0 = 0; t0 < Nt; ++t0)
                    {
                        const int t1 = (t0 + dt) % Nt;
                        one_config += std::real(
                            (loop_p[config][t1] - loop_p_mean[t1])
                            * std::conj(
                                loop_p[config][t0] - loop_p_mean[t0]));
                    }
                    disconnected += one_config
                        / static_cast<double>(Nt * configuration_count);
                }
                momentum_csv
                    << n << "," << p << "," << p * p << ","
                    << dt << "," << pion << "," << disconnected
                    << "," << pion - eta_flavors * disconnected
                    << "\n";
            }
        }

        write_single_series_svg(
            "output/domain_wall/residual_mass.svg",
            "m_res(t)",
            "C_J5(t) / C_PP(t)",
            residual_mass,
            residual_mass_average);

        std::ofstream csv("output/domain_wall/correlator.csv");
        csv << "t,CPP_average,CJ5_average,m_res,"
            << "disconnected,eta_singlet\n";

        for (int t = 0; t < Nt; ++t)
        {
            csv
                << t << ","
                << CPP_average[t] << ","
                << CJ5_average[t] << ","
                << residual_mass[t] << ","
                << disconnected_average[t] << ","
                << eta_average[t] << "\n";
        }

        std::ofstream by_config(
            "output/domain_wall/channels_by_config.csv");
        by_config << "config";
        for (int t = 0; t < Nt; ++t)
            by_config << ",pion_t" << t;
        for (int t = 0; t < Nt; ++t)
            by_config << ",disconnected_t" << t;
        for (int t = 0; t < Nt; ++t)
            by_config << ",eta_t" << t;
        by_config << "\n";
        for (int config = 0; config < configuration_count; ++config)
        {
            by_config << config;
            for (double value : CPP_all[config])
                by_config << "," << value;
            for (double value : disconnected_all[config])
                by_config << "," << value;
            for (double value : eta_all[config])
                by_config << "," << value;
            by_config << "\n";
        }

        write_two_series_svg(
            "output/domain_wall/pion_eta_correlators.svg",
            "Pion and eta correlators",
            "correlator magnitude",
            CPP_average,
            eta_average,
            "|C_pion(t)|",
            "|C_eta(t)|",
            true);

        std::cout << std::setprecision(10)
            << "\nall configs = ("
            << trajectories << ", 2, "
            << Nt << ", " << Nx << ")"
            << "\nanalysis configs = ("
            << configuration_count << ", 2, "
            << Nt << ", " << Nx << ")"
            << "\nm_res = "
            << residual_mass_average
            << "\n\nWrote:\n"
            << "  output/domain_wall/residual_mass_correlators.svg\n"
            << "  output/domain_wall/residual_mass.svg\n"
            << "  output/domain_wall/correlator.csv\n";
        std::cout
            << "  output/domain_wall/channels_by_config.csv\n"
            << "  output/domain_wall/pion_eta_correlators.svg\n";
        std::cout
            << "  output/domain_wall/momentum_correlators.csv\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
