#include "distillation.hpp"
#include "npy_io.hpp"
#include "parameters.hpp"
#include "types.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

// Measures the distillation objects (Laplacian eigenvalues, perambulators,
// momentum and derivative elementals) on every retained configuration and
// writes them as float64 .npy files with a trailing re/im axis. The Wick
// contractions themselves happen in scripts/distillation_contract.py.

namespace
{

void store(std::vector<double>& flat, std::size_t base, Complex value)
{
    flat[base] = value.real();
    flat[base + 1] = value.imag();
}

}

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
        const int n = distillation_vector_count();
        const std::vector<int> t_sources = distillation_t_sources();
        const int source_count = static_cast<int>(t_sources.size());
        if (max_momentum < 0 || max_momentum > Nx / 2)
            throw std::runtime_error(
                "analysis.max_momentum is outside 0..Nx/2");
        const int momentum_count = max_momentum + 1;

        if (analysis_threads > 0)
            omp_set_num_threads(analysis_threads);

        std::cout
            << "distillation: " << configuration_count << " configs, "
            << n << " vectors, " << source_count << " source times, "
            << momentum_count << " momenta\n"
            << "solves per config = "
            << source_count * n * Ns << "\n";

        const std::size_t tau_config_stride =
            static_cast<std::size_t>(source_count) * Nt
            * (Ns * n) * (Ns * n) * 2;
        const std::size_t elem_config_stride =
            static_cast<std::size_t>(momentum_count) * Nt * n * n * 2;

        std::vector<double> tau_flat(
            configuration_count * tau_config_stride);
        std::vector<double> momentum_flat(
            configuration_count * elem_config_stride);
        std::vector<double> derivative_flat(
            configuration_count * elem_config_stride);
        std::vector<double> eigenvalue_flat(
            static_cast<std::size_t>(configuration_count) * Nt * n);

#pragma omp parallel for schedule(dynamic)
        for (int config = 0; config < configuration_count; ++config)
        {
#pragma omp critical(progress_output)
            std::cout
                << "distilling config " << config + 1
                << "/" << configuration_count << '\n';

            Gauge theta(Ngauge);
            const std::size_t offset =
                static_cast<std::size_t>(config) * Ngauge;
            for (int i = 0; i < Ngauge; ++i)
                theta[i] = array.data[offset + i];

            const DistillationBasis basis =
                build_distillation_basis(theta, n);
            const Elementals elementals =
                build_elementals(theta, basis, momentum_count);

            for (int t = 0; t < Nt; ++t)
                for (int k = 0; k < n; ++k)
                    eigenvalue_flat[
                        (static_cast<std::size_t>(config) * Nt + t) * n + k]
                        = basis.eigenvalues[t][k];

            for (int p = 0; p < momentum_count; ++p)
            {
                for (int t = 0; t < Nt; ++t)
                {
                    for (int k = 0; k < n; ++k)
                    {
                        for (int l = 0; l < n; ++l)
                        {
                            const std::size_t base =
                                (config * elem_config_stride)
                                + ((((static_cast<std::size_t>(p) * Nt + t)
                                     * n + k) * n + l) * 2);
                            store(momentum_flat, base,
                                  elementals.momentum[p][t](k, l));
                            store(derivative_flat, base,
                                  elementals.derivative[p][t](k, l));
                        }
                    }
                }
            }

            for (int source = 0; source < source_count; ++source)
            {
                const std::vector<Eigen::MatrixXcd> tau =
                    compute_perambulator(
                        theta, basis, t_sources[source]);
                for (int t = 0; t < Nt; ++t)
                {
                    for (int row = 0; row < Ns * n; ++row)
                    {
                        for (int column = 0; column < Ns * n; ++column)
                        {
                            const std::size_t base =
                                (config * tau_config_stride)
                                + ((((static_cast<std::size_t>(source) * Nt
                                      + t) * (Ns * n) + row) * (Ns * n)
                                    + column) * 2);
                            store(tau_flat, base, tau[t](row, column));
                        }
                    }
                }
            }
        }

        const std::string directory = "output/domain_wall/distillation";
        std::filesystem::create_directories(directory);

        const std::size_t nn = static_cast<std::size_t>(n);
        const std::size_t spin_n = static_cast<std::size_t>(Ns) * nn;
        save_npy_float64(
            directory + "/perambulators.npy",
            {static_cast<std::size_t>(configuration_count),
             static_cast<std::size_t>(source_count),
             static_cast<std::size_t>(Nt), spin_n, spin_n, 2},
            tau_flat);
        save_npy_float64(
            directory + "/elementals_momentum.npy",
            {static_cast<std::size_t>(configuration_count),
             static_cast<std::size_t>(momentum_count),
             static_cast<std::size_t>(Nt), nn, nn, 2},
            momentum_flat);
        save_npy_float64(
            directory + "/elementals_derivative.npy",
            {static_cast<std::size_t>(configuration_count),
             static_cast<std::size_t>(momentum_count),
             static_cast<std::size_t>(Nt), nn, nn, 2},
            derivative_flat);
        save_npy_float64(
            directory + "/laplacian_eigenvalues.npy",
            {static_cast<std::size_t>(configuration_count),
             static_cast<std::size_t>(Nt), nn},
            eigenvalue_flat);
        std::vector<double> sources_as_double(
            t_sources.begin(), t_sources.end());
        save_npy_float64(
            directory + "/t_sources.npy",
            {static_cast<std::size_t>(source_count)},
            sources_as_double);

        std::cout
            << "\nWrote:\n"
            << "  " << directory << "/perambulators.npy\n"
            << "  " << directory << "/elementals_momentum.npy\n"
            << "  " << directory << "/elementals_derivative.npy\n"
            << "  " << directory << "/laplacian_eigenvalues.npy\n"
            << "  " << directory << "/t_sources.npy\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
