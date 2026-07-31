#pragma once

#include "types.hpp"
#include "parameters.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

inline Corr mean_correlator(const std::vector<Corr>& data, int leave_out = -1)
{
    Corr mean(Nt, 0.0);
    int count = 0;

    for (int config = 0; config < static_cast<int>(data.size()); ++config)
    {
        if (config == leave_out)
            continue;
        for (int t = 0; t < Nt; ++t)
            mean[t] += data[config][t];
        ++count;
    }

    if (count == 0)
        throw std::runtime_error("No configurations in mean_correlator");

    for (double& value : mean)
        value /= static_cast<double>(count);
    return mean;
}

inline Corr cosh_effective_mass(const Corr& correlator)
{
    Corr effective_mass(Nt, std::numeric_limits<double>::quiet_NaN());

    for (int t = 1; t < Nt - 1; ++t)
    {
        if (!std::isfinite(correlator[t]) || correlator[t] <= 0.0)
            continue;

        const double ratio =
            (correlator[t - 1] + correlator[t + 1]) / (2.0 * correlator[t]);

        if (std::isfinite(ratio) && ratio >= 1.0)
            effective_mass[t] = std::acosh(ratio);
    }

    return effective_mass;
}

struct PlateauWindow
{
    int first_t = -1;
    int last_t = -1;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double relative_drift = std::numeric_limits<double>::quiet_NaN();
};

inline PlateauWindow find_flat_plateau(const Corr& effective_mass,
                                       bool print_diagnostics)
{
    const int last_independent_t = Nt / 2;

    if (print_diagnostics)
    {
        std::cout << "\nAutomatic contiguous-window plateau search\n"
                  << "-------------------------------------------\n"
                  << "Window size: " << plateau_window_size << '\n'
                  << "Maximum relative drift: "
                  << 100.0 * plateau_flatness_tolerance << "%\n\n"
                  << "Candidate windows:\n";
    }

    for (int first_t = earliest_plateau_time;
         first_t + plateau_window_size - 1 <= last_independent_t;
         ++first_t)
    {
        const int last_t = first_t + plateau_window_size - 1;
        bool all_valid = true;
        double sum = 0.0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();

        for (int t = first_t; t <= last_t; ++t)
        {
            const double value = effective_mass[t];
            if (!std::isfinite(value) || value <= 0.0)
            {
                all_valid = false;
                break;
            }
            sum += value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        if (!all_valid)
        {
            if (print_diagnostics)
                std::cout << "t = " << first_t << " through " << last_t
                          << "   INVALID\n";
            continue;
        }

        const double mean = sum / plateau_window_size;
        const double relative_drift = (maximum - minimum) / mean;
        const bool accepted = relative_drift <= plateau_flatness_tolerance;

        if (print_diagnostics)
            std::cout << "t = " << first_t << " through " << last_t
                      << "   mean = " << mean
                      << "   drift = " << 100.0 * relative_drift << "%"
                      << (accepted ? "   ACCEPTED" : "   rejected") << '\n';

        if (accepted)
            return {first_t, last_t, mean, relative_drift};
    }

    throw std::runtime_error(
        "No contiguous effective-mass window passed the flatness test");
}

inline double average_over_window(const Corr& effective_mass,
                                  const PlateauWindow& window)
{
    double sum = 0.0;
    for (int t = window.first_t; t <= window.last_t; ++t)
    {
        if (!std::isfinite(effective_mass[t]))
            throw std::runtime_error(
                "A jackknife sample is invalid inside the selected plateau");
        sum += effective_mass[t];
    }
    return sum / static_cast<double>(window.last_t - window.first_t + 1);
}

inline double jackknife_error(const std::vector<double>& samples)
{
    if (samples.size() < 2)
        throw std::runtime_error("Need at least two jackknife samples");

    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

    double squared_deviation_sum = 0.0;
    for (double value : samples)
        squared_deviation_sum += (value - mean) * (value - mean);

    return std::sqrt((samples.size() - 1.0) / samples.size()
                     * squared_deviation_sum);
}
