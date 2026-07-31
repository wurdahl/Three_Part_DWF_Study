#pragma once

#include "indexing.hpp"
#include "parameters.hpp"
#include "types.hpp"

#include <cmath>

struct GaugeActionForce
{
    double action;
    Gauge force;
};

inline double plaquette_angle(const Gauge& theta, int t, int x)
{
    const int tp = plus_periodic(t, Nt);
    const int xp = plus_periodic(x, Nx);

    return theta[gauge_index(0, t, x)]
         + theta[gauge_index(1, tp, x)]
         - theta[gauge_index(0, t, xp)]
         - theta[gauge_index(1, t, x)];
}

inline GaugeActionForce gauge_action_force(const Gauge& theta)
{
    GaugeActionForce result;
    result.action = 0.0;
    result.force.assign(Ngauge, 0.0);

    std::vector<double> sin_p(Nt * Nx, 0.0);

    for (int t = 0; t < Nt; ++t)
    {
        for (int x = 0; x < Nx; ++x)
        {
            const double p = plaquette_angle(theta, t, x);
            result.action += beta * (1.0 - std::cos(p));
            sin_p[t * Nx + x] = std::sin(p);
        }
    }

    for (int t = 0; t < Nt; ++t)
    {
        const int tm = minus_periodic(t, Nt);

        for (int x = 0; x < Nx; ++x)
        {
            const int xm = minus_periodic(x, Nx);
            const double here = sin_p[t * Nx + x];

            result.force[gauge_index(0, t, x)] =
                beta * (here - sin_p[t * Nx + xm]);

            result.force[gauge_index(1, t, x)] =
                beta * (sin_p[tm * Nx + x] - here);
        }
    }

    return result;
}

inline double average_plaquette(const Gauge& theta)
{
    double sum = 0.0;

    for (int t = 0; t < Nt; ++t)
        for (int x = 0; x < Nx; ++x)
            sum += std::cos(plaquette_angle(theta, t, x));

    return sum / static_cast<double>(Nt * Nx);
}
