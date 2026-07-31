#pragma once

#include "parameters.hpp"

inline int fermion_index(int s, int t, int x, int spin)
{
    return (((s * Nt + t) * Nx + x) * Ns + spin);
}

inline int gauge_index(int mu, int t, int x)
{
    return ((mu * Nt + t) * Nx + x);
}

inline int plus_periodic(int x, int extent)
{
    return (x + 1) % extent;
}

inline int minus_periodic(int x, int extent)
{
    return (x - 1 + extent) % extent;
}
