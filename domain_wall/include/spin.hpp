#pragma once

#include "types.hpp"

inline Eigen::Matrix2cd sigma1()
{
    Eigen::Matrix2cd m;
    m << 0.0, 1.0,
         1.0, 0.0;
    return m;
}

inline Eigen::Matrix2cd sigma2()
{
    Eigen::Matrix2cd m;
    m << 0.0, Complex(0.0, -1.0),
         Complex(0.0, 1.0), 0.0;
    return m;
}

inline Eigen::Matrix2cd sigma3()
{
    Eigen::Matrix2cd m;
    m << 1.0, 0.0,
         0.0, -1.0;
    return m;
}

inline Eigen::Matrix2cd identity2()
{
    return Eigen::Matrix2cd::Identity();
}

inline Eigen::Matrix2cd projector_R()
{
    return 0.5 * (identity2() + sigma3());
}

inline Eigen::Matrix2cd projector_L()
{
    return 0.5 * (identity2() - sigma3());
}
