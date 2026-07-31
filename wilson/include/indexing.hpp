#pragma once
#include "parameters.hpp"
#include <cmath>

inline int site_index(int t,int x,int spin)
{
    return ((t*Nx+x)*Ns+spin);
}
inline int link_index(int mu,int t,int x)
{
    return (mu*Nt+t)*Nx+x;
}
inline double wrap_angle(double a)
{
    a=std::fmod(a+pi,2.0*pi);
    if(a<0.0) a+=2.0*pi;
    return a-pi;
}
