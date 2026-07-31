#pragma once

#include "indexing.hpp"
#include "parameters.hpp"
#include "types.hpp"

#include <cmath>
#include <vector>

inline SparseC build_wilson_matrix(const Gauge& theta)
{
    const Mat2 I=Mat2::Identity();
    Mat2 gamma_t; gamma_t<<0.0,1.0,1.0,0.0;
    Mat2 gamma_x; gamma_x<<0.0,Complex(0,-1),Complex(0,1),0.0;

    std::vector<TripletC> entries;
    entries.reserve(Ntot*10);

    auto add=[&](int rt,int rx,int ct,int cx,
                 const Mat2& A,Complex coeff)
    {
        for(int a=0;a<Ns;++a)
            for(int b=0;b<Ns;++b)
            {
                Complex value=coeff*A(a,b);
                if(std::abs(value)>0.0)
                    entries.emplace_back(site_index(rt,rx,a),
                                         site_index(ct,cx,b),value);
            }
    };

    for(int t=0;t<Nt;++t)
    for(int x=0;x<Nx;++x)
    {
        add(t,x,t,x,I,2.0+m0);

        int tp=(t+1)%Nt;
        double bctp=(t==Nt-1)?time_boundary_sign:1.0;
        Complex Ut=std::exp(Complex(0,theta[link_index(0,t,x)]));
        add(t,x,tp,x,I-gamma_t,-0.5*bctp*Ut);

        int tm=(t-1+Nt)%Nt;
        double bctm=(t==0)?time_boundary_sign:1.0;
        Complex Utd=std::exp(Complex(0,-theta[link_index(0,tm,x)]));
        add(t,x,tm,x,I+gamma_t,-0.5*bctm*Utd);

        int xp=(x+1)%Nx;
        double bcxp=(x==Nx-1)?space_boundary_sign:1.0;
        Complex Ux=std::exp(Complex(0,theta[link_index(1,t,x)]));
        add(t,x,t,xp,I-gamma_x,-0.5*bcxp*Ux);

        int xm=(x-1+Nx)%Nx;
        double bcxm=(x==0)?space_boundary_sign:1.0;
        Complex Uxd=std::exp(Complex(0,-theta[link_index(1,t,xm)]));
        add(t,x,t,xm,I+gamma_x,-0.5*bcxm*Uxd);
    }

    SparseC D(Ntot,Ntot);
    D.setFromTriplets(entries.begin(),entries.end());
    D.makeCompressed();
    return D;
}
