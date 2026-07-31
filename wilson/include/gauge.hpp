#pragma once

#include "cg_solver.hpp"
#include "indexing.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "wilson_operator.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <vector>

inline std::mt19937_64 rng(random_seed);
inline std::normal_distribution<double> gaussian(0.0,1.0);
inline std::uniform_real_distribution<double> uniform01(0.0,1.0);

inline double plaquette_angle(const Gauge& theta,int t,int x)
{
    int tp=(t+1)%Nt,xp=(x+1)%Nx;
    return theta[link_index(0,t,x)]
         + theta[link_index(1,tp,x)]
         - theta[link_index(0,t,xp)]
         - theta[link_index(1,t,x)];
}

inline double average_plaquette(const Gauge& theta)
{
    double sum=0.0;
    for(int t=0;t<Nt;++t)
        for(int x=0;x<Nx;++x)
            sum+=std::cos(plaquette_angle(theta,t,x));
    return sum/(Nt*Nx);
}

inline double gauge_action_force(const Gauge& theta,Gauge& force)
{
    force.assign(ndim*Nt*Nx,0.0);
    std::vector<double> sp(Nt*Nx);
    double action=0.0;

    for(int t=0;t<Nt;++t)
    for(int x=0;x<Nx;++x)
    {
        double p=plaquette_angle(theta,t,x);
        action+=beta*(1.0-std::cos(p));
        sp[t*Nx+x]=std::sin(p);
    }

    for(int t=0;t<Nt;++t)
    for(int x=0;x<Nx;++x)
    {
        int tm=(t-1+Nt)%Nt,xm=(x-1+Nx)%Nx;
        force[link_index(0,t,x)] =
            beta*(sp[t*Nx+x]-sp[t*Nx+xm]);
        force[link_index(1,t,x)] =
            beta*(sp[tm*Nx+x]-sp[t*Nx+x]);
    }
    return action;
}

// Implements S_pf = phi^\dagger (D D^\dagger)^(-1) phi.
//
// Integrating over phi gives det(D D^\dagger)=|det D|^2.
// Therefore this is a two-degenerate-flavor Wilson ensemble.
inline double pseudofermion_action_force(const Gauge& theta,
                                         const VectorC& phi,
                                         Gauge& force)
{
    SparseC D=build_wilson_matrix(theta);
    VectorC eta=solve_DDdag(D,phi,force_cg_tolerance,
                           force_cg_max_iterations);
    VectorC xi=D.adjoint()*eta;
    double action=std::real(phi.dot(eta));

    force.assign(ndim*Nt*Nx,0.0);

    Mat2 I=Mat2::Identity();
    Mat2 gt; gt<<0.0,1.0,1.0,0.0;
    Mat2 gx; gx<<0.0,Complex(0,-1),Complex(0,1),0.0;

    for(int t=0;t<Nt;++t)
    for(int x=0;x<Nx;++x)
    {
        int tp=(t+1)%Nt,tm=(t-1+Nt)%Nt;
        int xp=(x+1)%Nx,xm=(x-1+Nx)%Nx;

        double bctp=(t==Nt-1)?time_boundary_sign:1.0;
        double bctm=(t==0)?time_boundary_sign:1.0;
        double bcxp=(x==Nx-1)?space_boundary_sign:1.0;
        double bcxm=(x==0)?space_boundary_sign:1.0;

        Complex Ut=std::exp(Complex(0,theta[link_index(0,t,x)]));
        Complex Utd=std::exp(Complex(0,-theta[link_index(0,tm,x)]));
        Complex Ux=std::exp(Complex(0,theta[link_index(1,t,x)]));
        Complex Uxd=std::exp(Complex(0,-theta[link_index(1,t,xm)]));

        Eigen::Vector2cd eta_here,xi_tp,xi_tm,xi_xp,xi_xm;
        for(int a=0;a<2;++a)
        {
            eta_here[a]=eta[site_index(t,x,a)];
            xi_tp[a]=bctp*xi[site_index(tp,x,a)];
            xi_tm[a]=bctm*xi[site_index(tm,x,a)];
            xi_xp[a]=bcxp*xi[site_index(t,xp,a)];
            xi_xm[a]=bcxm*xi[site_index(t,xm,a)];
        }

        Eigen::Vector2cd vt=(-0.5*Complex(0,1)*Ut)*(I-gt)*xi_tp;
        Eigen::Vector2cd vtb=(0.5*Complex(0,1)*Utd)*(I+gt)*xi_tm;
        Eigen::Vector2cd vx=(-0.5*Complex(0,1)*Ux)*(I-gx)*xi_xp;
        Eigen::Vector2cd vxb=(0.5*Complex(0,1)*Uxd)*(I+gx)*xi_xm;

        force[link_index(0,t,x)] += -2.0*std::real(eta_here.dot(vt));
        force[link_index(0,tm,x)] += -2.0*std::real(eta_here.dot(vtb));
        force[link_index(1,t,x)] += -2.0*std::real(eta_here.dot(vx));
        force[link_index(1,t,xm)] += -2.0*std::real(eta_here.dot(vxb));
    }

    return action;
}

struct HMCResult
{
    Gauge theta;
    bool accepted;
    double dH;
    double acceptance_probability;
};

inline HMCResult hmc_step(const Gauge& old_theta)
{
    Gauge momentum(ndim*Nt*Nx);
    for(double& p:momentum) p=gaussian(rng);

    SparseC D0=build_wilson_matrix(old_theta);
    VectorC chi(Ntot);
    for(int i=0;i<Ntot;++i)
        chi[i]=Complex(gaussian(rng),gaussian(rng))/std::sqrt(2.0);
    VectorC phi=D0*chi;

    Gauge Fg,Fpf,total_force(momentum.size());
    double old_Sg=gauge_action_force(old_theta,Fg);
    double old_Spf=pseudofermion_action_force(old_theta,phi,Fpf);
    double old_K=0.5*std::inner_product(momentum.begin(),momentum.end(),
                                        momentum.begin(),0.0);

    Gauge theta=old_theta;
    for(size_t i=0;i<total_force.size();++i)
        total_force[i]=Fg[i]+Fpf[i];

    for(size_t i=0;i<momentum.size();++i)
        momentum[i]-=0.5*hmc_epsilon*total_force[i];

    for(int step=0;step<hmc_steps;++step)
    {
        for(size_t i=0;i<theta.size();++i)
            theta[i]=wrap_angle(theta[i]+hmc_epsilon*momentum[i]);

        gauge_action_force(theta,Fg);
        pseudofermion_action_force(theta,phi,Fpf);
        for(size_t i=0;i<total_force.size();++i)
            total_force[i]=Fg[i]+Fpf[i];

        double factor=(step==hmc_steps-1)?0.5:1.0;
        for(size_t i=0;i<momentum.size();++i)
            momentum[i]-=factor*hmc_epsilon*total_force[i];
    }

    double new_Sg=gauge_action_force(theta,Fg);
    double new_Spf=pseudofermion_action_force(theta,phi,Fpf);
    double new_K=0.5*std::inner_product(momentum.begin(),momentum.end(),
                                        momentum.begin(),0.0);

    double dH=(new_K+new_Sg+new_Spf)-(old_K+old_Sg+old_Spf);
    double pacc=std::min(1.0,std::exp(-dH));
    bool accepted=uniform01(rng)<pacc;
    return {accepted?theta:old_theta,accepted,dH,pacc};
}
