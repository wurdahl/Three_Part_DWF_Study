#pragma once

#include "cg_solver.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "wilson_operator.hpp"

#include <iostream>

inline Corr pseudoscalar_correlator(const Gauge& theta)
{
    SparseC D=build_wilson_matrix(theta);
    Eigen::MatrixXcd propagator=Eigen::MatrixXcd::Zero(Ntot,Ns);

    for(int source_spin=0;source_spin<Ns;++source_spin)
    {
        VectorC source=VectorC::Zero(Ntot);
        source[site_index(source_t,source_x,source_spin)]=1.0;

        SolveResult result=solve_CGNR(
            D,source,measurement_cg_tolerance,
            measurement_cg_max_iterations);

        std::cout<<"    spin="<<source_spin
                 <<" CG iterations="<<result.iterations
                 <<" normal residual="<<result.normal_residual
                 <<" D residual="<<result.dirac_residual<<"\n";

        propagator.col(source_spin)=result.x;
    }

    Corr C(Nt,0.0);
    for(int t=0;t<Nt;++t)
        for(int x=0;x<Nx;++x)
            for(int sink_spin=0;sink_spin<Ns;++sink_spin)
                for(int source_spin=0;source_spin<Ns;++source_spin)
                    C[t]+=std::norm(propagator(
                        site_index(t,x,sink_spin),source_spin));
    return C;
}
