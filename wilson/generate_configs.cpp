#include "config_io.hpp"
#include "gauge.hpp"
#include "parameters.hpp"
#include "svg_plot.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    try
    {
        std::filesystem::create_directories("output/wilson");
        std::ofstream history(history_filename);
        history<<"trajectory,accepted,dH,acceptance_probability,plaquette,saved\n";

        Gauge theta(ndim*Nt*Nx,0.0);
        std::vector<Gauge> saved;
        std::vector<double> plaquettes;

        int accepted_count=0;
        for(int trajectory=0;trajectory<trajectories;++trajectory)
        {
            HMCResult result=hmc_step(theta);
            theta=result.theta;
            if(result.accepted) ++accepted_count;

            bool save_now=
                trajectory>=thermalization_cut &&
                (trajectory-thermalization_cut)%thin==0;
            if(save_now) saved.push_back(theta);

            double plaq=average_plaquette(theta);
            plaquettes.push_back(plaq);
            history<<trajectory<<","<<result.accepted<<","
                   <<result.dH<<","<<result.acceptance_probability<<","
                   <<plaq<<","<<saved.size()<<"\n";

            std::cout<<"traj "<<std::setw(4)<<trajectory+1<<"/"
                     <<trajectories<<" accepted="<<result.accepted
                     <<" dH="<<std::scientific<<result.dH
                     <<" Pacc="<<result.acceptance_probability
                     <<" plaq="<<std::fixed<<plaq
                     <<" saved="<<saved.size()<<"\n";
        }

        save_ensemble(ensemble_filename,saved);
        write_gauge_history_svg("output/wilson/gauge_history.svg",
                                plaquettes,thermalization_cut);

        std::cout<<"\nSaved "<<saved.size()<<" configurations to "
                 <<ensemble_filename<<"\n"
                 <<"Acceptance rate = "
                 <<static_cast<double>(accepted_count)/trajectories<<"\n"
                 <<"Wrote output/wilson/gauge_history.svg\n";
    }
    catch(const std::exception& e)
    {
        std::cerr<<"error: "<<e.what()<<"\n";
        return 1;
    }
}
