#include "config_io.hpp"
#include "correlator.hpp"
#include "statistics.hpp"
#include "svg_plot.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>

int main()
{
    try
    {
        std::vector<Gauge> configs=load_ensemble(ensemble_filename);
        if(configs.size()<2)
            throw std::runtime_error("Need at least two configurations");

        if(analysis_threads>0)
            omp_set_num_threads(analysis_threads);

        std::vector<Corr> all(configs.size());
#pragma omp parallel for schedule(dynamic)
        for(int i=0;i<static_cast<int>(configs.size());++i)
        {
#pragma omp critical(progress_output)
            std::cout<<"Measuring configuration "<<i+1<<"/"
                     <<configs.size()<<"\n";
            all[i]=pseudoscalar_correlator(configs[i]);
        }

        Corr average=mean_correlator(all);
        Corr error(Nt,0.0);
        for(int t=0;t<Nt;++t)
        {
            double s=0.0;
            for(const Corr& C:all)
                s+=(C[t]-average[t])*(C[t]-average[t]);
            error[t]=std::sqrt(s/(all.size()-1.0)/all.size());
        }

        std::vector<Corr> jk_meff;
        for(int omit=0;omit<(int)all.size();++omit)
            jk_meff.push_back(cosh_effective_mass(
                mean_correlator(all,omit)));

        Corr meff(Nt,std::numeric_limits<double>::quiet_NaN());
        Corr meff_error(Nt,std::numeric_limits<double>::quiet_NaN());

        for(int t=0;t<Nt;++t)
        {
            std::vector<double> samples;
            for(const Corr& m:jk_meff)
                if(std::isfinite(m[t])) samples.push_back(m[t]);
            if(samples.size()!=jk_meff.size()) continue;
            meff[t]=std::accumulate(samples.begin(),samples.end(),0.0)
                    /samples.size();
            meff_error[t]=jackknife_error(samples);
        }

        PlateauWindow plateau=find_flat_plateau(meff,true);
        std::vector<double> jk_masses;
        for(const Corr& m:jk_meff)
            jk_masses.push_back(average_over_window(m,plateau));

        double mass=std::accumulate(jk_masses.begin(),jk_masses.end(),0.0)
                    /jk_masses.size();
        double mass_error=jackknife_error(jk_masses);

        std::cout<<std::setprecision(8)
                 <<"\nSelected plateau: t="<<plateau.first_t
                 <<" through "<<plateau.last_t<<"\n"
                 <<"Wilson pseudoscalar mass = "<<mass
                 <<" +/- "<<mass_error<<"\n";

        write_svg_plot("output/wilson/pseudoscalar_correlator.svg",
                       "Dynamical Wilson pseudoscalar correlator",
                       "C_PP(t)",average,error,true);

        write_svg_plot("output/wilson/cosh_effective_mass.svg",
                       "Dynamical Wilson cosh effective mass",
                       "m_eff(t)",meff,meff_error,false,mass,
                       plateau.first_t,plateau.last_t);

        std::ofstream corr("output/wilson/correlators_by_config.csv");
        corr<<"config";
        for(int t=0;t<Nt;++t) corr<<",t"<<t;
        corr<<"\n";
        for(size_t i=0;i<all.size();++i)
        {
            corr<<i;
            for(double x:all[i]) corr<<","<<x;
            corr<<"\n";
        }

        std::cout<<"Wrote:\n"
                 <<"  output/wilson/pseudoscalar_correlator.svg\n"
                 <<"  output/wilson/cosh_effective_mass.svg\n"
                 <<"  output/wilson/correlators_by_config.csv\n";
    }
    catch(const std::exception& e)
    {
        std::cerr<<"error: "<<e.what()<<"\n";
        return 1;
    }
}
