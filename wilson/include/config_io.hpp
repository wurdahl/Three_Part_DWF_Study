#pragma once

#include "parameters.hpp"
#include "types.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct EnsembleHeader
{
    char magic[8]={'W','I','L','C','F','G','1','\0'};
    std::int32_t nt=Nt,nx=Nx;
    double b=beta,m=m0;
    std::uint64_t count=0;
};

inline void save_ensemble(const std::string& filename,
                          const std::vector<Gauge>& configs)
{
    std::filesystem::create_directories(
        std::filesystem::path(filename).parent_path());
    std::ofstream out(filename,std::ios::binary);
    if(!out) throw std::runtime_error("Cannot write "+filename);
    EnsembleHeader h; h.count=configs.size();
    out.write(reinterpret_cast<const char*>(&h),sizeof(h));
    for(const Gauge& cfg:configs)
        out.write(reinterpret_cast<const char*>(cfg.data()),
                  cfg.size()*sizeof(double));
}

inline std::vector<Gauge> load_ensemble(const std::string& filename)
{
    std::ifstream in(filename,std::ios::binary);
    if(!in) throw std::runtime_error("Cannot open "+filename);
    EnsembleHeader h;
    in.read(reinterpret_cast<char*>(&h),sizeof(h));
    if(std::string(h.magic)!="WILCFG1")
        throw std::runtime_error("Wrong ensemble file format");
    if(h.nt!=Nt||h.nx!=Nx)
        throw std::runtime_error("Ensemble dimensions do not match");
    std::vector<Gauge> configs(h.count,Gauge(ndim*Nt*Nx));
    for(Gauge& cfg:configs)
        in.read(reinterpret_cast<char*>(cfg.data()),
                cfg.size()*sizeof(double));
    if(!in) throw std::runtime_error("Truncated ensemble file");
    return configs;
}
