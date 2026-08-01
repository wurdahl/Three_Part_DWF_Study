#pragma once

#include "../../common/config.hpp"
#include <string>

// These are runtime constants loaded once at program startup.
inline const int Nt = runtime_config().integer("dwf.Nt");
inline const int Nx = runtime_config().integer("dwf.Nx");
inline const int N5 = runtime_config().integer("dwf.N5");
inline constexpr int Ns = 2;
inline const int Ndof = N5 * Nt * Nx * Ns;
inline const int Ngauge = 2 * Nt * Nx;

inline const double beta = runtime_config().real("dwf.beta");
inline const double M5 = runtime_config().real("dwf.M5");
inline const double mf = runtime_config().real("dwf.mf");
inline const double epsilon = runtime_config().real("dwf.epsilon");
inline const int leapfrog_steps = runtime_config().integer("dwf.leapfrog_steps");
// OpenMP threads for the Dirac-operator site loops during HMC. Older
// parameter files without the key keep the previous single-threaded HMC.
inline const int hmc_threads = runtime_config().integer_or("dwf.hmc_threads", 1);
inline const int trajectories = runtime_config().integer("dwf.trajectories");
inline const int thermalization_cut = runtime_config().integer("dwf.thermalization_cut");
inline const int thin = runtime_config().integer("dwf.thin");
inline const double pseudofermion_rtol = runtime_config().real("dwf.pseudofermion_rtol");
inline const int pseudofermion_maxiter = runtime_config().integer("dwf.pseudofermion_maxiter");
inline const double propagator_rtol = runtime_config().real("dwf.propagator_rtol");
inline const int propagator_maxiter = runtime_config().integer("dwf.propagator_maxiter");
inline const int analysis_threads = runtime_config().integer("analysis.threads");
inline const int eta_noise_vectors = runtime_config().integer("eta.noise_vectors");
inline const int eta_flavors = runtime_config().integer("eta.flavors");
inline const unsigned long long eta_random_seed =
    runtime_config().unsigned_integer("eta.random_seed");
inline const int max_momentum =
    runtime_config().integer("analysis.max_momentum");
inline const unsigned long long random_seed =
    runtime_config().unsigned_integer("dwf.random_seed");

inline constexpr double pi = 3.141592653589793238462643383279502884;
inline const std::string configs_filename = "output/domain_wall/configs.npy";
inline const std::string analysis_configs_filename =
    "output/domain_wall/analysis_configs.npy";
inline const std::string history_filename =
    "output/domain_wall/run_info_history.csv";
