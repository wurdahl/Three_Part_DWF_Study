#pragma once

#include "../../common/config.hpp"
#include <string>

inline const int Nt = runtime_config().integer("wilson.Nt");
inline const int Nx = runtime_config().integer("wilson.Nx");
inline constexpr int Ns = 2;
inline constexpr int ndim = 2;
inline const int Ntot = Nt * Nx * Ns;

inline const double beta = runtime_config().real("wilson.beta");
inline const double m0 = runtime_config().real("wilson.m0");
inline const int trajectories = runtime_config().integer("wilson.trajectories");
inline const int thermalization_cut = runtime_config().integer("wilson.thermalization_cut");
inline const int thin = runtime_config().integer("wilson.thin");
inline const int hmc_steps = runtime_config().integer("wilson.hmc_steps");
inline const double trajectory_length = runtime_config().real("wilson.trajectory_length");
inline const double hmc_epsilon = trajectory_length / hmc_steps;
inline const int source_t = runtime_config().integer("wilson.source_t");
inline const int source_x = runtime_config().integer("wilson.source_x");
inline const double time_boundary_sign = runtime_config().real("wilson.time_boundary_sign");
inline const double space_boundary_sign = runtime_config().real("wilson.space_boundary_sign");
inline const double force_cg_tolerance = runtime_config().real("wilson.force_cg_tolerance");
inline const int force_cg_max_iterations =
    runtime_config().integer("wilson.force_cg_max_iterations");
inline const double measurement_cg_tolerance =
    runtime_config().real("wilson.measurement_cg_tolerance");
inline const int measurement_cg_max_iterations =
    runtime_config().integer("wilson.measurement_cg_max_iterations");
inline const int plateau_window_size = runtime_config().integer("analysis.plateau_window_size");
inline const double plateau_flatness_tolerance =
    runtime_config().real("analysis.plateau_flatness_tolerance");
inline const int earliest_plateau_time =
    runtime_config().integer("analysis.earliest_plateau_time");
inline const int analysis_threads = runtime_config().integer("analysis.threads");
inline const unsigned long long random_seed =
    runtime_config().unsigned_integer("wilson.random_seed");

inline constexpr double pi = 3.14159265358979323846;
inline const std::string ensemble_filename =
    "output/wilson/wilson_ensemble.bin";
inline const std::string history_filename =
    "output/wilson/hmc_history.csv";
