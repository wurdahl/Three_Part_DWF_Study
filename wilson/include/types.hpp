#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <vector>

using Complex = std::complex<double>;
using VectorC = Eigen::VectorXcd;
using SparseC = Eigen::SparseMatrix<Complex>;
using TripletC = Eigen::Triplet<Complex>;
using Mat2 = Eigen::Matrix2cd;
using Gauge = std::vector<double>;
using Corr = std::vector<double>;
