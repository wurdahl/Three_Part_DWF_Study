#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
mkdir -p bin output/domain_wall output/wilson

flags=(-std=c++17 -O3 -flto -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
       -Wall -Wextra -Wpedantic -I/usr/include/eigen3)

# Native instructions improve Eigen-heavy Dirac/CG kernels. Set NATIVE=0 when
# building binaries that will be copied to a different CPU.
if [[ "${NATIVE:-1}" == "1" ]]; then
  flags+=(-march=native -mtune=native)
fi
g++ "${flags[@]}" -Idomain_wall/include domain_wall/generate_configs.cpp -o bin/generate_domain_wall
g++ "${flags[@]}" -Idomain_wall/include domain_wall/analyze_configs.cpp -o bin/analyze_domain_wall
g++ "${flags[@]}" -Iwilson/include wilson/generate_configs.cpp -o bin/generate_wilson
g++ "${flags[@]}" -Iwilson/include wilson/analyze_configs.cpp -o bin/analyze_wilson

echo "Built four executables in bin/"
