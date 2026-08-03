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
g++ "${flags[@]}" -Idomain_wall/include domain_wall/build_perambulators.cpp -o bin/build_perambulators
g++ "${flags[@]}" -Iwilson/include wilson/generate_configs.cpp -o bin/generate_wilson
g++ "${flags[@]}" -Iwilson/include wilson/analyze_configs.cpp -o bin/analyze_wilson

echo "Built five executables in bin/"

# Optional CUDA generator. nvcc 12.x cannot emit Blackwell SASS, so embed
# compute_90 PTX and let the driver JIT it for the installed GPU.
if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 \
       -Idomain_wall/include \
       -c domain_wall/gpu_hmc.cu -o bin/gpu_hmc.o
  g++ "${flags[@]}" -Idomain_wall/include \
      domain_wall/generate_configs_gpu.cpp bin/gpu_hmc.o \
      -o bin/generate_domain_wall_gpu -lcudart
  echo "Built bin/generate_domain_wall_gpu"
else
  echo "nvcc not found; skipped bin/generate_domain_wall_gpu"
fi
