# Dynamical Wilson + Domain-Wall Fermions

This folder combines the two non-quenched U(1) projects:

- Wilson pseudofermion HMC
- Domain-wall pseudofermion HMC

Both generate two degenerate dynamical flavors because their pseudofermion
integral produces `det(D D†) = |det D|²`. Neither generator is quenched.

## Configure and build

Edit [`parameters.txt`](parameters.txt) at any time. The executables read it
when they start, so parameter changes do **not** require recompilation.

From this directory:

```bash
chmod +x compile.sh run.sh
./compile.sh
```

Eigen 3 must be installed at `/usr/include/eigen3`.

The release build uses `-O3`, link-time optimization, OpenMP, Eigen release
mode, and CPU-native vector instructions. To build a portable binary instead:

```bash
NATIVE=0 ./compile.sh
```

The domain-wall HMC parallelizes its Dirac-operator site loops with OpenMP;
`dwf.hmc_threads` in `parameters.txt` sets the thread count (parameter files
without the key keep the previous single-threaded behavior).

### Optional CUDA generator

If `nvcc` is on the PATH, `compile.sh` additionally builds
`bin/generate_domain_wall_gpu`, a fully device-resident GPU implementation of
the domain-wall HMC (fp32 MD force solves, fp64 accept/reject action solves,
CG in CUDA-graph chunks). Without `nvcc` the GPU build is skipped and nothing
else changes — every GPU code path in this repository is opt-in.

Validation switches: `DWF_GPU_FP64_MD=1` runs every solve in fp64, and
building with `-DDWF_CG_CHUNK=1` reproduces the CPU solver's exact stopping
iteration; together they reproduce the CPU Markov chain to machine precision.
With the default settings the GPU chain is a statistically equivalent
realization, not bit-identical to the CPU chain, so avoid mixing CPU- and
GPU-generated trajectories within one ensemble directory.

## Run

```bash
./run.sh generate-wilson
./run.sh analyze-wilson

./run.sh generate-domain-wall
./run.sh analyze-domain-wall
```

Run commands from this folder because `parameters.txt` is intentionally loaded
from the working directory. Outputs are kept separate under `output/wilson/`
and `output/domain_wall/`.

The full default dynamical runs are computationally expensive. For a smoke
test, temporarily use smaller lattices and fewer trajectories in
`parameters.txt`.

## Three-part study driver

Run the checkpointed study with:

```bash
python3 scripts/run_three_part_study.py
```

Part 1 uses four spatial extents, `Nx = 32, 56, 80, 96`. Each point runs
3,012 trajectories, discards the first 12, and retains every third subsequent
configuration, yielding exactly 1,000 measured configurations per volume.
Its checkpoints and results are stored under
`output/studies/part1_volume_1000/`, separate from the archived exploratory
12-point scan. Parts 2 and 3 retain their existing settings. Volume cases cap
`analysis.max_momentum` at `Nx / 2` (the analyzer's Nyquist bound), so small
spatial extents also run.

Setting `DWF_USE_GPU=1` makes the study generate with
`bin/generate_domain_wall_gpu` (requires the optional CUDA build; the default
is the CPU generator). In GPU mode the analyzers take every core and run one
case at a time, and generation is serialized unless the NVIDIA MPS daemon is
running (`nvidia-cuda-mps-control -d`), which lets concurrent chains share
the GPU cleanly.

`scripts/run_volume_scan_mf0.py` runs a companion four-volume scan
(`Nx = 4, 8, 16, 32`) at `mf = 0`, checkpointed under
`output/studies/part1_volume_1000_mf0/`.
`scripts/run_volume_scan_cosh2.py` runs both `mf = 0.2` and `mf = 0`
four-volume scans (`Nx = 4, 8, 16, 32`) through the full measurement chain
— standard analyzer, distillation, and the two-state correlator fits — and
writes summaries whose `pion_mass` column is the excited-state-safe cosh2
fit (plateau and distillation values ride along as extra columns). These
runs also archive `analysis_configs.npy` and the distillation data, so they
can be re-analyzed without regenerating the Markov chains.
`scripts/plot_finite_size.py` fits the resulting pion masses with
`m_inf + A exp(-B L)`, annotates the chi^2/dof, and writes a
`figs/pionMassVsMpiL_*.pdf` summary figure (requires matplotlib and scipy).

Measurement solves (the analyzer's propagators and the distillation
perambulators) use an even-odd preconditioned solver by default: the
spacetime hopping only connects opposite-parity sites and the constant
fifth-dimension block inverts in closed form, so CGNR runs on the
better-conditioned Schur complement over half the lattice. This is
~2-2.5x faster and agrees with the unpreconditioned path to solver
tolerance; set `dwf.even_odd = 0` to fall back for cross-checking.
Lattices with odd Nt or Nx fall back automatically.

The analyzers measure independent gauge configurations concurrently with
OpenMP. Set `analysis.threads = 0` for the runtime default, or choose a fixed
thread count. Domain-wall solves use substantial memory per worker, so reduce
this value if the machine begins swapping. HMC trajectories remain sequential
because each accepted configuration depends on the preceding trajectory.

## Shared analysis

After both C++ analyzers have produced their CSV files:

```bash
./run.sh plot
./run.sh estimate-mass
./run.sh gevp
```

The plot command writes `output/correlators_comparison.svg`. Each correlator is
normalized by its own maximum so the two formulations can share a meaningful
log-scale plot.

The mass command writes `output/mass_estimates.csv`. It computes

```text
m_eff(t) = acosh((C(t-1) + C(t+1)) / (2 C(t)))
```

and averages finite values over `analysis.fit_t_min` through
`analysis.fit_t_max` from `parameters.txt`. Wilson uncertainty is a
a reproducible circular moving-block bootstrap over gauge configurations. The
reported error is the bootstrap standard error, accompanied by a percentile
95% confidence interval. Configure the resample count, block size, and seed
with `analysis.bootstrap_*` in `parameters.txt`. A block size greater than one
helps retain short-range Markov-chain autocorrelation.

The domain-wall C++ analyzer also retains its original residual-mass
measurement `m_res = C_J5 / C_PP`; this is distinct from the pseudoscalar mass
estimated by the shared script.

## Distillation (hadspec-style) spectroscopy

An alternative measurement pipeline in the style of the Hadron Spectrum
Collaboration: quark fields are smeared with the lowest `distillation.n_vectors`
eigenvectors of the gauge-covariant spatial Laplacian on each timeslice, and
the propagator is stored as perambulators in that low-mode basis so that any
meson correlator becomes a small dense trace formed after the fact.

```bash
./run.sh perambulators    # C++: bin/build_perambulators
./run.sh contract         # python: Wick contractions + GEVP
./run.sh fit-correlators  # python: windowed 1- and 2-exponential fits
```

`bin/build_perambulators` reads `analysis_configs.npy` and writes, per
configuration, to `output/domain_wall/distillation/`:

- `perambulators.npy` — tau(t, t0) = V(t)^dag S(t, t0) V(t0) for each source
  time in `distillation.t_sources` (spin x eigenvector at both ends),
- `elementals_momentum.npy` — V(t)^dag e^{ipx} V(t) for each momentum up to
  `analysis.max_momentum`,
- `elementals_derivative.npy` — the symmetric covariant-derivative analogue,
- `laplacian_eigenvalues.npy` and `t_sources.npy`.

The solve count is `n_vectors * 2 spins * len(t_sources)` per configuration
(32 with the defaults, versus 16 for the standard analyzer).

`scripts/distillation_contract.py` then performs the Wick contractions
automatically for every interpolator listed in `distillation_ops.json`
(operators are `qbar Gamma q` with `Gamma` built from `id, g5, gt, gx`
products, a smeared or covariant-derivative spatial structure, and a momentum
index; conventions: `gamma_t = sigma1`, `gamma_x = sigma2`,
`gamma5 = sigma3`). Each channel's operator basis becomes a correlator
matrix, and a fixed-vector GEVP (`analysis.gevp_t0`) yields principal
correlators written both as ensemble means (`distillation_gevp.csv`) and per
configuration (`distillation_principal_by_config.csv`) for bootstrapping.
Only connected contractions are formed; the flavor-singlet disconnected
piece remains with the Z4-noise machinery of the standard analyzer.

## Correlator fits with excited-state terms

`scripts/fit_correlators.py` replaces the effective-mass plateau average
with windowed fits of the folded periodic correlator to

```text
cosh1: C(t) = A0 [e^{-E0 t} + e^{-E0 (Nt-t)}]
cosh2: C(t) = cosh1 + A1 [e^{-E1 t} + e^{-E1 (Nt-t)}],  E1 = E0 + dE > E0
```

The two-state form absorbs excited-state contamination so the window can
start well before any plateau. It fits the wall-source pion and eta plus
every distillation principal correlator, clips each window where the signal
leaves the noise, reports a t_min stability scan, and takes E0 errors from
refitting circular moving-block bootstrap resamples. Results land in
`output/domain_wall/correlator_fits.csv`.

This matters here: on test ensembles the wall-source pion effective mass is
still falling across the whole `fit_t_min..fit_t_max` window, so the plateau
average is biased high, while the cosh2 fit and the distillation GEVP ground
state agree with each other at a distinctly lower mass.

## Pion and eta spectroscopy

The pion is measured in the flavor-nonsinglet connected pseudoscalar channel.
The two-flavor eta is measured in the flavor-singlet channel using

```text
C_eta(t) = C_connected(t) - Nf C_disconnected(t).
```

The disconnected Wick contraction is estimated from complex Z4 volume-noise
sources on the physical domain-wall boundary field. Configure its cost with
`eta.noise_vectors`; its random stream is reproducible through
`eta.random_seed`. The analyzer writes ensemble averages to
`output/domain_wall/correlator.csv` and bootstrap-ready measurements to
`output/domain_wall/channels_by_config.csv`.

Pion and eta are different flavor representations and are therefore not mixed
into one GEVP. `./run.sh gevp` instead performs a separate 2x2 time-shift
(Krylov) GEVP in each channel and writes
`output/domain_wall/gevp_spectrum.csv`. Negative or non-finite excited
principal correlators indicate insufficient statistics or an unstable basis;
they are retained as diagnostics rather than silently converted into masses.

Disconnected diagrams are intrinsically noisy. A credible eta result should
be checked for stability under increased `eta.noise_vectors`, a larger gauge
ensemble, changes of `analysis.fit_t_min/max`, and changes of
`analysis.gevp_t0`.
