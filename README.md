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
12-point scan. Parts 2 and 3 retain their existing settings.

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
