#!/usr/bin/env python3
"""Fermion-mass scan at fixed volume (Nx = 40, Nt = 64): mf = 0.1, 0.2,
0.5, to sit alongside the existing mf = 0 point from the Nt = 64 volume
scan. Tests the two-dimensional GMOR analogue m_pi ~ mf^(2/3) against a
nearly mf-independent, anomaly-dominated eta'.

Thinning is set per mass: these ensembles decorrelate far faster than
the gapless mf = 0 chain (binning there showed autocorrelation over
~24-48 trajectories, versus ~3 at mf = 0.2), so 2,000 configurations
cost a fraction of the trajectories. Checkpoints land under
output/studies/part3_mf_Nt64/."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from run_three_part_study import (
    STUDIES, read_case_results, run_case, write_summary)

DESTINATION = STUDIES / "part3_mf_Nt64"
MF0_CASE = STUDIES / "part1_volume_Nt64_mf0/Nx_040"

# mf -> thinning. mf = 0.1 keeps a wider gap as a hedge; verify with a
# binning check once the ensembles exist.
MF_VALUES = {"0.1": 6, "0.2": 4, "0.5": 4}
CONFIGS = 2000


def changes_for(mf, thin):
    return {
        "dwf.Nx": "40",
        "dwf.Nt": "64",
        "dwf.mf": mf,
        "dwf.trajectories": str(60 + thin * CONFIGS),
        "dwf.thermalization_cut": "60",
        "dwf.thin": str(thin),
        "dwf.pseudofermion_maxiter": "3000",
        "dwf.propagator_maxiter": "6000",
        "analysis.max_momentum": "5",
        "analysis.fit_t_max": "20",
        "analysis.bootstrap_block_size": "4",
        "distillation.t_sources": "0,32",
    }


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=len(MF_VALUES)) as pool:
        futures = {
            pool.submit(
                run_case, f"Nt64 Nx=40 mf={mf}", changes_for(mf, thin),
                DESTINATION / f"mf_{float(mf):04.2f}"): float(mf)
            for mf, thin in MF_VALUES.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    # The mf = 0 point already exists from the Nt = 64 volume scan.
    if (MF0_CASE / "COMPLETE").exists():
        results[0.0] = read_case_results(MF0_CASE)

    write_summary(DESTINATION / "summary.csv", "mf",
                  sorted(results.items()))


if __name__ == "__main__":
    main()
