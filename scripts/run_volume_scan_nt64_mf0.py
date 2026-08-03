#!/usr/bin/env python3
"""Six-volume mf = 0 scan at Nt = 64: Nx = 4, 8, 16, 32, 40, 64.

Nt = 64 keeps the temporal extent ahead of the shrinking 1/L pion gap
(at Nx = 64 the expected m_pi ~ 0.067 barely decays across Nt = 32).
Compared with the Nt = 32 scan, thinning rises from 3 to 12 (binning
analysis showed autocorrelation reaching ~24-48 trajectories near the
gapless point) and each volume retains 2,000 configurations, so the
per-point errors improve despite the stronger decorrelation. Solver
iteration caps are raised for the light modes at the largest volume.
Checkpoints land under output/studies/part1_volume_Nt64_mf0/."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from run_three_part_study import STUDIES, run_case, write_summary

NX_VALUES = [4, 8, 16, 32, 40, 64]
DESTINATION = STUDIES / "part1_volume_Nt64_mf0"


def changes_for(nx):
    return {
        "dwf.Nx": str(nx),
        "dwf.Nt": "64",
        "dwf.mf": "0.0",
        # 60 + 12 * 2000: 2,000 retained configs, 12 trajectories apart.
        "dwf.trajectories": "24060",
        "dwf.thermalization_cut": "60",
        "dwf.thin": "12",
        "dwf.pseudofermion_maxiter": "3000",
        "dwf.propagator_maxiter": "6000",
        "analysis.max_momentum": str(min(5, nx // 2)),
        "analysis.fit_t_max": "20",
        "analysis.bootstrap_block_size": "4",
        "distillation.t_sources": "0,32",
    }


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=len(NX_VALUES)) as pool:
        futures = {
            pool.submit(
                run_case, f"Nt64 mf0 Nx={nx}", changes_for(nx),
                DESTINATION / f"Nx_{nx:03d}"): nx
            for nx in NX_VALUES
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    write_summary(DESTINATION / "summary.csv", "Nx",
                  sorted(results.items()))


if __name__ == "__main__":
    main()
