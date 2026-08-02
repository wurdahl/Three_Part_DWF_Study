#!/usr/bin/env python3
"""Four-volume scans (Nx = 4, 8, 16, 32) at mf = 0.2 and mf = 0, measured
with the excited-state-aware estimators: two-state (cosh2) wall-source
fits and the distillation GEVP, 1000 configurations per volume.
Checkpoints land under output/studies/part1_volume_cosh2*/."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from run_three_part_study import STUDIES, run_case, write_summary

NX_VALUES = [4, 8, 16, 32]
SCANS = {
    "part1_volume_cosh2": "0.2",
    "part1_volume_cosh2_mf0": "0.0",
}


def main():
    tasks = []
    for study, mf in SCANS.items():
        (STUDIES / study).mkdir(parents=True, exist_ok=True)
        for nx in NX_VALUES:
            tasks.append((
                study, nx, f"{study} Nx={nx}",
                {"dwf.Nx": str(nx), "dwf.trajectories": "3012",
                 "dwf.mf": mf,
                 "analysis.max_momentum": str(min(5, nx // 2))},
                STUDIES / study / f"Nx_{nx:03d}"))

    results = {study: {} for study in SCANS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(run_case, label, changes, destination):
                (study, nx)
            for study, nx, label, changes, destination in tasks
        }
        for future in as_completed(futures):
            study, nx = futures[future]
            results[study][nx] = future.result()

    for study in SCANS:
        write_summary(STUDIES / study / "summary.csv", "Nx",
                      sorted(results[study].items()))


if __name__ == "__main__":
    main()
