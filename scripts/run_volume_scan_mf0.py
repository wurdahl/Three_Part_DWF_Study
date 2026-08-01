#!/usr/bin/env python3
"""Checkpointed four-volume scan at mf = 0, mirroring part 1 of the
three-part study. Results land in output/studies/part1_volume_1000_mf0/,
separate from the mf = 0.2 checkpoints."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from run_three_part_study import STUDIES, run_case, write_summary

NX_VALUES = [4, 8, 16, 32]
DESTINATION = STUDIES / "part1_volume_1000_mf0"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                run_case, f"mf0 Nx={nx}",
                {"dwf.Nx": str(nx), "dwf.trajectories": "3012",
                 "dwf.mf": "0.0",
                 "analysis.max_momentum": str(min(5, nx // 2))},
                DESTINATION / f"Nx_{nx:03d}"): nx
            for nx in NX_VALUES
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    write_summary(DESTINATION / "summary.csv", "Nx",
                  sorted(results.items()))


if __name__ == "__main__":
    main()
