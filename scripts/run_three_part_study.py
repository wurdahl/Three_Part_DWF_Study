#!/usr/bin/env python3
"""Checkpointed parallel three-part dynamical domain-wall study."""

import csv
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "parameters.txt"
STUDIES = ROOT / "output/studies"

NX_VALUES = [32, 56, 80, 96]
GPU_LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()


def mps_active():
    """Concurrent GPU processes share cleanly only under the MPS daemon."""
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    return (Path(pipe_dir) / "control").exists()
MF_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7,
             0.9, 1.1, 1.3, 1.5, 1.75, 2.0]

BASE = {
    "dwf.Nt": "32",
    "dwf.N5": "12",
    "dwf.beta": "3.0",
    "dwf.mf": "0.2",
    "dwf.trajectories": "48",
    "dwf.thermalization_cut": "12",
    "dwf.thin": "3",
    "dwf.leapfrog_steps": "40",
    "eta.noise_vectors": "8",
    "analysis.threads": "4",
    "analysis.fit_t_min": "3",
    "analysis.fit_t_max": "11",
    "analysis.gevp_t0": "2",
    "analysis.max_momentum": "5",
    "analysis.bootstrap_resamples": "1000",
    "analysis.bootstrap_block_size": "2",
}


def rendered_parameters(changes):
    pending = dict(changes)
    output = []
    for raw in PARAMETERS.read_text().splitlines():
        line = raw.split("#", 1)[0]
        if "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in pending:
                comment = (" #" + raw.split("#", 1)[1]
                           if "#" in raw else "")
                output.append(f"{key} = {pending.pop(key)}{comment}")
                continue
        output.append(raw)
    if pending:
        raise KeyError(f"Unknown parameters: {sorted(pending)}")
    return "\n".join(output) + "\n"


def execute(command, cwd, log):
    with log.open("a") as stream:
        stream.write(f"\n$ {' '.join(map(str, command))}\n")
        stream.flush()
        subprocess.run(
            [str(item) for item in command], cwd=cwd, stdout=stream,
            stderr=subprocess.STDOUT, check=True)


def read_masses(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        result[row["model"]] = (
            float(row["mass"]),
            float(row.get(
                "bootstrap_standard_error",
                row.get("jackknife_error", "nan"))))
    return result


def archive(work, destination):
    source_dir = work / "output/domain_wall"
    names = [
        "correlator.csv", "channels_by_config.csv",
        "momentum_correlators.csv", "gevp_spectrum.csv",
        "pion_eta_correlators.svg", "residual_mass.svg",
        "gauge_history.svg", "run_info_history.csv",
    ]
    for name in names:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    shutil.copy2(
        work / "output/mass_estimates.csv",
        destination / "mass_estimates.csv")
    shutil.copy2(work / "parameters.txt", destination / "parameters.txt")


def run_case(label, changes, destination):
    if (destination / "COMPLETE").exists():
        print(f"SKIP {label}: checkpoint exists", flush=True)
        return read_masses(destination / "mass_estimates.csv")

    settings = dict(BASE)
    settings.update(changes)
    # With GPU generation the CPU is free during analysis, so each
    # analyzer takes every core (threads = 0 is the OpenMP default) and
    # runs one case at a time instead of four 4-thread analyzers.
    gpu_generation = os.environ.get("DWF_USE_GPU") == "1"
    if gpu_generation and "analysis.threads" not in changes:
        settings["analysis.threads"] = "0"
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    (work / "parameters.txt").write_text(rendered_parameters(settings))
    shutil.copytree(
        ROOT / "scripts", work / "scripts",
        ignore=shutil.ignore_patterns("__pycache__"))
    (work / "output/domain_wall").mkdir(parents=True)
    log = destination / "run.log"
    log.write_text("")

    print(f"START {label}", flush=True)
    use_gpu = os.environ.get("DWF_USE_GPU") == "1"
    generator = (
        "bin/generate_domain_wall_gpu" if use_gpu
        else "bin/generate_domain_wall")
    try:
        if use_gpu and not mps_active():
            # Without MPS, concurrent processes time-slice the GPU with
            # heavy context switching, so generation runs one case at a
            # time; the CPU analyzers still overlap freely.
            with GPU_LOCK:
                execute([ROOT / generator], work, log)
        else:
            execute([ROOT / generator], work, log)
        if use_gpu:
            with ANALYSIS_LOCK:
                execute([ROOT / "bin/analyze_domain_wall"], work, log)
        else:
            execute([ROOT / "bin/analyze_domain_wall"], work, log)
        execute(["python3", "scripts/estimate_mass.py"], work, log)
        execute(["python3", "scripts/gevp_spectrum.py"], work, log)
        archive(work, destination)
        (destination / "COMPLETE").write_text("complete\n")
        masses = read_masses(destination / "mass_estimates.csv")
        print(
            f"DONE {label}: pion={masses.get('domain_wall_pion')} "
            f"eta={masses.get('domain_wall_eta')}", flush=True)
        return masses
    finally:
        if (destination / "COMPLETE").exists():
            shutil.rmtree(work)


def write_summary(path, independent_name, records):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            independent_name, "pion_mass", "pion_error",
            "eta_mass", "eta_error"])
        for value, masses in records:
            pion = masses.get("domain_wall_pion", (float("nan"),) * 2)
            eta = masses.get("domain_wall_eta", (float("nan"),) * 2)
            writer.writerow([value, pion[0], pion[1], eta[0], eta[1]])


def main():
    STUDIES.mkdir(parents=True, exist_ok=True)
    tasks = []
    for nx in NX_VALUES:
        tasks.append((
            "volume", nx, f"part1 Nx={nx}",
            # The analyzer requires max_momentum <= Nx/2.
            {"dwf.Nx": str(nx), "dwf.trajectories": "3012",
             "analysis.max_momentum": str(min(5, nx // 2))},
            STUDIES / "part1_volume_1000" / f"Nx_{nx:03d}"))
    tasks.append((
        "dispersion", 6.0, "part2 beta=6 dispersion",
        {"dwf.Nx": "64", "dwf.Nt": "40", "dwf.beta": "6.0",
         "dwf.trajectories": "60", "dwf.thermalization_cut": "12",
         "dwf.thin": "4"},
        STUDIES / "part2_dispersion/beta_6"))
    for mf in MF_VALUES:
        tasks.append((
            "mf", mf, f"part3 mf={mf}",
            {"dwf.Nx": "48", "dwf.mf": str(mf)},
            STUDIES / "part3_mf" / f"mf_{mf:04.2f}"))

    results = {"volume": {}, "dispersion": {}, "mf": {}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(run_case, label, changes, destination):
                (group, value)
            for group, value, label, changes, destination in tasks
        }
        for future in as_completed(futures):
            group, value = futures[future]
            results[group][value] = future.result()

    write_summary(
        STUDIES / "part1_volume_1000/summary.csv", "Nx",
        sorted(results["volume"].items()))
    write_summary(
        STUDIES / "part3_mf/summary.csv", "mf",
        sorted(results["mf"].items()))


if __name__ == "__main__":
    main()
