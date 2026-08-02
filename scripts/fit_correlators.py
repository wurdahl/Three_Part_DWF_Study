#!/usr/bin/env python3
"""Windowed exponential fits to correlators, with excited-state terms.

Fits periodic one- and two-state models to the folded correlator on a
time window [t_min, t_max]:

    cosh1: C(t) = A0 [e^{-E0 t} + e^{-E0 (Nt-t)}]
    cosh2: C(t) = cosh1 + A1 [e^{-E1 t} + e^{-E1 (Nt-t)}],  E1 = E0 + dE

The two-state model absorbs excited-state contamination so the window can
start earlier than an effective-mass plateau. dE > 0 is enforced through
the fit bounds, keeping E0 the lower state. Central values come from the
ensemble-mean correlator; errors from refitting circular moving-block
bootstrap resamples (same scheme as estimate_mass.py). A t_min stability
scan of the central fit is reported alongside the primary window.

Sources: the standard analyzer channels (pion, eta) and, when present,
every GEVP principal correlator from distillation_contract.py.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from correlator_io import ROOT, load_domain_wall, load_parameter


def fold(correlator):
    """Average t with Nt-t; the periodic correlator is time-symmetric."""
    n_t = correlator.shape[-1]
    half = n_t // 2
    folded = np.empty(correlator.shape[:-1] + (half + 1,))
    for t in range(half + 1):
        folded[..., t] = 0.5 * (correlator[..., t]
                                + correlator[..., (n_t - t) % n_t])
    return folded


def two_sided(t, amplitude, energy, n_t):
    return amplitude * (np.exp(-energy * t)
                        + np.exp(-energy * (n_t - t)))


def make_models(n_t):
    def cosh1(t, a0, e0):
        return two_sided(t, a0, e0, n_t)

    def cosh2(t, a0, e0, a1, de):
        return two_sided(t, a0, e0, n_t) + two_sided(t, a1, e0 + de, n_t)

    return {"cosh1": (cosh1, 2), "cosh2": (cosh2, 4)}


def block_resample_means(configurations, resamples, block_size, seed):
    """Circular moving-block bootstrap means, one row per replica."""
    rng = np.random.default_rng(seed)
    count = len(configurations)
    blocks = math.ceil(count / block_size)
    starts = rng.integers(0, count, size=(resamples, blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % count
    indices = indices.reshape(resamples, -1)[:, :count]
    return configurations[indices].mean(axis=1)


def initial_guess(model_name, folded, t_fit, n_t):
    finite = np.where(folded > 0, folded, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = finite[:-1] / finite[1:]
    slopes = np.log(ratio[np.isfinite(ratio) & (ratio > 0)])
    e0 = float(np.median(slopes)) if slopes.size else 0.5
    e0 = min(max(e0, 1e-3), 5.0)
    t_anchor = t_fit[len(t_fit) // 2]
    denom = math.exp(-e0 * t_anchor) + math.exp(-e0 * (n_t - t_anchor))
    a0 = folded[t_anchor] / denom if denom else 1.0
    if model_name == "cosh1":
        return [a0, e0]
    return [a0, e0, 0.5 * abs(a0), 1.0]


ENERGY_CEILING = 10.0  # lattice units; larger fits are pure noise


def fit_window(model, p0, t_fit, values, sigma):
    lower = [-np.inf, 1e-6] + ([-np.inf, 1e-6] if len(p0) == 4 else [])
    upper = ([np.inf, ENERGY_CEILING]
             + ([np.inf, ENERGY_CEILING] if len(p0) == 4 else []))
    popt, pcov = curve_fit(
        model, t_fit, values, p0=p0, sigma=sigma, absolute_sigma=True,
        bounds=(lower, upper), maxfev=20000)
    residual = (values - model(t_fit, *popt)) / sigma
    dof = max(len(t_fit) - len(popt), 1)
    chi2 = float(residual @ residual) / dof
    return popt, pcov, chi2


def bootstrap_energies(model, p0, t_fit, replica_means, sigma):
    energies = []
    for replica in replica_means:
        try:
            popt, _, _ = fit_window(
                model, list(p0), t_fit, replica[t_fit], sigma)
            energies.append(popt[1])
        except (RuntimeError, ValueError):
            continue
    if len(energies) < max(2, len(replica_means) // 2):
        return math.nan
    energies = np.asarray(energies)
    return float(energies.std(ddof=1))


def gather_sources():
    sources = {}
    try:
        _, pion = load_domain_wall()
        _, eta = load_domain_wall("eta_singlet")
        if pion:
            sources["pion"] = np.asarray(pion)
        if eta:
            sources["eta"] = np.asarray(eta)
    except FileNotFoundError:
        pass

    principal = ROOT / "output/domain_wall/distillation_principal_by_config.csv"
    if principal.exists():
        with principal.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = [key for key in rows[0] if key != "config"]
        names = sorted({key.rsplit("_t", 1)[0] for key in keys})
        for name in names:
            n_t = sum(1 for key in keys if key.startswith(name + "_t"))
            sources[f"dist_{name}"] = np.asarray(
                [[float(row[f"{name}_t{t}"]) for t in range(n_t)]
                 for row in rows])
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-min", type=int, default=2,
                        help="primary window start (default 2)")
    parser.add_argument("--t-max", type=int, default=None,
                        help="window end (default Nt/2)")
    parser.add_argument("--resamples", type=int, default=500,
                        help="bootstrap refits per source/model")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="subset of sources to fit")
    args = parser.parse_args()

    block_size = int(load_parameter("analysis.bootstrap_block_size"))
    seed = int(load_parameter("analysis.bootstrap_seed"))

    sources = gather_sources()
    if args.sources:
        sources = {name: sources[name] for name in args.sources}
    if not sources:
        raise FileNotFoundError("No correlator sources were found")

    output_rows = []
    for name, configurations in sources.items():
        n_t = configurations.shape[1]
        folded_configs = fold(configurations)
        folded_mean = folded_configs.mean(axis=0)
        replica_means = block_resample_means(
            folded_configs, args.resamples, block_size, seed)
        folded_sigma = replica_means.std(axis=0, ddof=1)

        # Fit only while the signal is resolved: past the first timeslice
        # where the mean is consistent with zero, the correlator is noise.
        resolved = np.abs(folded_mean) > folded_sigma
        signal_end = n_t // 2
        for t in range(1, n_t // 2 + 1):
            if not resolved[t]:
                signal_end = t - 1
                break
        t_max = args.t_max if args.t_max is not None else signal_end
        models = make_models(n_t)

        print(f"\n{name}: {configurations.shape[0]} configs, "
              f"signal resolved to t={signal_end}, t_max={t_max}")
        if t_max < 6:
            print("  skipped: window too short for a stable fit")
            continue
        for model_name, (model, n_params) in models.items():
            scan_energies = {}
            for t_min in range(1, t_max - n_params):
                t_fit = np.arange(t_min, t_max + 1)
                sigma = np.where(
                    folded_sigma[t_fit] > 0, folded_sigma[t_fit], np.inf)
                try:
                    p0 = initial_guess(
                        model_name, folded_mean, t_fit, n_t)
                    popt, pcov, chi2 = fit_window(
                        model, p0, t_fit, folded_mean[t_fit], sigma)
                except (RuntimeError, ValueError):
                    continue
                scan_energies[t_min] = (popt, pcov, chi2)

            primary_t_min = (args.t_min if model_name == "cosh2"
                             else int(load_parameter("analysis.fit_t_min")))
            for t_min, (popt, pcov, chi2) in sorted(scan_energies.items()):
                kind = "primary" if t_min == primary_t_min else "scan"
                e0_error = math.sqrt(max(pcov[1, 1], 0.0))
                e1 = popt[1] + popt[3] if len(popt) == 4 else math.nan
                if kind == "primary":
                    t_fit = np.arange(t_min, t_max + 1)
                    sigma = np.where(
                        folded_sigma[t_fit] > 0,
                        folded_sigma[t_fit], np.inf)
                    e0_error = bootstrap_energies(
                        model, popt, t_fit, replica_means, sigma)
                    print(f"  {model_name} [{t_min},{t_max}]: "
                          f"E0 = {popt[1]:.6f} +/- {e0_error:.6f}"
                          + (f", E1 = {e1:.4f}" if len(popt) == 4 else "")
                          + f", chi2/dof = {chi2:.2f}")
                output_rows.append([
                    name, model_name, kind, t_min, t_max,
                    popt[1], e0_error, popt[0],
                    e1, popt[3] if len(popt) == 4 else math.nan,
                    popt[2] if len(popt) == 4 else math.nan, chi2])

    path = ROOT / "output/domain_wall/correlator_fits.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "source", "model", "kind", "t_min", "t_max",
            "E0", "E0_error", "A0", "E1", "dE", "A1", "chi2_dof"])
        writer.writerows(output_rows)
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
