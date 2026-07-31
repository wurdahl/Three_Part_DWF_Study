#!/usr/bin/env python3
"""Estimate both masses with the periodic cosh effective-mass relation."""

import csv
import math
import random

from correlator_io import (
    ROOT, load_domain_wall, load_fit_range, load_parameter, load_wilson)


def effective_mass(correlator):
    result = [math.nan] * len(correlator)
    for t in range(1, len(correlator) - 1):
        denominator = 2.0 * correlator[t]
        ratio = ((correlator[t - 1] + correlator[t + 1]) / denominator
                 if denominator else math.nan)
        if math.isfinite(ratio) and ratio >= 1.0:
            result[t] = math.acosh(ratio)
    return result


def window_average(values, first, last):
    selected = [values[t] for t in range(first, last + 1)
                if t < len(values) and math.isfinite(values[t])]
    if not selected:
        raise ValueError(f"No finite effective masses in t={first}..{last}")
    return sum(selected) / len(selected)


def percentile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return (sorted_values[lower] * (1.0 - fraction)
            + sorted_values[upper] * fraction)


def bootstrap_mass(
        configurations, first, last, resamples, block_size, seed):
    if len(configurations) < 2:
        return math.nan, math.nan, math.nan, 0
    if resamples < 2:
        raise ValueError("analysis.bootstrap_resamples must be at least 2")
    if not 1 <= block_size <= len(configurations):
        raise ValueError(
            "analysis.bootstrap_block_size must be between 1 and "
            "the number of configurations")

    rng = random.Random(seed)
    count = len(configurations)
    samples = []
    for _ in range(resamples):
        indices = []
        while len(indices) < count:
            start = rng.randrange(count)
            indices.extend(
                (start + offset) % count for offset in range(block_size))
        indices = indices[:count]
        average = [
            sum(configurations[index][t] for index in indices) / count
            for t in range(len(configurations[0]))]
        try:
            samples.append(
                window_average(effective_mass(average), first, last))
        except ValueError:
            # A noisy nonlinear resample can have no valid acosh points.
            continue

    if len(samples) < max(2, resamples // 2):
        raise ValueError(
            f"Only {len(samples)}/{resamples} bootstrap replicas produced "
            "a valid mass")
    center = sum(samples) / len(samples)
    standard_error = math.sqrt(
        sum((value - center) ** 2 for value in samples)
        / (len(samples) - 1))
    ordered = sorted(samples)
    lower = percentile(ordered, 0.025)
    upper = percentile(ordered, 0.975)
    return standard_error, lower, upper, len(samples)


def main():
    first, last = load_fit_range()
    resamples = int(load_parameter("analysis.bootstrap_resamples"))
    block_size = int(load_parameter("analysis.bootstrap_block_size"))
    seed = int(load_parameter("analysis.bootstrap_seed"))
    output = []
    for model_index, (model, loader) in enumerate((
            ("wilson_pion", load_wilson),
            ("domain_wall_pion", load_domain_wall),
            ("domain_wall_eta", lambda: load_domain_wall("eta_singlet")))):
        try:
            correlator, configurations = loader()
        except FileNotFoundError:
            print(f"{model}: skipped (analyzer output not found)")
            continue
        masses = effective_mass(correlator)
        try:
            estimate = window_average(masses, first, last)
            error, ci_low, ci_high, valid = bootstrap_mass(
                configurations, first, last, resamples, block_size,
                seed + model_index * 1000003)
        except ValueError as problem:
            print(f"{model}: skipped ({problem})")
            continue
        output.append((
            model, estimate, error, ci_low, ci_high,
            first, last, valid, block_size))
        uncertainty = f" +/- {error:.8g}" if math.isfinite(error) else ""
        print(f"{model}: mass = {estimate:.8g}{uncertainty} "
              f"(95% bootstrap CI [{ci_low:.8g}, {ci_high:.8g}], "
              f"cosh plateau t={first}..{last})")

    if not output:
        raise FileNotFoundError("No analyzer correlator outputs were found")

    path = ROOT / "output/mass_estimates.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "model", "mass", "bootstrap_standard_error",
            "bootstrap_ci_2.5_percent", "bootstrap_ci_97.5_percent",
            "fit_t_min", "fit_t_max", "valid_bootstrap_resamples",
            "bootstrap_block_size"])
        writer.writerows(output)
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
