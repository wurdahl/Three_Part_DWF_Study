#!/usr/bin/env python3
"""Shared readers for the two analyzer CSV formats."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    return sum(values) / len(values)


def load_wilson():
    path = ROOT / "output/wilson/correlators_by_config.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No configurations in {path}")
    keys = [key for key in rows[0] if key.startswith("t")]
    configurations = [[float(row[key]) for key in keys] for row in rows]
    average = [mean([row[t] for row in configurations])
               for t in range(len(keys))]
    return average, configurations


def load_domain_wall(channel="CPP_average"):
    path = ROOT / "output/domain_wall/correlator.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No samples in {path}")
    average = [float(row[channel]) for row in rows]

    config_path = ROOT / "output/domain_wall/channels_by_config.csv"
    if not config_path.exists():
        return average, []
    prefix = "pion_t" if channel == "CPP_average" else "eta_t"
    with config_path.open(newline="") as stream:
        config_rows = list(csv.DictReader(stream))
    keys = [f"{prefix}{t}" for t in range(len(average))]
    configurations = [
        [float(row[key]) for key in keys] for row in config_rows]
    return average, configurations


def load_fit_range():
    values = {}
    for raw in (ROOT / "parameters.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return int(values["analysis.fit_t_min"]), int(values["analysis.fit_t_max"])


def load_parameter(key):
    for raw in (ROOT / "parameters.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            current, value = line.split("=", 1)
            if current.strip() == key:
                return value.strip()
    raise KeyError(key)
