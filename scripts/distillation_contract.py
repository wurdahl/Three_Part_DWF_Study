#!/usr/bin/env python3
"""Automatic Wick contractions and GEVP for the distillation measurement.

Reads the perambulators and elementals written by bin/build_perambulators
and a list of interpolators (distillation_ops.json, or built-in defaults),
forms every connected meson correlator C_ij(t) = <O_i(t) O_j(t0)^dagger>
in each channel, and solves a GEVP for the principal correlators.

Conventions match the C++ operator: gamma_t = sigma1, gamma_x = sigma2,
gamma5 = sigma3. Each interpolator is O = qbar Gamma q with a spatial
structure that is either "smeared" (momentum elemental) or "deriv"
(symmetric covariant derivative). The antiquark line uses gamma5
hermiticity of the effective boundary propagator, the same property the
wall-source pion measurement relies on. Only connected contractions are
formed; flavor-singlet disconnected pieces remain with the Z4-noise
machinery in the standard analyzer.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "output/domain_wall/distillation"

GAMMA_BASIS = {
    "id": np.eye(2, dtype=complex),
    "gt": np.array([[0, 1], [1, 0]], dtype=complex),
    "gx": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "g5": np.array([[1, 0], [0, -1]], dtype=complex),
}
G5_DIAG = np.array([1.0, -1.0])

DEFAULT_OPS = {
    "pion": [
        {"name": "g5", "gamma": "g5", "spatial": "smeared", "momentum": 0},
        {"name": "g5gt", "gamma": "g5gt", "spatial": "smeared",
         "momentum": 0},
        {"name": "g5_deriv", "gamma": "g5", "spatial": "deriv",
         "momentum": 0},
    ],
    "vector": [
        {"name": "gx", "gamma": "gx", "spatial": "smeared", "momentum": 0},
        {"name": "id_deriv", "gamma": "id", "spatial": "deriv",
         "momentum": 0},
    ],
}


def gamma_matrix(name):
    """Products like "g5gt" are parsed left to right."""
    matrix = np.eye(2, dtype=complex)
    rest = name
    while rest:
        for token in ("g5", "gt", "gx", "id"):
            if rest.startswith(token):
                matrix = matrix @ GAMMA_BASIS[token]
                rest = rest[len(token):]
                break
        else:
            raise ValueError(f"Unknown gamma name: {name}")
    return matrix


def load_complex(path):
    raw = np.load(path)
    return raw[..., 0] + 1j * raw[..., 1]


def load_operators(path):
    if path is not None:
        text = Path(path).read_text()
    elif (ROOT / "distillation_ops.json").exists():
        text = (ROOT / "distillation_ops.json").read_text()
    else:
        return DEFAULT_OPS
    return json.loads(text)["channels"]


def spatial_elemental(op, momentum, derivative):
    kind = op.get("spatial", "smeared")
    p = int(op.get("momentum", 0))
    if p >= momentum.shape[1]:
        raise ValueError(
            f"operator {op['name']}: momentum {p} not measured "
            f"(analysis.max_momentum too small)")
    if kind == "smeared":
        return momentum[:, p]
    if kind == "deriv":
        return derivative[:, p]
    raise ValueError(f"operator {op['name']}: unknown spatial '{kind}'")


def channel_correlators(ops, tau, t_sources, momentum, derivative):
    """Connected correlator matrix C[config, i, j, dt] for one channel.

    With A = tau(t, t0) and the backward line gamma5 A^dagger gamma5,
    C_ij = Tr[Phi_i(t) A Phi_j(t0)^dagger gamma5 A^dagger gamma5], the
    distillation form of Tr[Gamma_i S(t,t0) Gamma_j^dagger S(t0,t)],
    normalized to match the analyzer's positive C_PP convention.
    """
    n_config, n_source, n_t = tau.shape[:3]
    gammas = [gamma_matrix(op["gamma"]) for op in ops]
    spatials = [spatial_elemental(op, momentum, derivative) for op in ops]

    result = np.zeros((n_config, len(ops), len(ops), n_t), dtype=complex)
    for index, t0 in enumerate(t_sources):
        A = tau[:, index]
        conj_A = np.conj(A)
        for i, (gamma_i, spatial_i) in enumerate(zip(gammas, spatials)):
            for j, (gamma_j, spatial_j) in enumerate(zip(gammas, spatials)):
                # indices: a,k sink trace pair; p,q after Phi_i; r,s at the
                # source; u,v after Phi_j^dagger. c = config, t = sink time.
                absolute = np.einsum(
                    "ap,ctkq,ctpqrs,ur,cvs,u,a,ctakuv->ct",
                    gamma_i, spatial_i, A,
                    np.conj(gamma_j), np.conj(spatial_j[:, t0]),
                    G5_DIAG, G5_DIAG, conj_A,
                    optimize=True)
                result[:, i, j] += np.roll(absolute, -t0, axis=1)
    result /= len(t_sources)
    # Hermitize across the operator indices; exact for infinite statistics.
    result = 0.5 * (result + np.conj(np.transpose(result, (0, 2, 1, 3))))
    return np.real(result)


def principal_correlators(mean_matrix, per_config, t0, t_ref):
    """Fixed-vector GEVP: eigenvectors from (C(t_ref), C(t0)) on the
    ensemble mean, then lambda_n(t) projected per configuration so the
    principal correlators can be bootstrapped downstream."""
    n_ops, _, n_t = mean_matrix.shape
    if n_ops == 1:
        return per_config[:, 0, 0, :][:, None, :], np.eye(1)

    from scipy.linalg import eigh

    values, vectors = eigh(
        mean_matrix[:, :, t_ref], mean_matrix[:, :, t0])
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]

    projected = np.einsum(
        "ki,cklt,lj->cijt", vectors, per_config, vectors)
    diagonal = np.stack(
        [projected[:, level, level, :] for level in range(n_ops)], axis=1)
    return diagonal, vectors


def effective_mass(values, n_t):
    result = np.full(n_t, np.nan)
    for t in range(1, n_t - 1):
        if values[t] != 0.0:
            ratio = (values[t - 1] + values[t + 1]) / (2.0 * values[t])
            if np.isfinite(ratio) and ratio >= 1.0:
                result[t] = math.acosh(ratio)
    return result


def load_parameter(key, fallback=None):
    for raw in (ROOT / "parameters.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            current, value = line.split("=", 1)
            if current.strip() == key:
                return value.strip()
    if fallback is not None:
        return fallback
    raise KeyError(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, default=None,
                        help="interpolator list JSON "
                             "(default distillation_ops.json if present)")
    args = parser.parse_args()

    channels = load_operators(args.ops)
    tau = load_complex(DIST / "perambulators.npy")
    momentum = load_complex(DIST / "elementals_momentum.npy")
    derivative = load_complex(DIST / "elementals_derivative.npy")
    t_sources = [int(round(v)) for v in np.load(DIST / "t_sources.npy")]

    n_config, _, n_t = tau.shape[:3]
    n = momentum.shape[-1]
    tau = tau.reshape(n_config, len(t_sources), n_t, 2, n, 2, n)

    gevp_t0 = int(load_parameter("analysis.gevp_t0", "2"))
    t_ref = gevp_t0 + 1

    mean_rows = []
    gevp_rows = []
    by_config_columns = {}

    for channel, ops in channels.items():
        names = [op["name"] for op in ops]
        print(f"channel {channel}: ops {names}, "
              f"{n_config} configs, {len(t_sources)} sources")
        correlators = channel_correlators(
            ops, tau, t_sources, momentum, derivative)
        mean_matrix = correlators.mean(axis=0)
        error_matrix = correlators.std(axis=0, ddof=1) / math.sqrt(n_config)

        for i in range(len(ops)):
            for j in range(len(ops)):
                for t in range(n_t):
                    mean_rows.append([
                        channel, names[i], names[j], t,
                        mean_matrix[i, j, t], error_matrix[i, j, t]])

        principals, vectors = principal_correlators(
            mean_matrix, correlators, gevp_t0, t_ref)
        for level in range(principals.shape[1]):
            series = principals[:, level, :].mean(axis=0)
            masses = effective_mass(series, n_t)
            for t in range(n_t):
                gevp_rows.append([
                    channel, level, t, series[t], masses[t]])
            for t in range(n_t):
                by_config_columns[f"{channel}_n{level}_t{t}"] = \
                    principals[:, level, t]
            plateau = masses[2:n_t // 2]
            finite = plateau[np.isfinite(plateau)]
            summary = (f"m_eff ~ {np.median(finite):.5f}"
                       if finite.size else "no finite effective masses")
            print(f"  level {level}: {summary}")

    out_dir = ROOT / "output/domain_wall"
    with (out_dir / "distillation_correlators.csv").open(
            "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "op_row", "op_col", "t",
                         "mean", "standard_error"])
        writer.writerows(mean_rows)

    with (out_dir / "distillation_gevp.csv").open(
            "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "level", "t",
                         "principal_correlator", "effective_mass"])
        writer.writerows(gevp_rows)

    with (out_dir / "distillation_principal_by_config.csv").open(
            "w", newline="") as stream:
        writer = csv.writer(stream)
        keys = list(by_config_columns)
        writer.writerow(["config"] + keys)
        for config in range(n_config):
            writer.writerow(
                [config] + [by_config_columns[key][config]
                            for key in keys])

    print("Wrote:")
    print("  output/domain_wall/distillation_correlators.csv")
    print("  output/domain_wall/distillation_gevp.csv")
    print("  output/domain_wall/distillation_principal_by_config.csv")


if __name__ == "__main__":
    main()
