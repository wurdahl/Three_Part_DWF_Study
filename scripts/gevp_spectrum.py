#!/usr/bin/env python3
"""Two-dimensional time-shift GEVP for the pion and eta channels."""

import csv
import math

import numpy as np

from correlator_io import ROOT, load_domain_wall, load_parameter


def matrix(correlator, t):
    # Krylov basis {O, exp(-H) O}: C_ij(t)=C(t+i+j).
    return np.array([[correlator[t], correlator[t + 1]],
                     [correlator[t + 1], correlator[t + 2]]],
                    dtype=float)


def principal_correlators(correlator, t0):
    reference = 0.5 * (matrix(correlator, t0)
                       + matrix(correlator, t0).T)
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = scale * 1.0e-10
    eigenvalues = np.maximum(eigenvalues, floor)
    invsqrt = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T

    result = []
    for t in range(t0, len(correlator) - 2):
        whitened = invsqrt @ matrix(correlator, t) @ invsqrt
        values = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))[::-1]
        result.append((t, values))
    return result


def effective_masses(principal):
    rows = []
    for (t, current), (_, following) in zip(principal, principal[1:]):
        masses = []
        for a, b in zip(current, following):
            masses.append(math.log(a / b)
                          if a > 0.0 and b > 0.0 else math.nan)
        rows.append((t, current, masses))
    return rows


def main():
    t0 = int(load_parameter("analysis.gevp_t0"))
    output = ROOT / "output/domain_wall/gevp_spectrum.csv"
    all_rows = []
    for channel, column in (("pion", "CPP_average"),
                            ("eta", "eta_singlet")):
        correlator, _ = load_domain_wall(column)
        try:
            rows = effective_masses(principal_correlators(correlator, t0))
        except np.linalg.LinAlgError as problem:
            print(f"{channel}: GEVP failed: {problem}")
            continue
        for t, eigenvalues, masses in rows:
            all_rows.append(
                [channel, t, eigenvalues[0], eigenvalues[1],
                 masses[0], masses[1]])
        finite_ground = [row[2][0] for row in rows
                         if math.isfinite(row[2][0])]
        finite_excited = [row[2][1] for row in rows
                          if math.isfinite(row[2][1])]
        print(f"{channel}: {len(finite_ground)} finite ground-state and "
              f"{len(finite_excited)} finite excited-state effective masses")

    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "t", "lambda0", "lambda1",
                         "mass0", "mass1"])
        writer.writerows(all_rows)
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
