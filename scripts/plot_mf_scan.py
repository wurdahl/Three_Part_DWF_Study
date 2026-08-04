#!/usr/bin/env python3
"""Pion mass versus fermion mass, with the two-dimensional GMOR-analogue
power law m_pi = A (mf + m_res)^p fitted over the mf > 0 points.

The chiral-limit expectation for the two-flavor Schwinger model is
p = 2/3; the fitted exponent runs above that at heavy mf, so the local
slope between neighbouring masses is annotated as well."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "output/studies/part3_mf_Nt64/summary.csv"
FIGURE = ROOT / "figs/pionMassVsMf_beta3_Nx40_Nt64.pdf"

JLab_blue = "#2f7a79"
JLab_red = "#c0272d"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 16,
    "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.6, "xtick.major.width": 1.6,
    "ytick.major.width": 1.6, "xtick.major.size": 6,
    "ytick.major.size": 6, "xtick.direction": "in",
    "ytick.direction": "in",
})

eb_kw = dict(ms=9, mfc="white", mew=2.2, elinewidth=2.2, capsize=4,
             capthick=2.2, ls="none", zorder=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out", type=Path, default=FIGURE)
    parser.add_argument("--mres", type=float, default=8.2e-4)
    args = parser.parse_args()

    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mf = np.array([float(row["mf"]) for row in rows])
    mVal = np.array([float(row["pion_mass"]) for row in rows])
    mSig = np.array([float(row["pion_error"]) for row in rows])
    order = np.argsort(mf)
    mf, mVal, mSig = mf[order], mVal[order], mSig[order]

    def law(x, A, p):
        return A * np.power(x + args.mres, p)

    fit = mf > 0
    popt, pcov = curve_fit(law, mf[fit], mVal[fit], p0=[1.0, 2.0 / 3.0],
                           sigma=mSig[fit], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    xs = np.geomspace(1e-3, 0.7, 300)
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")

    ax.plot(xs, law(xs, *popt), color=JLab_blue, lw=2.5, zorder=2,
            label=rf"$A\,(m_f + m_\mathrm{{res}})^p$, $p = "
                  rf"{popt[1]:.3f}({round(perr[1] * 1000):d})$")
    ax.plot(xs, law(xs, popt[0], 2.0 / 3.0), color=JLab_red, lw=2.5,
            ls="--", zorder=2, label=r"same $A$ with $p = 2/3$")
    ax.errorbar(np.where(mf > 0, mf, args.mres), mVal, yerr=mSig,
                marker="o", color=JLab_blue, mec=JLab_blue, **eb_kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a m_f$")
    ax.set_ylabel(r"$a m_\pi$")
    ax.set_title(r"$am_\pi$ vs $am_f$:  $\beta = 3$, $N_x = 40$, "
                 r"$N_t = 64$", fontsize=17, pad=12)
    ax.legend(loc="upper left", fontsize=13, frameon=False)

    # The mf = 0 point is drawn at m_res, where the power law places it.
    ax.annotate(r"$m_f = 0$", xy=(args.mres, mVal[0]),
                xytext=(args.mres * 2.2, mVal[0] * 0.45),
                fontsize=13, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.4))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"p = {popt[1]:.4f}({perr[1]:.4f})   A = {popt[0]:.4f}")
    for i in range(len(mf[fit]) - 1):
        x0, x1 = mf[fit][i], mf[fit][i + 1]
        y0, y1 = mVal[fit][i], mVal[fit][i + 1]
        print(f"  local exponent {x0}->{x1}: "
              f"{np.log(y1 / y0) / np.log(x1 / x0):.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
