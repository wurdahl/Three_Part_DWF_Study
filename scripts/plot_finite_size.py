#!/usr/bin/env python3
"""Pion mass vs mpi*L with an exponential finite-volume fit.

Reads the part-1 volume-study summary and reproduces the style of
signProblem/figs/pionMassVsMpiL_beta4_Nt32.pdf: fit
m(Nx) = m_inf + A exp(-B Nx), x axis in units of the fitted
infinite-volume mass, log-2 scale with ticks at the data points.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "output/studies/part1_volume_1000/summary.csv"
FIGURE = ROOT / "figs/pionMassVsMpiL_beta3_Nt32.pdf"

JLab_blue = "#2f7a79"
JLab_red = "#c0272d"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

eb_kw = dict(ms=9, mfc="white", mew=2.2, elinewidth=2.2, capsize=4,
             capthick=2.2, ls="none", zorder=3)


def fmtErr(v, e):
    """0.7871(12)-style value with parenthesized error (2 sig figs)."""
    if not np.isfinite(e) or e <= 0:
        return f"{v:.4f}"
    ndig = max(-int(np.floor(np.log10(e))) + 1, 0)
    return f"{v:.{ndig}f}({round(e * 10**ndig):d})"


def fvExp(Nx, mInf, A, B):
    return mInf + A * np.exp(-B * Nx)


def fvLuscher(Nx, mInf, A):
    """Luscher-style wrapping correction m_inf (1 + A e^{-x}/sqrt(x)),
    x = m_inf L. The 1/2 power is the D = 2 (one spatial dimension)
    asymptotic of the wrapping propagator K_0(mL); the QCD analogue in
    D = 4 carries x^{3/2}."""
    x = mInf * Nx
    return mInf * (1.0 + A * np.exp(-x) / np.sqrt(x))


def numeric_band(model, xs, popt, pcov, rel_step=1e-6):
    """1-sigma band from a finite-difference Jacobian in the parameters."""
    center = model(xs, *popt)
    J = []
    for k, pk in enumerate(popt):
        step = rel_step * max(abs(pk), 1.0)
        shifted = list(popt)
        shifted[k] = pk + step
        J.append((model(xs, *shifted) - center) / step)
    J = np.stack(J)
    return np.sqrt(np.einsum("ix,ij,jx->x", J, pcov, J))


def chi2_dof(model, nxs, mVal, mSig, popt):
    residual = (mVal - model(nxs, *popt)) / mSig
    dof = len(nxs) - len(popt)
    return float(residual @ residual) / max(dof, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--mf", default="0.2",
                        help="fermion mass shown in the title")
    parser.add_argument("--out", type=Path, default=FIGURE)
    args = parser.parse_args()

    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    nxs = np.array([int(row["Nx"]) for row in rows])
    mVal = np.array([float(row["pion_mass"]) for row in rows])
    mSig = np.array([float(row["pion_error"]) for row in rows])

    order = np.argsort(nxs)
    nxs, mVal, mSig = nxs[order], mVal[order], mSig[order]

    p0 = [mVal[-1], mVal[0] - mVal[-1], 0.3]
    popt, pcov = curve_fit(fvExp, nxs, mVal, p0=p0, sigma=mSig,
                           absolute_sigma=True)
    mInf, dmInf = popt[0], np.sqrt(pcov[0, 0])

    poptL, pcovL = curve_fit(fvLuscher, nxs, mVal, p0=[mVal[-1], 10.0],
                             sigma=mSig, absolute_sigma=True)
    mInfL, dmInfL = poptL[0], np.sqrt(pcovL[0, 0])

    xs = np.geomspace(3.2, 43, 200)               # in Nx units; axis = mInf*Nx
    band = numeric_band(fvExp, xs, popt, pcov)
    bandL = numeric_band(fvLuscher, xs, poptL, pcovL)

    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")

    ax.axhline(mInf, color="0.8", lw=2.5, zorder=0)
    ax.fill_between(mInf * xs, fvExp(xs, *popt) - band,
                    fvExp(xs, *popt) + band,
                    color=JLab_blue, alpha=0.20, lw=0, zorder=1)
    ax.plot(mInf * xs, fvExp(xs, *popt), color=JLab_blue, lw=2.5,
            zorder=2, label=r"$m_\infty + A\,e^{-B L}$")

    ax.fill_between(mInf * xs, fvLuscher(xs, *poptL) - bandL,
                    fvLuscher(xs, *poptL) + bandL,
                    color=JLab_red, alpha=0.15, lw=0, zorder=1)
    ax.plot(mInf * xs, fvLuscher(xs, *poptL), color=JLab_red, lw=2.5,
            ls="--", zorder=2,
            label=r"$m_\infty\,(1 + A\,e^{-x}/\sqrt{x}),\ x = m_\infty L$")

    ax.errorbar(mInf * nxs, mVal, yerr=mSig,
                marker="o", color=JLab_blue, mec=JLab_blue, **eb_kw)

    ax.text(0.95, 0.90, rf"$am_\pi^\infty = {fmtErr(mInf, dmInf)}$",
            color=JLab_blue,
            ha="right", va="top", transform=ax.transAxes, fontsize=15,
            zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(0.95, 0.82, rf"$am_\pi^\infty = {fmtErr(mInfL, dmInfL)}$",
            color=JLab_red,
            ha="right", va="top", transform=ax.transAxes, fontsize=15,
            zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.78),
              fontsize=12, frameon=False)

    ax.set_xscale("log", base=2)
    ax.set_xticks(mInf * nxs, labels=[f"{x:.1f}" for x in mInf * nxs])
    ax.minorticks_off()
    ax.set_xlabel(r"$m_\pi^\infty L$")
    ax.set_ylabel(r"$a m_\pi$")
    ax.set_xlim(mInf * 3, mInf * 44)
    ax.set_ylim(bottom=0.0)

    ax.set_title(rf"$am_\pi$ vs $m_\pi^\infty L$:  $\beta = 3$, "
                 rf"$m_f = {args.mf}$, $N_t = 32$", fontsize=17, pad=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"exp:     m_inf = {fmtErr(mInf, dmInf)}   A = {popt[1]:.4f}   "
          f"B = {popt[2]:.4f}   chi2/dof = "
          f"{chi2_dof(fvExp, nxs, mVal, mSig, popt):.2f}")
    print(f"luscher: m_inf = {fmtErr(mInfL, dmInfL)}   A = {poptL[1]:.4f}"
          f"   chi2/dof = "
          f"{chi2_dof(fvLuscher, nxs, mVal, mSig, poptL):.2f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
