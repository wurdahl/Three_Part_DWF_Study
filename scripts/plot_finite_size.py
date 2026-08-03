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


def fvInvL(Nx, mInf, c):
    """Gapless-channel finite-size form: a critical (CFT) sector has all
    energies scaling as 1/L, so m(L) = m_inf + c/L with m_inf ~ 0."""
    return mInf + c / Nx


def read_residual_masses(summary):
    """Mean m_res per volume from the archived per-case correlator.csv,
    averaged over the analyzer's central window t = 3..Nt-4."""
    result = {}
    for case in sorted(summary.parent.glob("Nx_*/correlator.csv")):
        with case.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = [float(row["m_res"]) for row in rows[3:len(rows) - 3]]
        result[int(case.parent.name.split("_")[1])] = \
            sum(values) / len(values)
    return result


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
    parser.add_argument("--nt", default="32",
                        help="temporal extent shown in the title")
    parser.add_argument("--out", type=Path, default=FIGURE)
    parser.add_argument("--channel", choices=["pion", "eta"],
                        default="pion",
                        help="summary columns and labels to use")
    parser.add_argument("--invL", action="store_true",
                        help="also fit the gapless form m_inf + c/L and "
                             "plot against Nx instead of m_inf*L")
    parser.add_argument("--invL-min-nx", type=int, default=8,
                        help="smallest Nx included in the c/L fit; the "
                             "1/L form is asymptotic and tiny boxes are "
                             "not in that regime (default 8)")
    parser.add_argument("--mres", action="store_true",
                        help="mark the residual mass from the archived "
                             "per-case correlator.csv files")
    args = parser.parse_args()

    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    nxs = np.array([int(row["Nx"]) for row in rows])
    sym = r"\pi" if args.channel == "pion" else r"{\eta'}"
    mVal = np.array([float(row[f"{args.channel}_mass"]) for row in rows])
    mSig = np.array([float(row[f"{args.channel}_error"]) for row in rows])

    order = np.argsort(nxs)
    nxs, mVal, mSig = nxs[order], mVal[order], mSig[order]

    p0 = [mVal[-1], mVal[0] - mVal[-1], 0.3]
    popt, pcov = curve_fit(fvExp, nxs, mVal, p0=p0, sigma=mSig,
                           absolute_sigma=True)
    mInf, dmInf = popt[0], np.sqrt(pcov[0, 0])
    chi2 = chi2_dof(fvExp, nxs, mVal, mSig, popt)

    # With a second fit on the plot an m_inf-scaled axis is ambiguous, so
    # the invL variant plots directly against Nx.
    scale = 1.0 if args.invL else mInf
    x_low = nxs[0] * 0.82
    x_high = nxs[-1] * 1.18
    xs = np.geomspace(x_low, x_high, 200)         # in Nx units
    band = numeric_band(fvExp, xs, popt, pcov)

    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")

    ax.axhline(mInf, color="0.8", lw=2.5, zorder=0)
    ax.fill_between(scale * xs, fvExp(xs, *popt) - band,
                    fvExp(xs, *popt) + band,
                    color=JLab_blue, alpha=0.20, lw=0, zorder=1)
    ax.plot(scale * xs, fvExp(xs, *popt), color=JLab_blue, lw=2.5,
            zorder=2, label=r"$m_\infty + A\,e^{-B L}$")

    annotations = [
        (rf"$am_{sym}^\infty = {fmtErr(mInf, dmInf)}$"
         rf"$,\ \chi^2/\mathrm{{dof}} = {chi2:.1f}$", JLab_blue)]

    if args.invL:
        asymptotic = nxs >= args.invL_min_nx
        nxsI, mValI, mSigI = nxs[asymptotic], mVal[asymptotic], \
            mSig[asymptotic]
        poptI, pcovI = curve_fit(
            fvInvL, nxsI, mValI, p0=[0.0, mValI[0] * nxsI[0]],
            sigma=mSigI, absolute_sigma=True)
        mInfI, dmInfI = poptI[0], np.sqrt(pcovI[0, 0])
        chi2I = chi2_dof(fvInvL, nxsI, mValI, mSigI, poptI)
        bandI = numeric_band(fvInvL, xs, poptI, pcovI)
        ax.fill_between(scale * xs, fvInvL(xs, *poptI) - bandI,
                        fvInvL(xs, *poptI) + bandI,
                        color=JLab_red, alpha=0.15, lw=0, zorder=1)
        ax.plot(scale * xs, fvInvL(xs, *poptI), color=JLab_red, lw=2.5,
                ls="--", zorder=2,
                label=rf"$m_\infty + c/L$  ($N_x \geq {args.invL_min_nx}$)")
        annotations.append(
            (rf"$am_{sym}^\infty = {fmtErr(mInfI, dmInfI)}$"
             rf"$,\ c = {fmtErr(poptI[1], np.sqrt(pcovI[1, 1]))}$"
             rf"$,\ \chi^2/\mathrm{{dof}} = {chi2I:.1f}$", JLab_red))
        print(f"invL (Nx >= {args.invL_min_nx}): "
              f"m_inf = {fmtErr(mInfI, dmInfI)}   "
              f"c = {poptI[1]:.4f}   chi2/dof = {chi2I:.2f}")

    if args.mres:
        residuals = read_residual_masses(args.summary)
        if residuals:
            mres = sum(residuals.values()) / len(residuals)
            ax.axhline(mres, color="0.35", lw=1.8, ls=":", zorder=0)
            annotations.append(
                (rf"$am_\mathrm{{res}} = {mres:.5f}$", "0.35"))
            print(f"m_res per volume: "
                  + "  ".join(f"Nx={nx}: {value:.6f}"
                              for nx, value in sorted(residuals.items())))

    for index, (text, color) in enumerate(annotations):
        ax.text(0.95, 0.90 - 0.08 * index, text, color=color,
                ha="right", va="top", transform=ax.transAxes, fontsize=14,
                zorder=5, bbox=dict(facecolor="white", edgecolor="none",
                                    pad=2))
    ax.legend(loc="upper right",
              bbox_to_anchor=(0.98, 0.88 - 0.08 * len(annotations)),
              fontsize=12, frameon=False)

    ax.errorbar(scale * nxs, mVal, yerr=mSig,
                marker="o", color=JLab_blue, mec=JLab_blue, **eb_kw)

    ax.set_xscale("log", base=2)
    if args.invL:
        ax.set_xticks(nxs, labels=[str(nx) for nx in nxs])
        ax.set_xlabel(r"$N_x$")
    else:
        ax.set_xticks(mInf * nxs, labels=[f"{x:.1f}" for x in mInf * nxs])
        ax.set_xlabel(rf"$m_{sym}^\infty L$")
    ax.minorticks_off()
    ax.set_ylabel(rf"$a m_{sym}$")
    ax.set_xlim(scale * x_low, scale * x_high)
    ax.set_ylim(bottom=0.0)

    x_name = r"$N_x$" if args.invL else rf"$m_{sym}^\infty L$"
    ax.set_title(rf"$am_{sym}$ vs {x_name}:  $\beta = 3$, "
                 rf"$m_f = {args.mf}$, $N_t = {args.nt}$", fontsize=17, pad=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"exp: m_inf = {fmtErr(mInf, dmInf)}   A = {popt[1]:.4f}   "
          f"B = {popt[2]:.4f}   chi2/dof = {chi2:.2f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
