"""Figures of the paper (vector PDF, text set by LaTeX in Times so that only Type 1 fonts are embedded)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as ex

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "paper", "figs")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathptmx}",
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.3,
    "grid.color": "0.85",
    "pdf.fonttype": 42,
})

# colour-blind-safe hues (Okabe-Ito); every series also has its own marker and dash pattern
STYLE = {
    "opt": dict(color="#0072B2", marker="o", ls="-", label="optimal"),
    "cascade": dict(color="#D55E00", marker="s", ls="--", label="battery cascade"),
    "fixed_M": dict(color="#009E73", marker="^", ls="-.", label="always M"),
    "fixed_S": dict(color="#555555", marker="D", ls=":", label="always S"),
}
SITE_STYLE = {
    "dhaka": dict(color="#0072B2", marker="o", ls="-"),
    "ibadan": dict(color="#D55E00", marker="s", ls="--"),
    "niamey": dict(color="#009E73", marker="^", ls="-."),
}


def fig_battery():
    grid = ex.load_grid("grid")
    H = 8
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    for ax, p, title in zip(axes, (0.5, 0.95), (r"independent days ($p=0.5$)", r"cloudy spells ($p=0.95$)")):
        for key in ("opt", "cascade", "fixed_M", "fixed_S"):
            y = [100 * grid[(H, p, C)][key] for C in ex.CS]
            ax.plot(ex.CS, y, **STYLE[key], markerfacecolor="white" if key != "opt" else STYLE[key]["color"], markeredgewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ex.CS)
        ax.set_xticklabels([str(c) for c in ex.CS])
        ax.minorticks_off()
        ax.set_xlabel("battery (J)")
        ax.set_title(title, fontsize=8, pad=3)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"events detected (\%)")
    axes[0].set_ylim(45, 93)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, handlelength=2.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.125, right=0.995, top=0.915, bottom=0.335, wspace=0.08)
    fig.savefig(os.path.join(FIGS, "fig_battery.pdf"))
    plt.close(fig)


def fig_sites():
    d = json.load(open(os.path.join(RES, "sites.json")))
    S = d["sites"]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6))
    ax = axes[0]
    months = list(range(1, 13))
    for name, s in S.items():
        ax.plot(months, s["monthly_mean_kwh"], label=s["label"].split(",")[0], markerfacecolor="white", markeredgewidth=0.8,
                **SITE_STYLE[name])
    ax.set_xticks([1, 4, 7, 10])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    ax.set_ylabel(r"kWh\,m$^{-2}$\,day$^{-1}$")
    ax.set_title("monthly mean irradiance", fontsize=8, pad=3)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax = axes[1]
    H = 6
    for name, s in S.items():
        rows = sorted([r for r in s["settings"] if r["H"] == H], key=lambda r: r["C"])
        y = [100 * (r["shuffled_M_mean"] - r["fixed_M"]) for r in rows]
        ax.plot([r["C"] for r in rows], y, label=s["label"].split(",")[0], markerfacecolor="white", markeredgewidth=0.8,
                **SITE_STYLE[name])
    ax.set_xscale("log")
    ax.set_xticks(ex.CS)
    ax.set_xticklabels([str(c) for c in ex.CS])
    ax.minorticks_off()
    ax.set_xlabel("battery (J)")
    ax.set_ylabel("points")
    ax.set_title("cost of order, always M", fontsize=8, pad=3)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handlelength=2.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.125, right=0.99, top=0.9, bottom=0.36, wspace=0.40)
    fig.savefig(os.path.join(FIGS, "fig_sites.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_battery()
    fig_sites()
    print("figures written")
