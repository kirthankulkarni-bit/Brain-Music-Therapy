"""
make_public_figures.py - the paper's three figures, from public and synthetic data only.

Reads the JSON written by estimator_sweep_public.py and public_analyses.py; it computes
nothing itself, so a figure can never disagree with the numbers verify_claims checks.

    python scripts/make_public_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_ROOT, "results", "public")
OUT = os.path.join(_ROOT, "docs", "figures")

INK = "#1a1a1a"
ACCENT = "#c1440e"
MUTED = "#8a8a8a"
BAND = "#4a7fb5"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def fig_estimator():
    res = load("eegbci_af.json")
    dk = f"detect_h{res['hold_s']:g}"
    s = res["summary"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    base = s["deployed"]
    bx, by = base[dk]["median"], base["R"]["median"]
    for name in ["deployed"] + res["alternatives"]:
        m = s[name]
        x, y = m[dk]["median"], m["R"]["median"]
        xerr = [[x - m[dk]["q1"]], [m[dk]["q3"] - x]]
        yerr = [[y - m["R"]["q1"]], [m["R"]["q3"] - y]]
        deployed = name == "deployed"
        stream = name.startswith("streaming")
        color = ACCENT if deployed else (INK if stream else BAND)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, alpha=0.35,
                    lw=0.9, zorder=2)
        ax.scatter(x, y, s=110 if deployed else 55,
                   marker="s" if deployed else ("^" if stream else "o"),
                   color=color, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([xl[0], bx], by, yl[1], color=BAND, alpha=0.08, zorder=0)
    ax.axhline(by, color=ACCENT, lw=0.9, ls=":", zorder=1)
    ax.axvline(bx, color=ACCENT, lw=0.9, ls=":", zorder=1)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.text(0.015, 0.97, "shaded: faster and more\ninformative than deployed",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
    ax.annotate("deployed", xy=(bx, by), xytext=(bx * 0.55, by * 0.80), fontsize=8.5,
                color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax.plot([], [], "s", color=ACCENT, label="deployed")
    ax.plot([], [], "o", color=BAND, label="windowed, retuned")
    ax.plot([], [], "^", color=INK, label="streaming (causal, per-sample)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3)
    ax.set_xlabel(f"detection latency (s, log scale; crossing held {res['hold_s']:g} s)"
                  " - lower is better")
    ax.set_ylabel("information rate R\n" + r"$d \times \sqrt{n_{ind}}$ per minute"
                  " - higher is better")
    n_dom = len(res["dominating_medians"])
    ax.set_title(f"PhysioNet EEGBCI, AF7/AF8, {res['n_sessions']} subjects "
                 f"(median and interquartile range)\n{n_dom} of "
                 f"{len(res['alternatives'])} alternatives beat the deployed setting on both axes")
    path = os.path.join(OUT, "public_estimator_tradeoff.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_autocorrelation():
    a = load("analyses.json")["autocorrelation"]
    lags = np.arange(len(a["acf_mean"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.fill_between(lags, a["acf_q1"], a["acf_q3"], color=BAND, alpha=0.18, lw=0,
                    label="interquartile range across runs")
    ax.plot(lags, a["acf_mean"], color=BAND, lw=1.8, label="mean across runs")
    ax.axhline(np.exp(-1), color=MUTED, lw=0.8, ls="--")
    ax.text(lags[-1], np.exp(-1) + 0.03, "1/e", ha="right", fontsize=8, color=MUTED)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"Deployed index on {a['n_runs']} PhysioNet resting runs (AF7/AF8)\n"
                 f"median lag-1 rho = {a['rho_median']:.3f}")
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(OUT, "public_autocorrelation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_alpha():
    al = load("analyses.json")["alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, (label, v) in zip(axes, al.items()):
        g = np.asarray(v["trace_grid"])
        for key, color, name in (("eo", MUTED, "eyes open"), ("ec", BAND, "eyes closed")):
            q = np.asarray(v[f"trace_{key}_q"])
            ax.fill_between(g, q[0], q[1], color=color, alpha=0.18, lw=0)
            ax.plot(g, v[f"trace_{key}"], color=color, lw=1.6, label=name)
        ax.axhline(1.0, color=INK, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("time in run (s)")
        ax.set_title(f"{label}: median {v['ratio_median']:.2f}x, "
                     f"{v['fraction_increase']:.0%} of {v['n_subjects']} subjects rise",
                     fontsize=9)
    axes[0].set_ylabel("alpha power / subject's eyes-open mean")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    path = os.path.join(OUT, "public_alpha_validation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_heroic():
    """Muse 2 (HEROIC): eye-state alpha at the cue, and the estimator trade-off on TP9/TP10."""
    an = load("heroic_analyses.json")["alpha"]
    res = load("heroic_tp.json")
    dk = f"detect_h{res['hold_s']:g}"
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.6, 3.4),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    for label, color in (("AF7/AF8", ACCENT), ("TP9/TP10", BAND)):
        v = an[label]
        x = np.asarray(v["trace_rel_s"])
        q = np.asarray(v["trace_q"])
        a0.fill_between(x, q[0], q[1], color=color, alpha=0.15, lw=0)
        a0.plot(x, v["trace_median"], color=color, lw=1.6,
                label=f"{label}: {v['ratio_median_subjects']:.2f}x, "
                      f"{v['subjects_increase']}/{v['n_subjects']} subjects")
    a0.axvline(0, color=INK, lw=0.8, ls=":")
    a0.text(0.5, 0.03, "cue: open eyes", transform=a0.get_xaxis_transform(), fontsize=7.5,
            color=MUTED)
    a0.axhline(1.0, color=INK, lw=0.5)
    a0.set_yscale("log")
    a0.set_yticks([0.25, 0.5, 1.0, 2.0])
    a0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a0.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a0.set_xlabel("time from cue (s)")
    a0.set_ylabel("alpha / session's eyes-closed mean")
    a0.set_title("Alpha at eye opening, Muse 2", fontsize=9)
    a0.legend(frameon=False, fontsize=7.5, loc="upper right")

    s = res["subject_summary"]
    base = s["deployed"]
    bx, by = base[dk]["median"], base["R"]["median"]
    for name in ["deployed"] + res["alternatives"]:
        m = s[name]
        x, y = m[dk]["median"], m["R"]["median"]
        deployed = name == "deployed"
        stream = name.startswith("streaming")
        color = ACCENT if deployed else (INK if stream else BAND)
        a1.scatter(x, y, s=90 if deployed else 45,
                   marker="s" if deployed else ("^" if stream else "o"),
                   color=color, zorder=3, edgecolor="white", linewidth=0.8)
    a1.set_xscale("log")
    xl, yl = a1.get_xlim(), (0, a1.get_ylim()[1] * 1.05)
    a1.fill_between([xl[0], bx], by, yl[1], color=BAND, alpha=0.08, zorder=0)
    a1.axhline(by, color=ACCENT, lw=0.9, ls=":")
    a1.axvline(bx, color=ACCENT, lw=0.9, ls=":")
    a1.set_xlim(xl)
    a1.set_ylim(yl)
    a1.set_xticks([1, 2, 4, 8])
    a1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.annotate("deployed", xy=(bx, by), xytext=(bx * 0.45, by * 0.45), fontsize=8,
                color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    a1.set_xlabel("detection latency (s, log scale)")
    a1.set_ylabel(r"information rate $R$")
    a1.set_title(f"TP9/TP10, {res['n_subjects']} subjects (subject medians)\n"
                 f"{len(res['dominating_subject_medians'])} of {len(res['alternatives'])} "
                 "alternatives beat deployed on both axes", fontsize=9)
    path = os.path.join(OUT, "public_heroic_muse.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    figs = [fig_estimator, fig_autocorrelation, fig_alpha]
    if os.path.exists(os.path.join(RES, "heroic_analyses.json")):
        figs.append(fig_heroic)
    for fn in figs:
        print(f"  OK  {os.path.basename(fn())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
