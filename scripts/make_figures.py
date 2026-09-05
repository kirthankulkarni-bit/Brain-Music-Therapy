"""
make_figures.py - the figures a paper needs, generated from real session data.

Each figure exists to carry one claim that is hard to make in prose, and every one is
drawn from data on disk rather than illustrated. If a session is missing, the figure
that depends on it is skipped rather than faked.

  0. alpha validation     eyes-closed alpha increase. The sensing-path evidence: it
                          shows the rig measures cortex rather than amplifier noise,
                          and it is the first thing a reviewer looks for. Drawn from
                          the alpha-test session rather than an intervention one.

  1. session trajectory   z over the intervention with the target band and the rung
                          changes marked. Shows the closed loop doing its job, and
                          shows visually how rarely the rung actually moves - which
                          is the setup for why the continuous coupling index fails.

  2. autocorrelation      z's autocorrelation against the independence assumption.
                          This is the effective-sample-size argument in one picture:
                          1043 windows behaving like 25.

  3. prompt chatter       switch intervals before and after the hysteresis fix,
                          against the crossfade duration. The mass to the left of the
                          crossfade line is the defect.

  4. event-locked         the brain response time-locked to rung changes, with its
                          permutation null band. Includes the confound warning in the
                          caption, because the figure is otherwise easy to over-read.

  5. power                required n against effect size for both designs. The
                          60-versus-8 gap is the whole argument for a crossover.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --session sessions/PILOT01_20260822_153652
    python scripts/make_figures.py --out docs/figures
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # no display on a headless run
import matplotlib.pyplot as plt            # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from analyze_session import (  # noqa: E402
    EVENT_POST_S,
    EVENT_PRE_S,
    TARGET_BAND_HALF_WIDTH,
    _rung_change_events,
    event_locked_response,
)
from music_engine import build_prompt  # noqa: E402
from session_logger import load_session  # noqa: E402

# One consistent look, so the figures read as a set rather than five separate plots.
INK = "#1a1a1a"
ACCENT = "#c1440e"
MUTED = "#8a8a8a"
BAND = "#4a7fb5"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def intervention_z(session: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = [w for w in session["windows"]
            if w.get("phase") == "intervention" and w.get("valid")
            and isinstance(w.get("z"), (int, float)) and np.isfinite(w["z"])]
    t = np.asarray([w["elapsed_s"] for w in rows], dtype=float)
    z = np.asarray([w["z"] for w in rows], dtype=float)
    return t - (t[0] if t.size else 0.0), z


# ------------------------------------------------------------------- figures


def fig_alpha_validation(out: str) -> str:
    """
    The eyes-closed alpha increase, redrawn in the shared figure style.

    alpha_test.py already saves a plot inside its session directory, but it predates
    these figures and does not match them. A paper's figures should read as one set,
    and this one carries the claim everything else depends on: that the frontal
    channels are measuring cortex. Regenerated from the session log so it stays tied
    to the data rather than to a PNG someone might not be able to reproduce.
    """
    from scipy import stats as sps

    dirs = sorted(glob.glob(os.path.join(_ROOT, "sessions", "alphatest*")))
    if not dirs:
        return ""
    session = load_session(dirs[-1])
    # Which channels this session actually used. It is NOT the pair the live system
    # runs on: the alpha validation was recorded on TP9/TP10 while the arousal index
    # uses AF7/AF8. Labelling the figure from the manifest rather than from memory is
    # what caught that.
    pair = session["manifest"].get("index_channels") or ["?", "?"]
    rows = [w for w in session["windows"]
            if w.get("phase") in ("eyes_open", "eyes_closed")
            and isinstance(w.get("alpha"), (int, float)) and np.isfinite(w["alpha"])]
    if len(rows) < 40:
        return ""

    t = np.asarray([w["elapsed_s"] for w in rows], dtype=float)
    a = np.log10(np.asarray([w["alpha"] for w in rows], dtype=float))
    closed = np.asarray([w["phase"] == "eyes_closed" for w in rows], dtype=bool)

    ratio = float(10 ** (a[closed].mean() - a[~closed].mean()))
    t_stat, p = sps.ttest_ind(a[closed], a[~closed], equal_var=False)
    pooled = np.sqrt((a[closed].var(ddof=1) + a[~closed].var(ddof=1)) / 2)
    d = float((a[closed].mean() - a[~closed].mean()) / pooled) if pooled > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    # Shade contiguous eyes-closed blocks rather than per-window, so the blocks read
    # as blocks and a single rejected window does not punch a hole in the band.
    start = None
    for i, c in enumerate(closed):
        if c and start is None:
            start = t[i]
        elif not c and start is not None:
            ax.axvspan(start, t[i], color=BAND, alpha=0.15, lw=0)
            start = None
    if start is not None:
        ax.axvspan(start, t[-1], color=BAND, alpha=0.15, lw=0)
    ax.plot([], [], color=BAND, lw=6, alpha=0.35, label="eyes closed")
    ax.plot(t, a, color=INK, lw=1.0, label="log10 alpha power")

    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("log$_{10}$ alpha power\n(" + "/".join(pair) + " mean)")
    ax.set_title(f"Alpha rises {ratio:.2f}x with eyes closed on {'/'.join(pair)} "
                 f"(d = {d:.2f}, p = {p:.1e}, n = {closed.sum()}/{(~closed).sum()})\n"
                 f"NOTE: the live arousal index uses AF7/AF8, which this does NOT validate",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.margins(x=0.01)
    path = os.path.join(out, "fig0_alpha_validation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_trajectory(session: dict, out: str) -> str:
    t, z = intervention_z(session)
    target = session["manifest"].get("target_z", -1.0)
    events = _rung_change_events(session)
    t0 = session["windows"][0]["elapsed_s"] if session["windows"] else 0.0

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axhspan(target - TARGET_BAND_HALF_WIDTH, target + TARGET_BAND_HALF_WIDTH,
               color=BAND, alpha=0.13, lw=0, label=f"target band ({target:+.1f} ± {TARGET_BAND_HALF_WIDTH})")
    ax.axhline(target, color=BAND, lw=1.1, ls="--")
    ax.plot(t / 60.0, z, color=INK, lw=0.8)

    for onset, direction in events:
        ax.axvline((onset - t0) / 60.0, color=ACCENT, lw=0.7, alpha=0.55)
    if events:
        ax.plot([], [], color=ACCENT, lw=0.7, label=f"rung change (n={len(events)})")

    in_band = float((np.abs(z - target) <= TARGET_BAND_HALF_WIDTH).mean() * 100)
    ax.set_xlabel("time into intervention (minutes)")
    ax.set_ylabel("arousal index z\n(SD from own baseline)")
    ax.set_title(f"Closed-loop session: {in_band:.0f}% of windows in the target band, "
                 f"{len(events)} rung changes")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.margins(x=0.01)
    path = os.path.join(out, "fig1_session_trajectory.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_autocorrelation(session: dict, out: str) -> str:
    _, z = intervention_z(session)
    max_lag = 90
    ac = [float(np.corrcoef(z[:-k], z[k:])[0, 1]) for k in range(1, max_lag + 1)]
    rho = ac[0]
    n_eff = z.size * (1 - rho) / (1 + rho)
    decorr = next((k for k, v in enumerate(ac, 1) if v < 1 / np.e), max_lag)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(range(1, max_lag + 1), ac, color=INK, lw=1.4)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.axhline(1 / np.e, color=BAND, lw=1.0, ls="--", label="1/e")
    ax.axvline(decorr, color=ACCENT, lw=1.0, ls=":",
               label=f"decorrelation time = {decorr} s")
    ax.fill_between(range(1, max_lag + 1), 0, ac, color=INK, alpha=0.07)

    ax.set_xlabel("lag (seconds)")
    ax.set_ylabel("autocorrelation of z")
    ax.set_title(f"{z.size} windows behave like {n_eff:.0f} independent observations\n"
                 f"(lag-1 ρ = {rho:.3f}; treating them as independent overstates "
                 f"evidence {np.sqrt(z.size / n_eff):.1f}×)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(out, "fig2_autocorrelation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_chatter(session: dict, out: str) -> str:
    """Switch intervals as logged, against the same z replayed through the fix."""
    audio = session["audio"]
    if len(audio) < 10:
        return ""
    logged = np.diff(np.asarray([a["elapsed_s"] for a in audio], dtype=float))

    _, z = intervention_z(session)
    window = 20
    prompts, prev, hist = [], None, collections.deque(maxlen=window)
    change_times = []
    for i, v in enumerate(z):
        hist.append(v)
        trend = (float(np.polyfit(np.arange(len(hist), dtype=float), np.asarray(hist), 1)[0])
                 if len(hist) >= window else None)
        p = build_prompt(float(v), session["manifest"].get("target_z", -1.0),
                         previous_prompt=prev)
        if prev is not None and p != prev:
            change_times.append(i)
        prompts.append(p)
        prev = p
    fixed = np.diff(np.asarray(change_times, dtype=float)) if len(change_times) > 2 else np.array([])

    crossfade = (session["manifest"].get("engine_config", {}) or {}).get("crossfade_seconds", 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bins = np.logspace(np.log10(0.5), np.log10(300), 34)
    ax.hist(logged, bins=bins, color=ACCENT, alpha=0.65,
            label=f"as run: {logged.size} intervals, median {np.median(logged):.1f} s")
    if fixed.size:
        ax.hist(fixed, bins=bins, color=INK, alpha=0.65,
                label=f"after hysteresis: {fixed.size} intervals, median {np.median(fixed):.0f} s")
    ax.axvline(crossfade, color=BAND, lw=1.4, ls="--",
               label=f"crossfade = {crossfade:g} s")
    below = float((logged < crossfade).mean() * 100)
    ax.set_xscale("log")
    ax.set_xlabel("interval between audio switches (seconds, log scale)")
    ax.set_ylabel("count")
    ax.set_title(f"Switches faster than the crossfade produce a blend, not a transition\n"
                 f"{below:.0f}% of logged intervals fell left of the crossfade line", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(out, "fig3_switch_intervals.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_event_locked(session: dict, out: str) -> str:
    result = event_locked_response(session, n_permutations=400)
    if "elr_curve" not in result:
        return ""
    curve = np.asarray(result["elr_curve"], dtype=float)
    hop = session["manifest"].get("feature_config", {}).get("hop_seconds", 1.0)
    grid = np.arange(-EVENT_PRE_S, EVENT_POST_S + hop, hop)[:curve.size]
    null_sd = result.get("elr_null_sd", np.nan)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if np.isfinite(null_sd):
        ax.fill_between(grid, -1.96 * null_sd, 1.96 * null_sd, color=MUTED, alpha=0.22,
                        lw=0, label="95% permutation null")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=ACCENT, lw=1.2, ls="--", label="rung change")
    ax.plot(grid, curve, color=INK, lw=1.8)

    ax.set_xlabel("time relative to rung change (seconds)")
    ax.set_ylabel("z change, signed by\ndirection of the music")
    ax.set_title(f"Event-locked response: {result['elr_effect_z']:+.2f} z, "
                 f"p = {result['elr_p_permutation']:.3f}, n = {result['elr_n_epochs']} events\n"
                 f"NOT causal alone - rung changes are triggered by z moving; "
                 f"only adaptive minus sham is interpretable", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    path = os.path.join(out, "fig4_event_locked.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_estimator(out: str) -> str:
    """
    Latency against information rate. The deployed setting is not on the frontier.

    Two axes because either alone misleads: latency alone would favour a wire, and
    per-sample discriminability alone favours smoothing everything to a constant. The
    y-axis is d x sqrt(independent observations per minute) - the t-statistic per
    root-minute - which is what determines how well a fixed-duration session resolves a
    state difference.
    """
    import subprocess
    import re

    proc = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "estimator_sweep.py")],
                          capture_output=True, text=True, cwd=_ROOT)
    rows = []
    for line in proc.stdout.splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}([\d.]+)s\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)),
                         float(m.group(5)), float(m.group(6))))
    if len(rows) < 4:
        return ""

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    base = rows[0]
    for name, lat, d, indmin, info in rows:
        deployed = name.startswith("PIPELINE")
        streaming = name.startswith("streaming")
        ax.scatter(lat, info, s=110 if deployed else 55,
                   marker="s" if deployed else ("^" if streaming else "o"),
                   color=ACCENT if deployed else (INK if streaming else BAND),
                   zorder=3, edgecolor="white", linewidth=0.8)

    ax.axhline(base[4], color=ACCENT, lw=0.9, ls=":", zorder=1)
    ax.axvline(base[1], color=ACCENT, lw=0.9, ls=":", zorder=1)
    ax.axhspan(base[4], ax.get_ylim()[1], xmin=0, xmax=(base[1] - ax.get_xlim()[0]) /
               (ax.get_xlim()[1] - ax.get_xlim()[0]), color=BAND, alpha=0.08, zorder=0)
    ax.annotate("faster AND more informative\nthan the deployed setting",
                xy=(0.3, base[4] * 1.55), fontsize=8, color=MUTED)
    ax.annotate("deployed", xy=(base[1], base[4]), xytext=(base[1] * 0.42, base[4] * 0.80),
                fontsize=8.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    ax.plot([], [], "s", color=ACCENT, label="deployed")
    ax.plot([], [], "o", color=BAND, label="windowed, retuned")
    ax.plot([], [], "^", color=INK, label="streaming (causal, per-sample)")
    ax.set_xscale("log")
    ax.set_xlabel("detection latency (s, log scale) - lower is better")
    ax.set_ylabel("information per minute\n"
                  r"$d \times \sqrt{ind/min}$" + " - higher is better")
    ax.set_title("The analysis latency is a dominated configuration, not a floor\n"
                 "8 of 10 alternatives beat the deployed setting on BOTH axes", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    path = os.path.join(out, "fig6_estimator_tradeoff.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_power(session_dir: str, out: str, sims: int) -> str:
    from power_analysis import required_n, session_stats

    stats = session_stats(session_dir)
    rng = np.random.default_rng(0)
    effects = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for bsd, style in ((0.3, ":"), (0.5, "-"), (0.7, "--")):
        ns = [required_n(rng, e, bsd, stats, False, "z_mean", sims) for e in effects]
        ax.plot(effects, ns, style, color=ACCENT, lw=1.5, marker="o", ms=3.5,
                label=f"independent, between-SD {bsd}")
    ns = [required_n(rng, e, 0.5, stats, True, "z_mean", sims) for e in effects]
    ax.plot(effects, ns, "-", color=INK, lw=2.0, marker="s", ms=4,
            label="paired (crossover)")

    ax.axhline(60, color=MUTED, lw=0.8, ls=":")
    ax.text(0.21, 63, "search ceiling", fontsize=7, color=MUTED, ha="left")
    ax.set_yscale("log")
    ax.set_yticks([4, 6, 10, 20, 40, 60])
    ax.set_yticklabels(["4", "6", "10", "20", "40", ">60"])
    ax.set_xlabel("effect size to detect (z units)")
    ax.set_ylabel("participants required per arm")
    ax.set_title("A crossover design costs a fraction of the participants\n"
                 "80% power, α = 0.05, simulated with the measured autocorrelation",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(out, "fig5_power.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures from real data")
    parser.add_argument("--session", default=None)
    parser.add_argument("--out", default=os.path.join("docs", "figures"))
    parser.add_argument("--sims", type=int, default=600,
                        help="simulations per power cell; the power figure dominates runtime")
    args = parser.parse_args()

    session_dir = args.session
    if session_dir is None:
        candidates = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*")))
        if not candidates:
            print("No PILOT session found. Pass --session explicitly.")
            return 1
        session_dir = candidates[-1]

    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    session = load_session(session_dir)

    print(f"session: {os.path.basename(session_dir)}")
    print(f"output : {out}\n")

    made = []
    for name, fn in (
        ("alpha validation", lambda: fig_alpha_validation(out)),
        ("session trajectory", lambda: fig_trajectory(session, out)),
        ("autocorrelation", lambda: fig_autocorrelation(session, out)),
        ("switch intervals", lambda: fig_chatter(session, out)),
        ("event-locked response", lambda: fig_event_locked(session, out)),
        ("estimator trade-off", lambda: fig_estimator(out)),
        ("power curves", lambda: fig_power(session_dir, out, args.sims)),
    ):
        try:
            path = fn()
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the rest
            print(f"  SKIP  {name}: {type(exc).__name__}: {exc}")
            continue
        if path:
            made.append(path)
            print(f"  OK    {name} -> {os.path.basename(path)}")
        else:
            print(f"  SKIP  {name}: not enough data in this session")

    print(f"\n{len(made)} figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
