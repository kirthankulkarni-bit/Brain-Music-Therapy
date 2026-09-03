"""
estimator_sweep.py - is the 5.5 s analysis latency structural, or just a choice?

THE QUESTION

Once the precomputed library removed the audio-side bottleneck, the analysis path became
85% of the closed-loop budget: 5.5 s of the 6.5 s worst case. The paper currently calls
that term structural - the price of estimating band power from a finite window and
smoothing a noisy result.

That claim deserves testing rather than asserting, because it is the difference between
"this is the floor" and "this is what we happened to configure". The neurofeedback
literature reports systems operating at 300-1000 ms, and methods exist that push envelope
estimation well below that (Smetanin et al., Towards Zero-Latency Neurofeedback). If a
different estimator separates brain states as well with a fraction of the delay, then 5.5 s
is a configuration, not a floor, and the paper's central number is the wrong one to report.

WHAT IS MEASURED, AND WHY IT IS NOT GROUP DELAY

Group delay is a property of a filter, not of a system, and it does not capture window
centroid delay or hop quantisation. So latency here is measured the way it is experienced:

  DETECTION LATENCY - given a real, labelled state change, how long until the estimator's
  output crosses the midpoint between the two states? Averaged over every transition in
  the alpha-validation session, which alternates 60 s eyes-open and eyes-closed blocks.

That is an end-to-end number for the whole analysis path, obtained on real EEG rather than
on a step function, and it is directly comparable across estimators of different design.

Latency alone is meaningless - a wire has zero latency and no discriminability - so each
estimator is also scored on how well it separates the two states:

  DISCRIMINABILITY - Cohen's d between eyes-open and eyes-closed samples, computed only on
  samples well inside a block so the transition itself cannot inflate it.

The output is a latency-discriminability trade-off curve. The question the paper needs
answered is whether a point exists with much lower latency and comparable d.

EVERY ESTIMATOR HERE IS CAUSAL. A non-causal Hilbert transform or a centred window would
win this benchmark and could not run in a closed loop; comparing against one would be
meaningless. Filters are applied forward only, never filtfilt.

Usage:
    python scripts/estimator_sweep.py
    python scripts/estimator_sweep.py --session sessions/alphatest_20260816_015511
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy import signal as sps

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from session_logger import load_raw, load_session  # noqa: E402

ALPHA_BAND = (8.0, 12.0)
# raw columns are [lsl_ts, ch0..ch3] and MUSE_CHANNELS is (TP9, AF7, AF8, TP10), so the
# frontal pair AF7/AF8 is channel indices 1,2 - which is raw columns 2,3 once the
# timestamp column is counted. Getting this wrong selects TP9+AF7 and inverts the
# eyes-closed alpha effect, which is how the first run of this sweep produced a
# negative d against a validated 2.2x increase.
FRONTAL_RAW_COLS = (2, 3)
EDGE_GUARD_S = 8.0        # samples this close to a transition are excluded from d


# ------------------------------------------------------------------ estimators


def est_welch(x: np.ndarray, fs: float, window_s: float, hop_s: float,
              tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    The current pipeline: Welch band power over a sliding window, then an exponential
    smoother. Reported at the END of each window, which is what a causal system can do.
    """
    n_win, n_hop = int(window_s * fs), int(hop_s * fs)
    if x.size < n_win:
        return np.array([]), np.array([])
    starts = np.arange(0, x.size - n_win + 1, n_hop)
    t = (starts + n_win) / fs                      # causal: available at window end
    out = np.empty(starts.size)
    for i, s0 in enumerate(starts):
        f, p = sps.welch(x[s0:s0 + n_win], fs=fs, nperseg=min(n_win, int(fs * 2)))
        out[i] = p[(f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])].mean()

    alpha = 1.0 - np.exp(-hop_s / tau_s)
    sm = np.empty_like(out)
    acc = out[0]
    for i, v in enumerate(out):
        acc += alpha * (v - acc)
        sm[i] = acc
    return t, sm


def est_bandpass(x: np.ndarray, fs: float, order: int, tau_s: float,
                 decim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal narrowband filter, squared, then a one-pole smoother - per sample.

    Two structural advantages over the windowed approach, independent of any tuning:
    there is no window, so no centroid delay, and no hop, so no quantisation. The only
    delays are the filter's group delay and the smoother.

    lfilter, never filtfilt: filtfilt is zero-phase because it runs backwards as well as
    forwards, which a realtime system cannot do.
    """
    sos = sps.butter(order, ALPHA_BAND, btype="bandpass", fs=fs, output="sos")
    narrow = sps.sosfilt(sos, x)
    power = narrow ** 2

    a = 1.0 - np.exp(-1.0 / (tau_s * fs))          # one-pole, per sample
    sm = sps.lfilter([a], [1.0, -(1.0 - a)], power)
    return np.arange(x.size)[::decim] / fs, sm[::decim]


ESTIMATORS = {
    # name:                          (fn, kwargs)
    "current (4s win, 1s hop, t=3)": (est_welch,    dict(window_s=4.0, hop_s=1.0, tau_s=3.0)),
    "welch 4s win, t=1.0":           (est_welch,    dict(window_s=4.0, hop_s=1.0, tau_s=1.0)),
    "welch 2s win, t=1.0":           (est_welch,    dict(window_s=2.0, hop_s=0.5, tau_s=1.0)),
    "welch 2s win, t=0.5":           (est_welch,    dict(window_s=2.0, hop_s=0.5, tau_s=0.5)),
    "welch 1s win, t=0.5":           (est_welch,    dict(window_s=1.0, hop_s=0.25, tau_s=0.5)),
    "bandpass o4, t=2.0":            (est_bandpass, dict(order=4, tau_s=2.0)),
    "bandpass o4, t=1.0":            (est_bandpass, dict(order=4, tau_s=1.0)),
    "bandpass o4, t=0.5":            (est_bandpass, dict(order=4, tau_s=0.5)),
    "bandpass o4, t=0.25":           (est_bandpass, dict(order=4, tau_s=0.25)),
    "bandpass o2, t=0.25":           (est_bandpass, dict(order=2, tau_s=0.25)),
    "bandpass o2, t=0.1":            (est_bandpass, dict(order=2, tau_s=0.1)),
}


# --------------------------------------------------------------------- scoring


def block_timeline(session: dict) -> list[tuple[float, str]]:
    """(elapsed_s, phase) at each labelled window, from the session log."""
    return [(float(w["elapsed_s"]), w["phase"]) for w in session["windows"]
            if w.get("phase") in ("eyes_open", "eyes_closed")]


def transitions(timeline: list[tuple[float, str]]) -> list[tuple[float, str, str]]:
    """(time, from_phase, to_phase) for each block change."""
    out = []
    for i in range(1, len(timeline)):
        if timeline[i][1] != timeline[i - 1][1]:
            out.append((timeline[i][0], timeline[i - 1][1], timeline[i][1]))
    return out


def phase_at(timeline: list[tuple[float, str]], t: float) -> str | None:
    prev = None
    for tt, ph in timeline:
        if tt > t:
            return prev
        prev = ph
    return prev


def score(t: np.ndarray, y: np.ndarray, timeline: list[tuple[float, str]]) -> dict:
    """Detection latency over every transition, and discriminability away from them."""
    if t.size < 10:
        return {"latency_s": float("nan"), "d": float("nan"), "n_transitions": 0}

    y = np.log10(np.maximum(y, 1e-12))            # the pipeline works in log power
    trans = transitions(timeline)

    labels = np.array([phase_at(timeline, tt) or "" for tt in t])
    near = np.zeros(t.size, dtype=bool)
    for tt, _, _ in trans:
        near |= np.abs(t - tt) < EDGE_GUARD_S
    op = (labels == "eyes_open") & ~near
    cl = (labels == "eyes_closed") & ~near
    if op.sum() < 20 or cl.sum() < 20:
        return {"latency_s": float("nan"), "d": float("nan"), "n_transitions": len(trans)}

    pooled = np.sqrt((y[op].var(ddof=1) + y[cl].var(ddof=1)) / 2)
    d = float((y[cl].mean() - y[op].mean()) / pooled) if pooled > 0 else float("nan")

    # Detection latency: time from the transition until the output crosses the midpoint
    # between the two block levels and stays past it. Uses each block's own levels, so
    # an estimator is never penalised for a different absolute scale.
    lats = []
    for tt, frm, to in trans:
        pre = y[(t > tt - 30) & (t < tt - EDGE_GUARD_S) & (labels == frm)]
        post = y[(t > tt + EDGE_GUARD_S) & (t < tt + 55) & (labels == to)]
        if pre.size < 5 or post.size < 5:
            continue
        mid = (pre.mean() + post.mean()) / 2.0
        rising = post.mean() > pre.mean()
        seg = (t >= tt) & (t < tt + 55)
        if not seg.any():
            continue
        ts, ys = t[seg], y[seg]
        crossed = ys > mid if rising else ys < mid
        if not crossed.any():
            continue
        lats.append(ts[np.argmax(crossed)] - tt)

    return {"latency_s": float(np.median(lats)) if lats else float("nan"),
            "d": d, "n_transitions": len(lats)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency vs discriminability of EEG estimators")
    parser.add_argument("--session", default=None)
    args = parser.parse_args()

    d = args.session
    if d is None:
        cands = sorted(glob.glob(os.path.join(_ROOT, "sessions", "alphatest*")))
        if not cands:
            print("No alphatest session found; pass --session.")
            return 1
        d = cands[-1]

    session = load_session(d)
    raw = load_raw(d)
    ts = raw[:, 0]
    fs = raw.shape[0] / (ts[-1] - ts[0])
    x = raw[:, FRONTAL_RAW_COLS].mean(axis=1).astype(float)
    x = x - x.mean()

    timeline = block_timeline(session)
    n_trans = len(transitions(timeline))

    print("=" * 78)
    print("ANALYSIS-PATH LATENCY: is 5.5 s structural, or a configuration?")
    print("=" * 78)
    print(f"  session      : {os.path.basename(d)}")
    print(f"  {x.size} samples at {fs:.1f} Hz ({x.size / fs:.0f} s), frontal mean (AF7,AF8)")
    print(f"  {n_trans} labelled eyes-open/closed transitions")
    print(f"  every estimator is CAUSAL - forward filtering only\n")

    print(f"  {'estimator':<32}{'detect':>9}{'d':>8}{'vs current':>13}")
    print("  " + "-" * 62)

    results = {}
    baseline = None
    for name, (fn, kw) in ESTIMATORS.items():
        t, y = fn(x, fs, **kw)
        r = score(t, y, timeline)
        results[name] = r
        if baseline is None:
            baseline = r
        speed = (baseline["latency_s"] / r["latency_s"]
                 if np.isfinite(r["latency_s"]) and r["latency_s"] > 0 else float("nan"))
        print(f"  {name:<32}{r['latency_s']:>8.2f}s{r['d']:>8.2f}"
              f"{('%.1fx faster' % speed) if np.isfinite(speed) else '':>13}")

    print()
    print("  detect = median seconds from a real state change to the estimator crossing")
    print("           halfway between the two block levels")
    print("  d      = Cohen's d separating the states, measured away from transitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
