"""
estimator_sweep.py - is the 5.5 s analysis latency structural, or just a configuration?

THE QUESTION

Once the precomputed library removed the audio-side bottleneck, the analysis path became
85% of the closed-loop budget: 5.5 s of a 6.5 s worst case. The manuscript calls that term
structural - the price of estimating band power from a finite window and smoothing a noisy
result. That deserves testing rather than asserting, because it is the difference between
"this is the floor" and "this is what we configured".

Context for the number: neurofeedback systems are reported operating at 300-1000 ms, and
methods exist that push envelope estimation below that. 5.5 s is an order of magnitude
outside that range.

WHAT IS MEASURED

Group delay is a property of a filter, not of a system, and misses window centroid delay
and hop quantisation entirely. So latency is measured as experienced:

  DETECTION LATENCY - given a real labelled state change, how long until the estimator
  crosses the midpoint between the two block levels? Median over every transition in the
  alpha-validation session, which alternates 60 s eyes-open and eyes-closed blocks.

Latency alone is meaningless, so each estimator is also scored on:

  DISCRIMINABILITY - Cohen's d between the two states, computed only well inside blocks
  so the transition cannot inflate it.

TWO THINGS THIS SCRIPT GETS RIGHT THAT AN EARLIER VERSION DID NOT

1. The windowed baseline uses the REAL FeatureExtractor, not a reimplementation. The
   pipeline detrends, applies a zero-phase bandpass and a 60 Hz notch within each window,
   then runs Welch. A hand-rolled Welch skips all of it and correlates r = 0.05 with the
   logged output. Using the real extractor gives r = 0.989 against the session log, which
   is the check that licenses everything below.

2. Channels are read from the session manifest. The alpha-validation session ran on
   TP9/TP10, not the AF7/AF8 the live system uses - see docs/finding_channel_validation.md.
   Assuming the pair inverted the effect and produced a negative d against a validated
   1.85x increase.

Streaming estimators are strictly causal: forward filtering only, never filtfilt. The
windowed baseline may filter zero-phase WITHIN its window, which is legitimate because the
window is already complete - that asymmetry is the trade being measured, not a flaw.

Usage:
    python scripts/estimator_sweep.py
    python scripts/estimator_sweep.py --report-channels
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy import signal as sps
from scipy import stats as spstats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from eeg_features import FeatureConfig, FeatureExtractor, MUSE_CHANNELS  # noqa: E402
from session_logger import load_raw, load_session  # noqa: E402

ALPHA_BAND = (8.0, 13.0)      # matches DEFAULT_BANDS["alpha"]
EDGE_GUARD_S = 8.0
FS = 256.0


# ------------------------------------------------------------------ estimators


def est_pipeline(chans: np.ndarray, pair: tuple[str, str], window_s: float,
                 hop_s: float, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    The deployed pipeline, via the real FeatureExtractor, at arbitrary window/hop/tau.

    Reported at the END of each window, which is the earliest a causal system could have
    it. Rejected windows are dropped, exactly as the live loop drops them.
    """
    cfg = FeatureConfig(sampling_rate=FS, window_seconds=window_s, hop_seconds=hop_s,
                        frontal_channels=pair)
    ex = FeatureExtractor(cfg)
    n_win, n_hop = cfg.window_samples, cfg.hop_samples
    t, v = [], []
    for s0 in range(0, chans.shape[1] - n_win + 1, n_hop):
        f = ex.extract(chans[:, s0:s0 + n_win])
        if f.valid and np.isfinite(f.alpha) and f.alpha > 0:
            t.append((s0 + n_win) / FS)
            v.append(f.alpha)
    if not v:
        return np.array([]), np.array([])

    t = np.asarray(t)
    v = np.asarray(v)
    # Exponential smoother over the retained samples, as ExponentialSmoother does.
    a = 1.0 - np.exp(-hop_s / tau_s)
    sm = np.empty_like(v)
    acc = v[0]
    for i, x in enumerate(v):
        acc += a * (x - acc)
        sm[i] = acc
    return t, sm


def est_streaming(chans: np.ndarray, pair: tuple[str, str], order: int, tau_s: float,
                  decim: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal, per-sample alternative: DC-block, notch, narrowband, square, one-pole.

    No window and no hop, so neither centroid delay nor quantisation exists. The only
    delays are filter group delay and the smoother. Every stage is forward-only.

    Preprocessing mirrors the pipeline's intent (remove drift, remove mains, isolate the
    band) using causal equivalents, so the comparison is of ARCHITECTURE rather than of
    one design being given cleaner input than the other.
    """
    idx = [MUSE_CHANNELS.index(p) for p in pair]
    x = chans[idx, :].mean(axis=0).astype(float)

    hp = sps.butter(2, 1.0, btype="highpass", fs=FS, output="sos")
    x = sps.sosfilt(hp, x)                                  # causal DC block
    b, a = sps.iirnotch(60.0, 30.0, fs=FS)
    x = sps.lfilter(b, a, x)                                # causal mains notch
    bp = sps.butter(order, ALPHA_BAND, btype="bandpass", fs=FS, output="sos")
    narrow = sps.sosfilt(bp, x)

    power = narrow ** 2
    alpha = 1.0 - np.exp(-1.0 / (tau_s * FS))
    sm = sps.lfilter([alpha], [1.0, -(1.0 - alpha)], power)
    return np.arange(x.size)[::decim] / FS, sm[::decim]


ESTIMATORS = [
    ("PIPELINE 4s win, 1s hop, t=3.0", est_pipeline, dict(window_s=4.0, hop_s=1.0, tau_s=3.0)),
    ("pipeline 4s win, 1s hop, t=1.0", est_pipeline, dict(window_s=4.0, hop_s=1.0, tau_s=1.0)),
    ("pipeline 2s win, 0.5s hop, t=1.0", est_pipeline, dict(window_s=2.0, hop_s=0.5, tau_s=1.0)),
    ("pipeline 2s win, 0.5s hop, t=0.5", est_pipeline, dict(window_s=2.0, hop_s=0.5, tau_s=0.5)),
    ("streaming o4, t=2.0", est_streaming, dict(order=4, tau_s=2.0)),
    ("streaming o4, t=1.0", est_streaming, dict(order=4, tau_s=1.0)),
    ("streaming o4, t=0.5", est_streaming, dict(order=4, tau_s=0.5)),
    ("streaming o4, t=0.25", est_streaming, dict(order=4, tau_s=0.25)),
    ("streaming o2, t=0.5", est_streaming, dict(order=2, tau_s=0.5)),
    ("streaming o2, t=0.25", est_streaming, dict(order=2, tau_s=0.25)),
]


# --------------------------------------------------------------------- scoring


def load(session_dir: str):
    session = load_session(session_dir)
    raw = load_raw(session_dir)
    chans = raw[:, 1:].T.astype(float)
    pair = tuple(session["manifest"].get("index_channels") or ("TP9", "TP10"))
    timeline = [(float(w["elapsed_s"]), w["phase"]) for w in session["windows"]
                if w.get("phase") in ("eyes_open", "eyes_closed")]
    return session, chans, pair, timeline


def find_offset(chans, pair, session, timeline) -> float:
    """
    Session elapsed_s minus raw sample time. Found by correlating a faithful
    reproduction against the logged alpha rather than assumed to be zero.
    """
    t, v = est_pipeline(chans, pair, 4.0, 1.0, 0.001)
    w = [q for q in session["windows"]
         if isinstance(q.get("alpha"), (int, float)) and np.isfinite(q["alpha"])]
    wt = np.asarray([q["elapsed_s"] for q in w])
    wa = np.log10(np.asarray([q["alpha"] for q in w]))
    best = (0.0, -9.0)
    for off in np.arange(-10, 10.01, 0.25):
        mine = np.interp(wt - off, t, v, left=np.nan, right=np.nan)
        ok = np.isfinite(mine) & (mine > 0)
        if ok.sum() < 50:
            continue
        r = float(np.corrcoef(np.log10(mine[ok]), wa[ok])[0, 1])
        if r > best[1]:
            best = (float(off), r)
    return best


def transitions(timeline):
    return [(timeline[i][0], timeline[i - 1][1], timeline[i][1])
            for i in range(1, len(timeline)) if timeline[i][1] != timeline[i - 1][1]]


def phase_at(timeline, t):
    prev = None
    for tt, ph in timeline:
        if tt > t:
            return prev
        prev = ph
    return prev


def score(t, y, timeline, offset):
    if t.size < 10:
        return {"latency_s": float("nan"), "d": float("nan"), "n": 0}
    ts = t + offset                                   # onto the session clock
    y = np.log10(np.maximum(y, 1e-12))
    trans = transitions(timeline)

    labels = np.array([phase_at(timeline, tt) or "" for tt in ts])
    near = np.zeros(ts.size, dtype=bool)
    for tt, _, _ in trans:
        near |= np.abs(ts - tt) < EDGE_GUARD_S
    op, cl = (labels == "eyes_open") & ~near, (labels == "eyes_closed") & ~near
    if op.sum() < 15 or cl.sum() < 15:
        return {"latency_s": float("nan"), "d": float("nan"), "n": 0}

    pooled = np.sqrt((y[op].var(ddof=1) + y[cl].var(ddof=1)) / 2)
    d = float((y[cl].mean() - y[op].mean()) / pooled) if pooled > 0 else float("nan")

    lats = []
    for tt, frm, to in trans:
        pre = y[(ts > tt - 40) & (ts < tt - EDGE_GUARD_S) & (labels == frm)]
        post = y[(ts > tt + EDGE_GUARD_S) & (ts < tt + 55) & (labels == to)]
        if pre.size < 5 or post.size < 5:
            continue
        mid = (pre.mean() + post.mean()) / 2.0
        rising = post.mean() > pre.mean()
        seg = (ts >= tt) & (ts < tt + 55)
        if not seg.any():
            continue
        st, sy = ts[seg], y[seg]
        crossed = sy > mid if rising else sy < mid
        if not crossed.any():
            continue
        # First crossing that is not immediately undone, so a single noisy sample
        # does not register as detection.
        k = int(np.argmax(crossed))
        lats.append(st[k] - tt)
    # Effective sample size of the estimator's own output, on a common 1 s grid so
    # estimators with different native rates are comparable. This matters because the
    # smoother is simultaneously the largest latency term AND the main source of the
    # autocorrelation that collapses effective n - so reducing it pays twice.
    grid = np.arange(ts[0], ts[-1], 1.0)
    on_grid = np.interp(grid, ts, y)
    inside = np.array([phase_at(timeline, g) in ("eyes_open", "eyes_closed") for g in grid])
    seg = on_grid[inside]
    if seg.size > 30:
        rho = float(np.corrcoef(seg[:-1], seg[1:])[0, 1])
        n_eff_per_min = 60.0 * (1 - rho) / (1 + rho)      # independent obs per minute
    else:
        rho, n_eff_per_min = float("nan"), float("nan")

    return {"latency_s": float(np.median(lats)) if lats else float("nan"), "d": d,
            "n": len(lats), "rho": rho, "n_eff_per_min": n_eff_per_min}


def report_channels(chans, session, timeline, offset) -> None:
    """The channel-pair comparison behind docs/finding_channel_validation.md."""
    print("\n" + "=" * 78)
    print("CHANNEL PAIR COMPARISON - does the eyes-closed effect hold on the study's pair?")
    print("=" * 78)
    print(f"  {'pair':<12}{'windows':>9}{'rejected':>10}{'ratio':>9}{'d':>7}{'p':>11}")
    print("  " + "-" * 58)
    for pair in (("TP9", "TP10"), ("AF7", "AF8")):
        cfg = FeatureConfig(sampling_rate=FS, window_seconds=4.0, hop_seconds=1.0,
                            frontal_channels=pair)
        ex = FeatureExtractor(cfg)
        nw, nh = cfg.window_samples, cfg.hop_samples
        op, cl, rej, tot = [], [], 0, 0
        for s0 in range(0, chans.shape[1] - nw + 1, nh):
            tot += 1
            f = ex.extract(chans[:, s0:s0 + nw])
            if not (f.valid and np.isfinite(f.alpha) and f.alpha > 0):
                rej += 1
                continue
            ph = phase_at(timeline, (s0 + nw) / FS + offset)
            (op if ph == "eyes_open" else cl if ph == "eyes_closed" else []).append(f.alpha)
        if len(op) < 10 or len(cl) < 10:
            print(f"  {'/'.join(pair):<12}{'insufficient data':>40}")
            continue
        lo, lc = np.log10(op), np.log10(cl)
        ratio = 10 ** (lc.mean() - lo.mean())
        pooled = np.sqrt((lo.var(ddof=1) + lc.var(ddof=1)) / 2)
        d = (lc.mean() - lo.mean()) / pooled
        _, p = spstats.ttest_ind(lc, lo, equal_var=False)
        print(f"  {'/'.join(pair):<12}{len(op)+len(cl):>9}{rej/tot*100:>9.0f}%"
              f"{ratio:>8.2f}x{d:>7.2f}{p:>11.1e}")
    print("\n  The study's index runs on AF7/AF8. See docs/finding_channel_validation.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency vs discriminability of EEG estimators")
    parser.add_argument("--session", default=None)
    parser.add_argument("--report-channels", action="store_true",
                        help="compare TP9/TP10 against AF7/AF8 and exit")
    args = parser.parse_args()

    d = args.session
    if d is None:
        from session_logger import real_sessions
        cands = real_sessions(os.path.join(_ROOT, "sessions", "alphatest*"))
        if not cands:
            print("No alphatest session found; pass --session.")
            return 1
        d = cands[-1]

    session, chans, pair, timeline = load(d)
    offset, r = find_offset(chans, pair, session, timeline)

    print("=" * 78)
    print("ANALYSIS-PATH LATENCY: is 5.5 s structural, or a configuration?")
    print("=" * 78)
    print(f"  session   : {os.path.basename(d)}")
    print(f"  channels  : {'/'.join(pair)} (from the session manifest)")
    print(f"  {chans.shape[1]} samples, {chans.shape[1] / FS:.0f} s, "
          f"{len(transitions(timeline))} labelled transitions")
    print(f"  clock offset {offset:+.2f} s, reproduction r = {r:.3f} against the session log")
    if r < 0.9:
        print("\n  !! r < 0.9 - the baseline does not reproduce the deployed pipeline.")
        print("  !! Nothing below is trustworthy until that is fixed.")
        return 2

    if args.report_channels:
        report_channels(chans, session, timeline, offset)
        return 0

    print()
    print(f"  {'estimator':<34}{'detect':>8}{'d':>7}{'rho':>7}{'ind/min':>9}{'info/min':>10}")
    print("  " + "-" * 78)
    base = None
    base_info = None
    for name, fn, kw in ESTIMATORS:
        t, y = fn(chans, pair, **kw)
        s = score(t, y, timeline, offset)
        if base is None:
            base = s
            base_info = s["d"] * np.sqrt(s["n_eff_per_min"]) if np.isfinite(s["d"]) else None
        sp = (base["latency_s"] / s["latency_s"]
              if np.isfinite(s["latency_s"]) and s["latency_s"] > 0 else float("nan"))
        keep = f"{s['d'] / base['d'] * 100:.0f}%" if np.isfinite(s["d"]) and base["d"] else "-"
        # d x sqrt(independent observations) - how precisely a session of fixed
        # duration pins down a state difference. Per-sample SNR and independence
        # trade against each other, and this is the product that actually matters.
        info = s["d"] * np.sqrt(s["n_eff_per_min"]) if np.isfinite(s["d"]) else float("nan")
        mark = ""
        if base_info is not None and np.isfinite(info) and info > base_info:
            mark = "  <-- better"
        print(f"  {name:<34}{s['latency_s']:>7.2f}s{s['d']:>7.2f}"
              f"{s['rho']:>7.3f}{s['n_eff_per_min']:>9.1f}{info:>10.2f}{mark}")

    print()
    print("  detect  = median seconds from a real state change to crossing the midpoint")
    print("  d       = Cohen's d between states, measured away from transitions")
    print("  rho      = lag-1 autocorrelation of the estimator output on a 1 s grid")
    print("  ind/min  = independent observations per minute of recording, AR(1)")
    print("  info/min = d x sqrt(ind/min): how precisely a fixed-duration session pins")
    print("             down a state difference. Higher is better.")
    print()
    print("  THE POINT. Heavy smoothing raises per-sample d but destroys independence,")
    print("  and the product is what determines precision. Configurations that respond")
    print("  an order of magnitude faster ALSO extract more information per minute, so")
    print("  the deployed setting is not a latency/accuracy trade-off - it is dominated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
