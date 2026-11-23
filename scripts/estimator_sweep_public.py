"""
estimator_sweep_public.py - the estimator comparison of Section 4, on public and synthetic data.

Same question as scripts/estimator_sweep.py - is the deployed 5.5 s analysis latency a
floor or a configuration? - answered without any recording of the author. Each session
(see scripts/public_data.py) carries exact labelled eyes-open/eyes-closed transitions.

Every configuration is scored per session, then summarised across sessions by the median,
with the interquartile range as spread:

  detect    seconds from a labelled transition to a midpoint crossing that HOLDS for
            --hold seconds (default 0.5). A crossing undone sooner is noise, not
            detection; counting it would flatter the noisiest estimators. The first-
            crossing rule (hold 0) is also recorded, as a sensitivity check.
  d         Cohen's d, eyes closed vs open, at least 8 s from any transition
  ind/min   60 (1 - rho) / (1 + rho), rho the lag-1 autocorrelation on a 1 s grid
  R         d x sqrt(ind/min)

The windowed estimators call the deployed FeatureExtractor; nothing is reimplemented.

Usage:
    python scripts/estimator_sweep_public.py --source eegbci --pair AF7 AF8 --out results/public/eegbci_af.json
    python scripts/estimator_sweep_public.py --source eegbci --pair TP9 TP10 --out results/public/eegbci_tp.json
    python scripts/estimator_sweep_public.py --source synthetic --ratio 1.34 --seeds 50 --out results/public/synth_134.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import signal as sps

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402
from public_data import EC, EO, sessions, site  # noqa: E402

ALPHA_BAND = (8.0, 13.0)
EDGE_GUARD_S = 8.0
HOLD_S = 0.5

WINDOWED = [
    ("deployed", 4.0, 1.0, 3.0),
    ("windowed 4/1/1.0", 4.0, 1.0, 1.0),
    ("windowed 2/0.5/1.0", 2.0, 0.5, 1.0),
    ("windowed 2/0.5/0.5", 2.0, 0.5, 0.5),
]
STREAMING = [(f"streaming o{o} t={t:g}", o, t) for o, t in
             [(4, 2.0), (4, 1.0), (4, 0.5), (4, 0.25), (2, 0.5), (2, 0.25)]]
FRONTIER = [(f"streaming o{o} t={t:g}", o, t) for o, t in
            [(4, 0.10), (4, 0.05), (2, 0.10), (2, 0.05)]]


# ------------------------------------------------------------------ estimators


def windowed(sess, pair, window_s, hop_s, tau_s):
    fs = sess["fs"]
    cfg = FeatureConfig(sampling_rate=fs, window_seconds=window_s, hop_seconds=hop_s,
                        channels=sess["channels"], frontal_channels=pair)
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    chans = sess["chans"]
    t, v = [], []
    rejected = total = 0
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        total += 1
        f = ex.extract(chans[:, s0:s0 + nw])
        if f.valid and np.isfinite(f.alpha) and f.alpha > 0:
            t.append((s0 + nw) / fs)
            v.append(f.alpha)
        else:
            rejected += 1
    if len(v) < 10:
        return np.array([]), np.array([]), rejected / max(total, 1)
    v = np.asarray(v)
    a = 1.0 - np.exp(-hop_s / tau_s)
    sm = np.empty_like(v)
    acc = v[0]
    for i, x in enumerate(v):
        acc += a * (x - acc)
        sm[i] = acc
    return np.asarray(t), sm, rejected / total


def streaming(sess, pair, order, tau_s, out_hz=16.0):
    """Causal per-sample estimator, primed from the first sample like StreamingBandPower."""
    fs = sess["fs"]
    idx = [sess["channels"].index(p) for p in pair]
    x = sess["chans"][idx, :].mean(axis=0).astype(float)
    hp = sps.butter(2, 1.0, btype="highpass", fs=fs, output="sos")
    x = sps.sosfilt(hp, x, zi=sps.sosfilt_zi(hp) * x[0])[0]
    b, a = sps.iirnotch(60.0, 30.0, fs=fs)
    x = sps.lfilter(b, a, x, zi=sps.lfilter_zi(b, a) * x[0])[0]
    bp = sps.butter(order, ALPHA_BAND, btype="bandpass", fs=fs, output="sos")
    power = sps.sosfilt(bp, x) ** 2
    k = 1.0 - np.exp(-1.0 / (tau_s * fs))
    sm = sps.lfilter([k], [1.0, -(1.0 - k)], power, zi=[(1 - k) * power[0]])[0]
    step = max(1, int(round(fs / out_hz)))
    t = (np.arange(x.size) + 1) / fs
    return t[::step], sm[::step]


# --------------------------------------------------------------------- scoring


def phase_at(timeline, t):
    prev = None
    for tt, ph in timeline:
        if tt > t:
            return prev
        prev = ph
    return None


def transitions(timeline):
    return [(timeline[i][0], timeline[i - 1][1], timeline[i][1])
            for i in range(1, len(timeline))
            if timeline[i][1] in (EO, EC) and timeline[i - 1][1] in (EO, EC)
            and timeline[i][1] != timeline[i - 1][1]]


def detect(st, crossed, hold):
    """First crossing not undone within `hold` seconds (zero-order hold between samples)."""
    for k in np.flatnonzero(crossed):
        later = np.flatnonzero(~crossed[k + 1:])
        if not later.size or st[k + 1 + later[0]] - st[k] >= hold:
            return st[k]
    return None


def score(t, y, timeline, holds=(HOLD_S, 0.0)):
    nan = float("nan")
    empty = {"d": nan, "rho": nan, "ind_min": nan, "R": nan,
             **{f"detect_h{h:g}": nan for h in holds}}
    if t.size < 10:
        return empty
    y = np.log10(np.maximum(y, 1e-12))
    trans = transitions(timeline)
    labels = np.array([phase_at(timeline, tt) or "" for tt in t])
    near = np.zeros(t.size, dtype=bool)
    for tt, _, _ in trans:
        near |= np.abs(t - tt) < EDGE_GUARD_S
    op, cl = (labels == EO) & ~near, (labels == EC) & ~near
    if op.sum() < 5 or cl.sum() < 5:
        return empty
    pooled = np.sqrt((y[op].var(ddof=1) + y[cl].var(ddof=1)) / 2)
    d = float((y[cl].mean() - y[op].mean()) / pooled) if pooled > 0 else nan

    out = {}
    for h in holds:
        lats = []
        for tt, frm, to in trans:
            pre = y[(t > tt - 40) & (t < tt - EDGE_GUARD_S) & (labels == frm)]
            post = y[(t > tt + EDGE_GUARD_S) & (t < tt + 55) & (labels == to)]
            if pre.size < 3 or post.size < 3:
                continue
            mid = (pre.mean() + post.mean()) / 2.0
            # Strictly after: a window ending exactly at tt contains no post-transition
            # data, so a crossing there is chance, not detection.
            seg = (t > tt) & (labels == to)
            st, sy = t[seg], y[seg]
            if not st.size:
                continue
            crossed = sy > mid if post.mean() > pre.mean() else sy < mid
            hit = detect(st, crossed, h)
            if hit is not None:
                lats.append(hit - tt)
        out[f"detect_h{h:g}"] = float(np.median(lats)) if lats else nan

    grid = np.arange(t[0], t[-1], 1.0)
    on_grid = np.interp(grid, t, y)
    inside = np.array([phase_at(timeline, g) in (EO, EC) for g in grid])
    seg = on_grid[inside]
    rho = float(np.corrcoef(seg[:-1], seg[1:])[0, 1]) if seg.size > 30 else nan
    ind = 60.0 * (1 - rho) / (1 + rho) if np.isfinite(rho) else nan
    R = d * np.sqrt(ind) if np.isfinite(d) and np.isfinite(ind) and ind > 0 else nan
    return {"d": d, "rho": rho, "ind_min": ind, "R": float(R), **out}


# -------------------------------------------------------------------- summary


def summarise(rows, key):
    v = np.asarray([r[key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"median": None, "q1": None, "q3": None, "n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": int(v.size)}


def run(source, pair_names, seeds, ratio, hold, subjects=None):
    per = {}
    rejection = []
    names = []
    for sess in sessions(source, seeds=seeds, ratio=ratio, subjects=subjects):
        pair = tuple(site(p, source) for p in pair_names)
        names.append(sess["name"])
        for name, w, h, tau in WINDOWED:
            t, y, rej = windowed(sess, pair, w, h, tau)
            if name == "deployed":
                rejection.append(rej)
            per.setdefault(name, []).append(
                {"session": sess["name"], **score(t, y, sess["timeline"], (hold, 0.0))})
        for name, order, tau in STREAMING + FRONTIER:
            t, y = streaming(sess, pair, order, tau)
            per.setdefault(name, []).append(
                {"session": sess["name"], **score(t, y, sess["timeline"], (hold, 0.0))})
        print(f"  {sess['name']}", end="\r", flush=True)
    print()
    dk = f"detect_h{hold:g}"
    summary = {}
    for name, rows in per.items():
        summary[name] = {k: summarise(rows, k) for k in (dk, "detect_h0", "d", "rho",
                                                          "ind_min", "R")}
    base = per["deployed"]
    alternatives = [n for n, *_ in WINDOWED[1:]] + [n for n, *_ in STREAMING]
    dom = {}
    for name in alternatives + [n for n, *_ in FRONTIER]:
        wins = [a[dk] < b[dk] and a["R"] > b["R"]
                for a, b in zip(per[name], base)
                if all(np.isfinite([a[dk], b[dk], a["R"], b["R"]]))]
        dom[name] = {"fraction_of_sessions": float(np.mean(wins)) if wins else None,
                     "n": len(wins)}
    b = summary["deployed"]
    dominating = [n for n in alternatives
                  if summary[n][dk]["median"] is not None
                  and summary[n][dk]["median"] < b[dk]["median"]
                  and summary[n]["R"]["median"] > b["R"]["median"]]
    return {
        "source": source, "pair": list(pair_names), "hold_s": hold,
        "ratio": ratio if source == "synthetic" else None,
        "n_sessions": len(names), "sessions": names,
        "deployed_rejection": summarise([{"x": r} for r in rejection], "x"),
        "alternatives": alternatives, "frontier": [n for n, *_ in FRONTIER],
        "dominating_medians": dominating,
        "per_session_dominance": dom,
        "summary": summary, "per_session": per,
    }


def print_table(res):
    dk = f"detect_h{res['hold_s']:g}"
    s = res["summary"]
    print(f"\n  {res['source']} {'/'.join(res['pair'])}   n = {res['n_sessions']} sessions"
          f"   hold {res['hold_s']:g} s")
    print(f"  {'estimator':<22}{'detect':>8}{'1st-x':>8}{'d':>7}{'rho':>7}{'ind/min':>9}"
          f"{'R':>7}{'dom%':>7}")
    print("  " + "-" * 75)
    base_R = s["deployed"]["R"]["median"]
    for name in ["deployed"] + res["alternatives"] + res["frontier"]:
        m = {k: (v["median"] if v["median"] is not None else float("nan"))
             for k, v in s[name].items()}
        dom = res["per_session_dominance"].get(name, {}).get("fraction_of_sessions")
        dom = f"{dom * 100:.0f}" if dom is not None else "-"
        if name == res["frontier"][0]:
            print("  " + "-" * 75)
        print(f"  {name:<22}{m[dk]:>7.2f}s{m['detect_h0']:>7.2f}s{m['d']:>7.2f}"
              f"{m['rho']:>7.3f}{m['ind_min']:>9.1f}{m['R']:>7.2f}{dom:>7}")
    print(f"\n  alternatives beating deployed on both medians: "
          f"{len(res['dominating_medians'])} of {len(res['alternatives'])}")
    print(f"  deployed R = {base_R:.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--source", choices=["eegbci", "synthetic"], required=True)
    p.add_argument("--pair", nargs=2, default=["AF7", "AF8"])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--ratio", type=float, default=1.34)
    p.add_argument("--hold", type=float, default=HOLD_S)
    p.add_argument("--subjects", type=int, nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = run(args.source, args.pair, args.seeds, args.ratio, args.hold, args.subjects)
    print_table(res)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
