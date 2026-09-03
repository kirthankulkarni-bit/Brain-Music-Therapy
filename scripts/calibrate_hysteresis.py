"""
calibrate_hysteresis.py - derive the trend thresholds for a given estimator, from data.

WHY THIS IS NEEDED BEFORE CHANGING THE ESTIMATOR

The trend suffix in build_prompt is gated by two thresholds. They are not arbitrary: they
were derived from the measured noise of the DEPLOYED estimator, after PILOT01 showed that
the original single threshold of 0.05 sat five times BELOW its own noise floor and was
therefore thresholding noise - 491 prompt changes in 20 minutes, 30% of them faster than a
crossfade could resolve.

Any change to the estimator changes that noise. A faster estimator smooths less, so its
trend estimate is noisier in absolute terms, and thresholds calibrated for tau = 3 s would
chatter again at tau = 0.25 s. The failure would look new and have the same cause.

So this script does what was previously done by hand, and does it for whatever estimator
configuration is proposed:

  1. reconstruct z from a real recording under the given estimator
  2. compute the trend exactly as the control loop does
  3. measure the trend estimator's own noise
  4. set ENTER at a multiple of that noise, EXIT below it

The multiples are the policy choice and are exposed as flags. The measurement is not.

Usage:
    python scripts/calibrate_hysteresis.py                          # current deployed config
    python scripts/calibrate_hysteresis.py --streaming --tau 0.25   # proposed alternative
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from eeg_features import (  # noqa: E402
    FeatureConfig,
    FeatureExtractor,
    StreamingBandPower,
    MUSE_CHANNELS,
)
from session_logger import load_raw, load_session  # noqa: E402

FS = 256.0


def series_windowed(chans, pair, window_s, hop_s, tau_s):
    """log(beta/alpha) on the deployed path, at arbitrary window/hop/tau."""
    cfg = FeatureConfig(sampling_rate=FS, window_seconds=window_s, hop_seconds=hop_s,
                        frontal_channels=pair)
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    vals = []
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        f = ex.extract(chans[:, s0:s0 + nw])
        if f.valid and np.isfinite(f.log_beta_alpha):
            vals.append(f.log_beta_alpha)
    v = np.asarray(vals, dtype=float)
    if v.size < 10:
        return np.array([]), hop_s
    a = 1.0 - np.exp(-hop_s / tau_s)
    out = np.empty_like(v)
    acc = v[0]
    for i, x in enumerate(v):
        acc += a * (x - acc)
        out[i] = acc
    return out, hop_s


def series_streaming(chans, pair, tau_s, order, hop_s=1.0):
    """
    log(beta/alpha) from two streaming estimators, sampled onto the control-loop hop.

    The control loop consumes one value per hop regardless of how the estimate is
    produced, so the trend must be measured on that grid for the thresholds to transfer.
    """
    cfg = FeatureConfig(sampling_rate=FS)
    idx = [MUSE_CHANNELS.index(p) for p in pair]
    x = chans[idx, :].mean(axis=0)

    est = {b: StreamingBandPower(cfg, band=b, tau_seconds=tau_s, order=order)
           for b in ("alpha", "beta")}
    step = int(hop_s * FS)
    out = []
    for s0 in range(0, x.size - step + 1, step):
        chunk = x[s0:s0 + step]
        a = est["alpha"].push(chunk)
        b = est["beta"].push(chunk)
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            out.append(np.log10(b / a))
    return np.asarray(out, dtype=float), hop_s


def calibrate(z: np.ndarray, window_hops: int, enter_k: float, exit_k: float) -> dict:
    """Trend noise and the thresholds it implies."""
    if z.size < window_hops + 20:
        return {}
    idx = np.arange(window_hops, dtype=float)
    slopes = np.array([np.polyfit(idx, z[i - window_hops:i], 1)[0]
                       for i in range(window_hops, z.size)])
    sd = float(slopes.std(ddof=1))
    return {
        "n": int(z.size),
        "z_sd": float(z.std(ddof=1)),
        "trend_sd": sd,
        "enter": enter_k * sd,
        "exit": exit_k * sd,
        "observed_max_abs": float(np.abs(slopes).max()),
        "fires_at_enter": int((np.abs(slopes) >= enter_k * sd).sum()),
        "frac_fires": float((np.abs(slopes) >= enter_k * sd).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Derive trend hysteresis thresholds from data")
    p.add_argument("--session", default=None)
    p.add_argument("--streaming", action="store_true", help="calibrate the streaming estimator")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--hop", type=float, default=1.0)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--trend-hops", type=int, default=20,
                   help="hops in the least-squares trend window, matching _TREND_WINDOW_HOPS")
    p.add_argument("--enter-k", type=float, default=5.0, help="ENTER as a multiple of trend sd")
    p.add_argument("--exit-k", type=float, default=2.5, help="EXIT as a multiple of trend sd")
    args = p.parse_args()

    d = args.session
    if d is None:
        cands = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*"))) or \
                sorted(glob.glob(os.path.join(_ROOT, "sessions", "alphatest*")))
        if not cands:
            print("No session found; pass --session.")
            return 1
        d = cands[-1]

    session = load_session(d)
    raw = load_raw(d)
    chans = raw[:, 1:].T.astype(float)
    if not np.all(np.isfinite(chans)):
        n_bad = int((~np.isfinite(chans)).any(axis=0).sum())
        chans = chans[:, np.isfinite(chans).all(axis=0)]
        print(f"  note: dropped {n_bad} samples containing non-finite values")
    pair = tuple(session["manifest"].get("index_channels") or ("AF7", "AF8"))

    tau = args.tau if args.tau is not None else (0.25 if args.streaming else 3.0)

    # build_prompt operates on z, which is log(beta/alpha) normalised by the
    # participant's own BASELINE SD - so thresholds must be expressed in z units.
    # The estimators below produce raw log units, and reporting those directly would
    # hand over a threshold roughly 8x too small. Caught by noticing that a recalibration
    # of the deployed configuration disagreed with the value already in music_engine.py.
    baseline = (session.get("baseline") or [{}])[0]
    baseline_sd = baseline.get("baseline_sd_log_beta_alpha")
    if not baseline_sd or not np.isfinite(baseline_sd) or baseline_sd <= 0:
        print("  This session has no usable baseline SD, so z units cannot be recovered.")
        print("  Thresholds would be in raw log units and are NOT comparable to")
        print("  _TREND_ENTER / _TREND_EXIT. Use a session with a completed baseline.")
        return 1

    print("=" * 74)
    print("HYSTERESIS CALIBRATION")
    print("=" * 74)
    print(f"  session   : {os.path.basename(d)}")
    print(f"  channels  : {'/'.join(pair)}")
    print(f"  estimator : {'streaming' if args.streaming else 'windowed'}, tau={tau:g}"
          + (f", order={args.order}" if args.streaming else
             f", window={args.window:g}, hop={args.hop:g}"))
    print()

    if args.streaming:
        z, hop = series_streaming(chans, pair, tau, args.order, args.hop)
    else:
        z, hop = series_windowed(chans, pair, args.window, args.hop, tau)
    z = z / float(baseline_sd)                       # into z units

    if z.size < 40:
        print(f"  only {z.size} usable values - not enough to calibrate.")
        return 1

    r = calibrate(z, args.trend_hops, args.enter_k, args.exit_k)
    print(f"  baseline SD (1 z)      : {baseline_sd:.4f} log units")
    print(f"  usable values          : {r['n']}")
    print(f"  sd of the index        : {r['z_sd']:.4f}")
    print(f"  sd of the trend        : {r['trend_sd']:.4f}   <- the noise being thresholded")
    print(f"  largest observed slope : {r['observed_max_abs']:.4f}")
    print()
    print(f"  ENTER = {args.enter_k:g} x trend sd = {r['enter']:.4f}")
    print(f"  EXIT  = {args.exit_k:g} x trend sd = {r['exit']:.4f}")
    print()
    print(f"  windows that would fire: {r['fires_at_enter']} of {r['n']} "
          f"({r['frac_fires']:.1%})")
    if r["enter"] > r["observed_max_abs"]:
        print("  ENTER sits above every slope in this recording, so the suffix cannot fire.")
        print("  That is the intended outcome when the trend is unmeasurable at this SNR -")
        print("  it fires only for excursions genuinely outside anything observed.")
    print()
    from music_engine import _TREND_ENTER, _TREND_EXIT
    print(f"  currently in music_engine.py: ENTER {_TREND_ENTER:g}, EXIT {_TREND_EXIT:g}")
    print()
    print("  Put these in music_engine.py as _TREND_ENTER / _TREND_EXIT if adopting this")
    print("  estimator, and rerun scripts/run_tests.py - the chatter regression test will")
    print("  tell you whether the controller still behaves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
