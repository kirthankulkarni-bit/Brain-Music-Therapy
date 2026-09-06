"""
alpha_sensitivity.py - is the alpha result an artifact of artifact rejection?

WHY THIS EXISTS

The first real alpha validation (2026-08-16) passed at 2.42x, but the rejection
rate was wildly asymmetric between conditions:

    eyes open    159 windows, 88 rejected  (55.3%)
    eyes closed  159 windows, 16 rejected  (10.1%)

All of them amplitude artifacts, which is exactly what you would predict: you blink
with your eyes open and not with them closed.

That is a differential-rejection confound. The surviving eyes-open windows are a
biased subsample of the eyes-open condition, so the contrast is no longer between
"eyes open" and "eyes closed" but between "eyes open, excluding the half of the
time I was blinking" and "eyes closed". If blink-contaminated windows carry inflated
broadband power through spectral leakage, discarding them lowers the eyes-open alpha
estimate and inflates the ratio.

This script reprocesses the saved raw recording at several rejection thresholds and
reports how the effect size moves. Reporting that curve is far more defensible than
reporting a single number and hoping no one asks.

Usage:
    python scripts/alpha_sensitivity.py                    # most recent alpha test
    python scripts/alpha_sensitivity.py sessions/alphatest_20260816_015511
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eeg_features import MUSE_CHANNELS, FeatureConfig, FeatureExtractor  # noqa: E402
from session_logger import load_raw, load_session  # noqa: E402

# The deployed value is read from FeatureConfig rather than written here, and inserted
# into the sweep, so the table always contains the setting the pipeline actually uses.
# It did not: this list was fixed at 100/150/200/300/450/600 and the code below called
# row[1] "the pipeline default". That was true when the default was 150 uV. It became
# 350 uV in response to THIS script's own finding, and nothing updated the script - so
# it kept reporting a threshold the pipeline had abandoned as the pipeline's, and the
# deployed value appeared nowhere in the sensitivity analysis.
def _thresholds() -> list:
    from eeg_features import FeatureConfig
    base = [100.0, 150.0, 200.0, 300.0, 450.0, 600.0]
    deployed = float(FeatureConfig(sampling_rate=256.0).reject_peak_to_peak_uv)
    if deployed not in base:
        base.append(deployed)
    return sorted(base) + [float("inf")]


THRESHOLDS = _thresholds()


def analyze(session_dir: str, quiet: bool = False):
    """
    Sweep the rejection thresholds. Returns the rows, or None if the session cannot be
    swept. `quiet` suppresses the report so verify_claims can reuse the computation
    rather than reimplement it - the numbers it asserts must come from this code path,
    not from a second copy of the arithmetic that could drift away from it.
    """
    session = load_session(session_dir)
    manifest = session["manifest"]

    required = ("sampling_rate", "block_seconds", "settle_seconds", "index_channels")
    missing = [k for k in required if k not in manifest]
    if missing:
        if not quiet:
            print(f"Session manifest is missing {missing}. Is this an alpha_test session?")
        return None

    sr = float(manifest["sampling_rate"])
    block_s = float(manifest["block_seconds"])
    settle_s = float(manifest["settle_seconds"])
    channels = tuple(manifest["index_channels"])

    raw = load_raw(session_dir, n_channels=len(MUSE_CHANNELS))
    if raw.shape[0] < sr * 30:
        print("Raw recording too short to reanalyze.")
        return 1

    elapsed = raw[:, 0].astype(np.float64)
    elapsed = elapsed - elapsed[0]

    window_n = int(4.0 * sr)
    hop_n = int(1.0 * sr)

    if not quiet:
        print("=" * 78)
        print(f"ALPHA SENSITIVITY TO ARTIFACT REJECTION  ({os.path.basename(session_dir)})")
        print("=" * 78)
        print(f"  channels {'+'.join(channels)}   {raw.shape[0]} samples   {elapsed[-1]:.0f} s   "
              f"{block_s:.0f} s blocks")
        print()
        print(f"  {'reject p2p >':>14} | {'n open':>7} | {'n closed':>8} | {'imbalance':>9} | "
              f"{'ratio':>7} | {'p':>10}")
        print("  " + "-" * 68)

    rows = []
    for threshold in THRESHOLDS:
        cfg = FeatureConfig(
            sampling_rate=sr,
            window_seconds=4.0,
            hop_seconds=1.0,
            frontal_channels=channels,
            reject_peak_to_peak_uv=threshold,
        )
        extractor = FeatureExtractor(cfg)

        opened, closed = [], []
        for start in range(0, raw.shape[0] - window_n, hop_n):
            t0 = elapsed[start]
            t1 = elapsed[start + window_n - 1]
            b0, b1 = int(t0 // block_s), int(t1 // block_s)
            if b0 != b1 or (t0 - b0 * block_s) < settle_s:
                continue  # straddles a transition, or inside the settle period
            feats = extractor.extract(raw[start:start + window_n, 1:].T)
            if not feats.valid:
                continue
            (closed if b0 % 2 == 1 else opened).append(feats.alpha)

        opened, closed = np.asarray(opened), np.asarray(closed)
        if opened.size < 5 or closed.size < 5:
            continue

        try:
            from scipy.stats import ttest_ind
            _, p = ttest_ind(np.log10(closed), np.log10(opened), equal_var=False)
        except ImportError:
            p = float("nan")

        ratio = float(closed.mean() / opened.mean())
        # How unbalanced the surviving samples are. 1.00 means rejection hit both
        # conditions equally, which is the assumption a naive comparison makes.
        imbalance = float(closed.size / opened.size)
        label = "none" if not np.isfinite(threshold) else f"{threshold:.0f} uV"
        rows.append((label, opened.size, closed.size, imbalance, ratio, p))
        if not quiet:
            print(f"  {label:>14} | {opened.size:>7} | {closed.size:>8} | "
                  f"{imbalance:>9.2f} | {ratio:>6.2f}x | {p:>10.2e}")

    if not rows:
        if not quiet:
            print("  No threshold produced enough usable windows.")
        return None

    if quiet:
        return rows

        print("\n" + "-" * 78)
        deployed_uv = float(FeatureConfig(sampling_rate=256.0).reject_peak_to_peak_uv)
        deployed = next((r for r in rows if r[0].startswith(f"{deployed_uv:.0f}")), None)
        strict = rows[1]       # the 150 uV row, kept because the alphatest was RECORDED at it
        permissive = rows[-1]  # no rejection
        # "Most balanced" must be an actual setting you could run, so exclude the
        # no-rejection row - that one is the lower bound of the sensitivity analysis,
        # not a candidate configuration.
        finite_rows = [r for r in rows if r[0] != "none"]
        balanced = min(finite_rows, key=lambda r: abs(r[3] - 1.0)) if finite_rows else rows[0]

        if deployed is not None:
            print(f"  pipeline default ({deployed_uv:.0f} uV): {deployed[4]:.2f}x   "
                  f"imbalance {deployed[3]:.2f}")
        print(f"  AS THIS SESSION WAS RECORDED (150 uV): {strict[4]:.2f}x   "
              f"imbalance {strict[3]:.2f}")
        print(f"  no rejection at all       : {permissive[4]:.2f}x   imbalance {permissive[3]:.2f}")
        print(f"  most balanced ({balanced[0]:>7})   : {balanced[4]:.2f}x   imbalance {balanced[3]:.2f}")
        print()
        print("  HOW TO READ THIS")
        print("  The two extremes bound the true effect from opposite directions.")
        print("  Strict rejection discards eyes-open windows preferentially, removing")
        print("  blink power from the open condition and inflating the ratio. No rejection")
        print("  leaves blink energy in the open condition, adding broadband power through")
        print("  spectral leakage and deflating it. The balanced threshold - where both")
        print("  conditions retain a similar number of windows - is the defensible number")
        print("  to report, with this whole table as the sensitivity analysis.")

        if all(np.isfinite(r[5]) and r[5] < 0.001 for r in rows):
            print()
            print("  The effect is significant at EVERY threshold including no rejection,")
            print("  so it is not manufactured by the rejection rule. Only its magnitude is")
            print("  threshold-dependent.")

        print()
        print("  IMPLICATION FOR THE MAIN STUDY")
        print(f"  The 150 uV threshold rejected {100 * (1 - strict[1] / permissive[1]):.0f}% of eyes-open windows here,")
        print(f"  against {100 * (1 - strict[2] / permissive[2]):.0f}% of eyes-closed. That differential is what inflates")
        print("  the ratio: it removes blink power from the open condition only.")
        print()
        print(f"  THIS WAS ACTED ON. FeatureConfig now rejects at {deployed_uv:.0f} uV and the value is")
        print("  fixed in the frozen pre-registration, so it cannot be tuned per session.")
        print()
        print("  BUT THE ALPHA VALIDATION SESSION WAS RECORDED AT 150 uV, before the change.")
        print("  Its headline ratio therefore carries the imbalance above. Report it with this")
        print("  table, not alone - the effect is real at every threshold, but the magnitude")
        print("  quoted depends on a rejection rule the pipeline no longer uses.")
        print("=" * 78)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpha result sensitivity to artifact rejection")
    parser.add_argument("session", nargs="?", default=None)
    args = parser.parse_args()

    session_dir = args.session
    if not session_dir:
        candidates = sorted(glob.glob(os.path.join(ROOT, "sessions", "alphatest_*")))
        if not candidates:
            print("No alpha test sessions found. Run scripts/alpha_test.py first.")
            return 1
        session_dir = candidates[-1]

    return 0 if analyze(session_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------- reusable by verify_claims


def _rows(session_dir=None):
    from session_logger import real_sessions

    if session_dir is None:
        found = real_sessions(os.path.join(ROOT, "sessions", "alphatest*"))
        if not found:
            raise FileNotFoundError("no alpha-validation session on disk")
        session_dir = found[-1]
    rows = analyze(session_dir, quiet=True)
    if not rows:
        raise RuntimeError(f"threshold sweep produced nothing for {session_dir}")
    return rows


def ratio_at_threshold(threshold_uv: float, session_dir=None):
    """(eyes-closed alpha ratio, description) at one rejection threshold."""
    label = f"{threshold_uv:.0f} uV"
    for row in _rows(session_dir):
        if row[0] == label:
            return float(row[4]), (f"{row[1]} open / {row[2]} closed, "
                                   f"imbalance {row[3]:.2f}, p {row[5]:.1e}")
    raise KeyError(f"{label} is not in the sweep; THRESHOLDS = {THRESHOLDS}")


def significance_across_thresholds(alpha: float = 0.001, session_dir=None):
    """
    (fraction of thresholds where the effect is significant, description).

    Must be 1.0. The point of the sweep is that the effect is not an artefact of the
    rejection rule, so a threshold at which it disappears would matter far more than any
    single ratio moving.
    """
    rows = _rows(session_dir)
    ok = sum(1 for r in rows if np.isfinite(r[5]) and r[5] < alpha)
    worst = max((r[5] for r in rows if np.isfinite(r[5])), default=float("nan"))
    return ok / len(rows), f"{ok}/{len(rows)} thresholds, worst p {worst:.1e}"
