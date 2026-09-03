"""
signal_quality.py - per-channel report: is this electrode measuring cortex?

WHY THIS EXISTS

contact_check.py answers "is the electrode attached", using amplitude and mains ratio.
That is necessary and not sufficient. An electrode can be attached, pass every amplitude
check, and still be dominated by ocular artefact rather than cortical activity - and the
arousal index would then be steering on eye movement.

This is not hypothetical here. In the alpha-validation session, AF7 passed enough windows
to be analysed and showed a SIGNIFICANT REVERSAL of the eyes-closed alpha effect
(0.83x, d = -0.32, p = 0.009). A pure noise channel gives d near zero; a significant
reversal means the channel was tracking something systematic and anti-correlated with
alpha. Eyes open means more blinking, and blinks put energy in the low band that leaks
upward. See docs/finding_channel_validation.md.

THE DIAGNOSTIC

Cortical alpha appears as a PEAK in the 8-13 Hz band, standing above the surrounding
spectrum. Ocular and movement artefact is low-frequency dominated and has no such peak.
So "is there an alpha peak" separates the two without needing an eyes-closed
manipulation, and can therefore be run on any recording, including one already
collected.

Three numbers per channel:

  alpha peak prominence - alpha power over a background formed by running-median
                          filtering the log spectrum. Above ~1.2 indicates a real peak.
  low-band dominance    - 2-8 Hz power over alpha power. Ocular contamination raises it.
  artefact rate         - fraction of samples beyond 100 uV, as a contact proxy.

VALIDATED AGAINST GROUND TRUTH before being trusted. On the alpha-validation session,
prominence rises with eyes closed on the channels known to carry the effect - TP9 1.10
to 2.08 (1.89x) and TP10 1.18 to 2.30 (1.95x) - and does not on AF7 (1.98 to 1.78,
0.90x), which is the channel independently shown to be contaminated.

An earlier version estimated the background by fitting 1/f between flanking bands and
had to be discarded: it needs a clean low-frequency flank, and low-frequency artefact is
exactly what contaminates these recordings. It scored TP9 as having no alpha peak, in a
session where TP9 shows a validated 1.71x eyes-closed increase at p = 6e-21. A
diagnostic that fails on a known case cannot be trusted on an unknown one.

Usage:
    python scripts/signal_quality.py
    python scripts/signal_quality.py --session sessions/PILOT01_20260822_153652
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

from eeg_features import MUSE_CHANNELS  # noqa: E402
from session_logger import load_raw, load_session  # noqa: E402

FS = 256.0
ALPHA = (8.0, 13.0)
# Background is estimated by median-filtering the log spectrum rather than by fitting
# 1/f between flanking bands. The flank approach was tried first and is unusable here:
# it needs a clean low-frequency flank, and low-frequency artefact is precisely what
# contaminates these recordings, so the contamination corrupts the baseline meant to
# detect it. It scored the alphatest's TP9 as having NO ALPHA PEAK when that channel
# shows a validated 1.71x eyes-closed increase at p = 6e-21.
BACKGROUND_MEDIAN_HZ = 8.0    # width of the running median, wide enough to smooth a peak

# Prominence is computed per short segment and the MEDIAN reported, not computed once
# over the whole recording. Alpha is intermittent - it comes and goes with drowsiness,
# eye closure and attention - so a single Welch across six minutes averages the peak
# away with the periods that lack it. Measured on the alpha-validation session, the
# whole-recording estimate scores TP9 at 0.99 (no peak) while the per-segment median
# separates eyes-open 1.10 from eyes-closed 2.08 on the same channel.
SEGMENT_S = 10.0


def alpha_prominence(x: np.ndarray) -> tuple[float, float]:
    """
    (peak prominence, low-band dominance) for one channel.

    Prominence is alpha-band power over a background formed by running-median filtering
    the log spectrum. A median filter wider than the peak passes the 1/f trend and
    removes the peak itself, so the ratio isolates local excess without ever fitting a
    line through contaminated low frequencies.
    """
    from scipy.ndimage import median_filter

    x = x - np.mean(x)
    f, p = sps.welch(x, fs=FS, nperseg=int(FS * 4), noverlap=int(FS * 2))
    ok = (f > 2) & (f < 45) & (p > 0)
    f, p = f[ok], p[ok]
    if f.size < 30:
        return float("nan"), float("nan")

    df = float(np.median(np.diff(f)))
    width = max(5, int(round(BACKGROUND_MEDIAN_HZ / df)) | 1)   # odd, >= 5 bins
    baseline = 10 ** median_filter(np.log10(p), size=width, mode="nearest")

    band = (f >= ALPHA[0]) & (f <= ALPHA[1])
    observed = float(np.mean(p[band]))
    background = float(np.mean(baseline[band]))
    prominence = observed / background if background > 0 else float("nan")

    low = (f >= 2) & (f < 8)
    dominance = float(np.mean(p[low]) / observed) if observed > 0 else float("nan")
    return prominence, dominance


# Reference values measured on the alpha-validation session, needed to read the numbers.
# Alpha depends strongly on eye closure, so an eyes-open recording legitimately shows a
# weak peak and that is NOT evidence of a bad channel.
#
#   TP9  eyes-open 1.10   eyes-closed 2.08
#   TP10 eyes-open 1.18   eyes-closed 2.30
#
# PILOT01, an eyes-open session throughout, reads 1.04-1.09 across all four channels -
# consistent with the eyes-open reference rather than with contamination.
PROM_EYES_OPEN_TYPICAL = 1.10
PROM_CLEAR_PEAK = 1.4


def verdict(prom: float, dom: float, art: float) -> str:
    """
    Interpretation, deliberately conservative about what it can conclude.

    A weak peak in an eyes-open recording is expected and says nothing bad. Only a
    channel whose peak is below the eyes-open reference AND whose low band dominates is
    flagged as suspect, and even then the tool cannot distinguish "eyes-open cortex"
    from "no cortex" without an eye-closure manipulation. That is what alpha_test.py is
    for, and why running it on the index channels is not optional.
    """
    if not np.isfinite(prom):
        return "no data"
    if art > 0.25:
        return "BAD CONTACT"
    if prom >= PROM_CLEAR_PEAK:
        return "clear alpha peak"
    if prom < PROM_EYES_OPEN_TYPICAL * 0.85 and dom > 10:
        return "SUSPECT: no peak, low-band dominated"
    return "consistent with eyes-open"


def main() -> int:
    p = argparse.ArgumentParser(description="Per-channel signal quality report")
    p.add_argument("--session", default=None)
    p.add_argument("--all", action="store_true", help="report every session on disk")
    args = p.parse_args()

    if args.all:
        dirs = [d for d in sorted(glob.glob(os.path.join(_ROOT, "sessions", "*")))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "raw_eeg.f32"))]
    elif args.session:
        dirs = [args.session]
    else:
        cands = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*")))
        dirs = [cands[-1]] if cands else []
    if not dirs:
        print("No session with a raw recording found.")
        return 1

    print("=" * 78)
    print("SIGNAL QUALITY - is each electrode measuring cortex, or something else?")
    print("=" * 78)
    print("  prominence = alpha power / interpolated 1/f background (>1.2 = real peak)")
    print("  low/alpha  = delta+theta over alpha (high = ocular or movement)")
    print("  artefact   = fraction of samples beyond 100 uV")
    print()

    for d in dirs:
        try:
            session = load_session(d)
            raw = load_raw(d)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {os.path.basename(d)}: unreadable ({type(exc).__name__})")
            continue
        ch = raw[:, 1:]
        keep = np.isfinite(ch).all(axis=1)
        ch = ch[keep]
        if ch.shape[0] < FS * 30:
            print(f"  {os.path.basename(d)}: too short")
            continue

        index_pair = session["manifest"].get("index_channels") or []
        print(f"  {os.path.basename(d)}   index channels: {'/'.join(index_pair) or 'unknown'}")
        print(f"    {'chan':<7}{'sd uV':>8}{'artefact':>10}{'prominence':>12}"
              f"{'low/alpha':>11}   verdict")
        print("    " + "-" * 66)
        step = int(SEGMENT_S * FS)
        for i, name in enumerate(MUSE_CHANNELS):
            x = ch[:, i].astype(float)
            x = x - np.median(x)
            art = float(np.mean(np.abs(x) > 100))
            proms, doms = [], []
            for s0 in range(0, x.size - step + 1, step):
                pr, dm = alpha_prominence(x[s0:s0 + step])
                if np.isfinite(pr):
                    proms.append(pr)
                    doms.append(dm)
            prom = float(np.median(proms)) if proms else float("nan")
            dom = float(np.median(doms)) if doms else float("nan")
            used = "*" if name in index_pair else " "
            print(f"    {used}{name:<6}{x.std():>8.1f}{art * 100:>9.1f}%{prom:>12.2f}"
                  f"{dom:>11.1f}   {verdict(prom, dom, art)}")
        print("    * = used by this session's arousal index")
        print()

    print("  HOW TO READ THIS. Alpha depends strongly on eye closure, so an eyes-open")
    print("  recording legitimately shows a weak peak. Reference, measured on the")
    print("  alpha-validation session: TP9 1.10 eyes-open against 2.08 eyes-closed.")
    print()
    print("  So 'consistent with eyes-open' means the channel looks normal for a session")
    print("  recorded with eyes open - NOT that it is confirmed cortical. This tool")
    print("  cannot separate eyes-open cortex from no cortex without an eye-closure")
    print("  manipulation. Only scripts/alpha_test.py can, and it has to be run ON THE")
    print("  INDEX CHANNELS - see docs/finding_channel_validation.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
