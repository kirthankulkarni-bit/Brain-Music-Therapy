"""
verify_sample_rate.py - settle the sampling rate empirically. RUN THIS FIRST.

The old live path declared sfreq = 128 with a comment calling it a Muse 2 hardware
spec. The Muse 2 streams EEG at 256 Hz. At 128 Hz every frequency on the axis is
halved, so the 8-13 Hz alpha mask actually read 16-26 Hz, the 13-30 Hz beta mask
actually read 26-60 Hz, and 60 Hz mains hum on dry electrodes landed exactly on
the inclusive top edge of the beta mask - making power line noise the dominant
contributor to "beta power".

This script does not take the stream's own metadata on faith. It checks three
independent things:

  1. The nominal rate the stream advertises.
  2. The empirical rate derived from LSL timestamps (samples / elapsed time).
  3. The location of the mains peak, which is the ground truth. 60 Hz (or 50 Hz)
     is the one frequency whose true value you already know, so whichever assumed
     sampling rate puts the mains peak where it belongs is the correct one.

Usage:
    python scripts/verify_sample_rate.py
    python scripts/verify_sample_rate.py --seconds 60 --mains 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from stream_utils import get_inlet  # noqa: E402


def mains_peak(signal: np.ndarray, assumed_rate: float, mains: float) -> tuple[float, float]:
    """Return (peak frequency near mains, sharpness) under an assumed sampling rate."""
    nperseg = int(min(signal.size, assumed_rate * 4))
    freqs, psd = welch(signal, fs=assumed_rate, nperseg=nperseg, detrend="linear")
    search = (freqs > mains * 0.6) & (freqs < min(mains * 1.4, assumed_rate / 2 * 0.98))
    if search.sum() < 3:
        return float("nan"), float("nan")
    peak_f = float(freqs[search][np.argmax(psd[search])])
    baseline = float(np.median(psd[search]))
    sharpness = float(np.max(psd[search]) / baseline) if baseline > 0 else float("nan")
    return peak_f, sharpness


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the true EEG sampling rate")
    parser.add_argument("--seconds", type=float, default=30.0, help="capture duration")
    parser.add_argument("--mains", type=float, default=60.0, help="power line frequency (60 NA, 50 EU)")
    parser.add_argument("--channel", type=int, default=1, help="channel index (1=AF7)")
    args = parser.parse_args()

    inlet, nominal_rate = get_inlet()
    if inlet is None:
        print("\nNo stream. Start BlueMuse, connect the headset, hit 'Start Streaming'.")
        return 1

    print(f"\nCapturing {args.seconds:.0f} s. Sit still, eyes open, jaw relaxed.")

    samples: list[list[float]] = []
    timestamps: list[float] = []
    t_start = time.time()
    while time.time() - t_start < args.seconds:
        chunk, ts = inlet.pull_chunk(timeout=0.5, max_samples=256)
        if chunk:
            samples.extend(chunk)
            timestamps.extend(ts)
        elapsed = time.time() - t_start
        print(f"  {elapsed:5.1f}s  {len(samples):6d} samples", end="\r")

    print()
    if len(samples) < 100:
        print("Captured almost nothing. Is the headset actually connected in BlueMuse?")
        return 1

    ts = np.asarray(timestamps, dtype=np.float64)
    span = ts[-1] - ts[0]
    empirical_rate = (len(ts) - 1) / span if span > 0 else float("nan")
    median_dt = float(np.median(np.diff(ts)))

    signal = np.asarray(samples, dtype=np.float64)[:, args.channel]

    print("\n" + "=" * 68)
    print("SAMPLING RATE VERIFICATION")
    print("=" * 68)
    print(f"  samples captured        : {len(ts)}")
    print(f"  timestamp span          : {span:.3f} s")
    print(f"  nominal rate (metadata) : {nominal_rate:g} Hz")
    print(f"  empirical rate (counted): {empirical_rate:.2f} Hz")
    print(f"  median inter-sample dt  : {median_dt * 1000:.3f} ms  -> {1 / median_dt:.2f} Hz")

    print(f"\n  Mains ground truth (expecting a peak at {args.mains:g} Hz):")
    print(f"  {'assumed rate':>14} | {'peak found':>11} | {'sharpness':>9} | verdict")
    print("  " + "-" * 62)

    best = None
    for assumed in (128.0, 256.0, float(nominal_rate or 0) or 256.0):
        if assumed <= 0:
            continue
        peak_f, sharp = mains_peak(signal, assumed, args.mains)
        error = abs(peak_f - args.mains) if np.isfinite(peak_f) else float("inf")
        verdict = "MATCHES mains" if error < 2.0 else f"off by {error:.1f} Hz"
        print(f"  {assumed:>11.0f} Hz | {peak_f:>8.2f} Hz | {sharp:>9.1f} | {verdict}")
        if best is None or error < best[1]:
            best = (assumed, error)

    print("\n" + "-" * 68)
    verified = empirical_rate
    print(f"  VERIFIED RATE: {verified:.2f} Hz")
    if best and best[1] < 2.0:
        print(f"  Mains check independently supports {best[0]:.0f} Hz.")
    else:
        print("  Mains peak was not clean enough to confirm - check electrode contact,")
        print("  or you may be on a well-filtered supply. The timestamp count still stands.")

    if abs(verified - 128.0) < 5.0:
        print("\n  This stream really is 128 Hz. That is unusual for a Muse 2 - check")
        print("  whether BlueMuse is decimating, before trusting it.")
    elif abs(verified - 256.0) < 5.0:
        print("\n  Confirmed 256 Hz, as expected. Every pre-correction log in")
        print("  logs_precorrection/ was analyzed at half this rate and cannot be")
        print("  cited as pilot data.")

    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
