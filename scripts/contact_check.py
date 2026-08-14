"""
contact_check.py - per-electrode signal quality. Run this before every session.

WHY THIS EXISTS

The baseline phase in live_music.py aborts if too few windows are valid, but it
takes 120 s to tell you that, and it reports a single pooled number: it cannot say
WHICH electrode is bad. Marginal contact is worse still - it does not abort at all,
it just quietly degrades every downstream number.

HOW CONTACT IS MEASURED

Dry electrodes with poor skin contact present high impedance, and high impedance
means more capacitive pickup of mains hum. So the ratio of 60 Hz power to total
power is a usable proxy for contact quality on each channel, without needing the
impedance measurement the Muse does not expose over LSL.

That is the same 60 Hz signal that contaminated the beta band under the old 128 Hz
declaration (see logs_precorrection/README.txt). The difference is that it is now
measured deliberately, on purpose, as a quality metric - instead of being silently
integrated into the feature the whole system was steering on.

Four checks per channel, each catching a distinct failure:

  line ratio   60 Hz power / total power     -> poor contact, high impedance
  RMS          amplitude in microvolts       -> too low = not touching,
                                                too high = motion or loose sensor
  flatline     near-zero variance            -> dead channel or disconnected
  saturation   fraction of samples railed    -> amplifier clipping

Usage:
    python scripts/contact_check.py                 # live, until Ctrl+C
    python scripts/contact_check.py --seconds 30    # fixed duration, then verdict
    python scripts/contact_check.py --demo          # synthetic, no headset needed
    python scripts/contact_check.py --mains 50      # outside North America
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eeg_features import MUSE_CHANNELS  # noqa: E402
from stream_utils import get_inlet  # noqa: E402

# Thresholds. The frontal pair carries the arousal index, so AF7/AF8 are the two
# that must be good; TP9/TP10 are informative but not load-bearing for this study.
LINE_RATIO_GOOD = 0.10
LINE_RATIO_FAIR = 0.25
RMS_MIN_UV = 1.0
RMS_MAX_UV = 80.0
FLATLINE_SD_UV = 0.1
SATURATION_FRACTION = 0.01
SATURATION_LEVEL_UV = 500.0

CRITICAL_CHANNELS = ("AF7", "AF8")


@dataclass
class ChannelQuality:
    name: str
    rms_uv: float
    line_ratio: float
    saturated_fraction: float
    verdict: str
    reason: str

    @property
    def is_usable(self) -> bool:
        return self.verdict in ("GOOD", "FAIR")


def assess(signal: np.ndarray, sampling_rate: float, mains: float, name: str) -> ChannelQuality:
    """Score one channel's most recent window."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()

    sd = float(signal.std())
    rms = float(np.sqrt(np.mean(signal ** 2)))
    saturated = float(np.mean(np.abs(signal) > SATURATION_LEVEL_UV))

    if sd < FLATLINE_SD_UV:
        return ChannelQuality(name, rms, float("nan"), saturated, "DEAD", "flatline, not contacting skin")
    if saturated > SATURATION_FRACTION:
        return ChannelQuality(name, rms, float("nan"), saturated, "BAD", "amplifier saturating")

    nperseg = int(min(signal.size, sampling_rate * 2))
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg, detrend="constant")

    band = (freqs >= 1.0) & (freqs <= min(80.0, sampling_rate / 2 * 0.95))
    line = (freqs >= mains - 2.0) & (freqs <= mains + 2.0)
    total = float(np.trapz(psd[band], freqs[band])) if band.sum() > 2 else float("nan")
    line_power = float(np.trapz(psd[line], freqs[line])) if line.sum() > 2 else 0.0
    line_ratio = line_power / total if total > 0 else float("nan")

    if rms < RMS_MIN_UV:
        return ChannelQuality(name, rms, line_ratio, saturated, "BAD", "amplitude too low, reseat sensor")
    if rms > RMS_MAX_UV:
        return ChannelQuality(name, rms, line_ratio, saturated, "BAD", "amplitude too high, motion or loose")
    if line_ratio > LINE_RATIO_FAIR:
        return ChannelQuality(name, rms, line_ratio, saturated, "BAD", "heavy mains pickup, poor contact")
    if line_ratio > LINE_RATIO_GOOD:
        return ChannelQuality(name, rms, line_ratio, saturated, "FAIR", "some mains pickup, usable")
    return ChannelQuality(name, rms, line_ratio, saturated, "GOOD", "clean")


class DemoInlet:
    """Synthetic 4-channel stream: two clean, one noisy, one dead. Verifies the logic."""

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self._t = 0.0
        self._last = time.time()
        self._rng = np.random.default_rng(11)

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 256):  # noqa: ARG002
        now = time.time()
        n = min(int((now - self._last) * self.sampling_rate), max_samples)
        if n <= 0:
            time.sleep(0.01)
            return [], []
        self._last += n / self.sampling_rate
        t = self._t + np.arange(n) / self.sampling_rate
        self._t = t[-1] + 1.0 / self.sampling_rate

        eeg = 12 * np.sin(2 * np.pi * 10 * t) + 4 * np.sin(2 * np.pi * 20 * t)
        mains = np.sin(2 * np.pi * 60 * t)
        rows = np.column_stack([
            eeg + 3 * mains + self._rng.normal(0, 5, n),    # TP9  clean
            eeg + 2 * mains + self._rng.normal(0, 5, n),    # AF7  clean
            eeg + 40 * mains + self._rng.normal(0, 5, n),   # AF8  poor contact
            np.zeros(n),                                    # TP10 dead
        ])
        return rows.tolist(), list(t + 1.0e9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-electrode contact quality check")
    parser.add_argument("--seconds", type=float, default=None, help="run for N seconds then stop")
    parser.add_argument("--mains", type=float, default=60.0, help="power line frequency")
    parser.add_argument("--window", type=float, default=4.0, help="assessment window, seconds")
    parser.add_argument("--demo", action="store_true", help="synthetic stream, no headset")
    args = parser.parse_args()

    if args.demo:
        inlet, sampling_rate = DemoInlet(), 256.0
        print("DEMO MODE - synthetic stream (TP9 clean, AF7 clean, AF8 poor, TP10 dead)")
    else:
        inlet, sampling_rate = get_inlet()
        if inlet is None:
            print("\nNo stream. Start BlueMuse, connect the headset, hit 'Start Streaming'.")
            return 1

    n_window = int(sampling_rate * args.window)
    buffers: List[List[float]] = [[] for _ in MUSE_CHANNELS]

    print(f"\nAssessing at {sampling_rate:g} Hz over {args.window:g} s windows. Ctrl+C to stop.")
    print("Sit still with a relaxed jaw. Adjust the headband until AF7 and AF8 read GOOD.\n")

    start = time.time()
    last_report = 0.0
    latest: List[ChannelQuality] = []

    try:
        while True:
            if args.seconds is not None and time.time() - start > args.seconds:
                break

            chunk, _ = inlet.pull_chunk(timeout=0.2, max_samples=256)
            for sample in chunk:
                for i in range(len(buffers)):
                    buffers[i].append(sample[i])
                    if len(buffers[i]) > n_window:
                        del buffers[i][:-n_window]

            if len(buffers[0]) < n_window:
                continue
            if time.time() - last_report < 1.0:
                continue
            last_report = time.time()

            latest = [
                assess(np.asarray(buffers[i]), sampling_rate, args.mains, name)
                for i, name in enumerate(MUSE_CHANNELS)
            ]

            elapsed = time.time() - start
            print(f"  t={elapsed:5.1f}s   " + "   ".join(
                f"{q.name}:{q.verdict:<4}" for q in latest
            ))
            for q in latest:
                marker = "*" if q.name in CRITICAL_CHANNELS else " "
                line = f"{q.line_ratio:.3f}" if np.isfinite(q.line_ratio) else "  -  "
                print(f"    {marker} {q.name:<5} rms {q.rms_uv:6.1f} uV   "
                      f"{args.mains:.0f}Hz ratio {line}   {q.verdict:<4} {q.reason}")
            print()

    except KeyboardInterrupt:
        print("\nStopped.")

    if not latest:
        print("Not enough data to assess. Is the headset streaming?")
        return 1

    print("=" * 66)
    print("VERDICT")
    print("=" * 66)
    critical = [q for q in latest if q.name in CRITICAL_CHANNELS]
    bad_critical = [q for q in critical if not q.is_usable]

    for q in latest:
        tag = " (drives the arousal index)" if q.name in CRITICAL_CHANNELS else ""
        print(f"  {q.name:<5} {q.verdict:<5} {q.reason}{tag}")

    print()
    if bad_critical:
        print("  DO NOT START A SESSION.")
        print(f"  Unusable: {', '.join(q.name for q in bad_critical)}. The frontal pair is averaged")
        print("  to produce the arousal index, so a bad channel there corrupts every")
        print("  downstream number. Reseat the headband, push hair aside, and dampen the")
        print("  sensors slightly with water. Then rerun this check.")
        return 2
    if any(q.verdict == "FAIR" for q in critical):
        print("  USABLE, BUT NOT CLEAN. The session will run; expect a higher artifact")
        print("  rejection rate. Worth two more minutes of adjustment first.")
        return 0
    print("  GOOD TO GO. Both frontal electrodes are clean.")
    print("  Next: python src/live_music.py --participant <ID> --duration 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
