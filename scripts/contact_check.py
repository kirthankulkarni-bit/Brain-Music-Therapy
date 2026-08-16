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

from eeg_features import MUSE_CHANNELS, FeatureConfig, FeatureExtractor  # noqa: E402
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

# The Muse 2's 12-bit ADC maps to +/-1000 uV (0.48828125 uV per bit), so a raw
# peak-to-peak of 2000 uV is exactly full scale. A channel pinned there is not
# drifting, it is RAILING: the amplifier input is floating because the electrode
# is not making skin contact. Worth distinguishing from ordinary saturation,
# because the fix is different and because a value landing on exactly 2000.0 is
# otherwise a confusing thing to read.
ADC_FULL_SCALE_P2P_UV = 2000.0
RAILING_FRACTION_OF_FULL_SCALE = 0.98

CRITICAL_CHANNELS_DEFAULT = ("AF7", "AF8")


@dataclass
class ChannelQuality:
    name: str
    rms_uv: float
    line_ratio: float
    saturated_fraction: float
    verdict: str
    reason: str
    drift_uv: float = float("nan")

    @property
    def is_usable(self) -> bool:
        return self.verdict in ("GOOD", "FAIR")


def assess(
    signal: np.ndarray,
    sampling_rate: float,
    mains: float,
    name: str,
    extractor: FeatureExtractor,
) -> ChannelQuality:
    """
    Score one channel's most recent window.

    Amplitude is judged on the BAND-PASSED signal, via the same filter the pipeline
    uses (FeatureExtractor.filter_signal). Judging it on raw data measures electrode
    drift rather than signal quality: unfiltered Muse EEG routinely swings hundreds
    of microvolts from slow polarization and post-donning settling, none of which
    survives the 1 Hz high-pass, and none of which the pipeline ever sees.

    Raw drift is still reported, because a large value is a useful signal in its own
    right - it usually means the headset has not settled yet.
    """
    raw = np.asarray(signal, dtype=np.float64)
    drift = float(np.ptp(raw - raw.mean()))

    if drift >= ADC_FULL_SCALE_P2P_UV * RAILING_FRACTION_OF_FULL_SCALE:
        return ChannelQuality(name, float("nan"), float("nan"), 1.0, "RAILING",
                              "electrode floating, amplifier at full scale", drift)

    try:
        filtered = extractor.filter_signal(raw)
    except ValueError as exc:
        return ChannelQuality(name, float("nan"), float("nan"), float("nan"),
                              "BAD", f"filter failed: {exc}", drift)

    sd = float(filtered.std())
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    saturated = float(np.mean(np.abs(filtered) > SATURATION_LEVEL_UV))

    if sd < FLATLINE_SD_UV:
        return ChannelQuality(name, rms, float("nan"), saturated, "DEAD",
                              "flatline, not contacting skin", drift)
    if saturated > SATURATION_FRACTION:
        return ChannelQuality(name, rms, float("nan"), saturated, "BAD",
                              "saturating even after filtering", drift)

    nperseg = int(min(filtered.size, sampling_rate * 2))
    freqs, psd = welch(filtered, fs=sampling_rate, nperseg=nperseg, detrend="constant")

    band = (freqs >= 1.0) & (freqs <= min(80.0, sampling_rate / 2 * 0.95))
    line = (freqs >= mains - 2.0) & (freqs <= mains + 2.0)
    total = float(np.trapz(psd[band], freqs[band])) if band.sum() > 2 else float("nan")
    line_power = float(np.trapz(psd[line], freqs[line])) if line.sum() > 2 else 0.0
    line_ratio = line_power / total if total > 0 else float("nan")

    if rms < RMS_MIN_UV:
        return ChannelQuality(name, rms, line_ratio, saturated, "BAD", "amplitude too low, reseat sensor", drift)
    if rms > RMS_MAX_UV:
        return ChannelQuality(name, rms, line_ratio, saturated, "BAD", "amplitude too high, motion or loose", drift)
    if line_ratio > LINE_RATIO_FAIR:
        return ChannelQuality(name, rms, line_ratio, saturated, "BAD", "heavy mains pickup, poor contact", drift)
    if line_ratio > LINE_RATIO_GOOD:
        return ChannelQuality(name, rms, line_ratio, saturated, "FAIR", "some mains pickup, usable", drift)
    return ChannelQuality(name, rms, line_ratio, saturated, "GOOD", "clean", drift)


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
    parser.add_argument("--critical", default="AF7,AF8",
                        help="the pair that must be good, matching --channels elsewhere")
    parser.add_argument("--demo", action="store_true", help="synthetic stream, no headset")
    args = parser.parse_args()

    critical = tuple(c.strip().upper() for c in args.critical.split(","))

    if args.demo:
        inlet, sampling_rate = DemoInlet(), 256.0
        print("DEMO MODE - synthetic stream (TP9 clean, AF7 clean, AF8 poor, TP10 dead)")
    else:
        inlet, sampling_rate = get_inlet()
        if inlet is None:
            print("\nNo stream. Start BlueMuse, connect the headset, hit 'Start Streaming'.")
            return 1

    # Deliberately NOT the pipeline's filter settings. Contact quality is measured
    # from mains pickup, so the mains frequency must survive: no notch, and a pass
    # band wide enough to include it. The pipeline's 1-45 Hz band-pass plus 60 Hz
    # notch would remove exactly the signal being measured.
    #
    # The 1 Hz high-pass is the part that matters and is shared: it strips the slow
    # electrode drift that would otherwise be mistaken for amplifier saturation.
    contact_band = (1.0, min(80.0, sampling_rate / 2 * 0.9))
    extractor = FeatureExtractor(FeatureConfig(sampling_rate=sampling_rate,
                                               window_seconds=args.window,
                                               bandpass=contact_band,
                                               notch_hz=None))
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
                assess(np.asarray(buffers[i]), sampling_rate, args.mains, name, extractor)
                for i, name in enumerate(MUSE_CHANNELS)
            ]

            elapsed = time.time() - start
            print(f"  t={elapsed:5.1f}s   " + "   ".join(
                f"{q.name}:{q.verdict:<4}" for q in latest
            ))
            for q in latest:
                marker = "*" if q.name in critical else " "
                line = f"{q.line_ratio:.3f}" if np.isfinite(q.line_ratio) else "  -  "
                print(f"    {marker} {q.name:<5} rms {q.rms_uv:6.1f} uV   "
                      f"{args.mains:.0f}Hz ratio {line}   drift {q.drift_uv:7.0f} uV   "
                      f"{q.verdict:<4} {q.reason}")
            print()

    except KeyboardInterrupt:
        print("\nStopped.")

    if not latest:
        print("Not enough data to assess. Is the headset streaming?")
        return 1

    print("=" * 66)
    print("VERDICT")
    print("=" * 66)
    critical = [q for q in latest if q.name in critical]
    bad_critical = [q for q in critical if not q.is_usable]

    for q in latest:
        tag = " (drives the arousal index)" if q.name in critical else ""
        print(f"  {q.name:<5} {q.verdict:<5} {q.reason}{tag}")

    railing = [q for q in latest if q.verdict == "RAILING"]
    if len(railing) >= 3:
        print("\n  ALL CHANNELS RAILING - THIS IS ALMOST CERTAINLY THE REFERENCE ELECTRODE.")
        print()
        print("  Every EEG channel on the Muse is measured against a single reference:")
        print("  the flat pad in the CENTRE of your forehead, between the eyebrows. If")
        print("  that one floats, all four channels rail together, which is what you are")
        print("  seeing. Individual bad electrodes fail one at a time instead.")
        print()
        print("  In order of how often each fixes it:")
        print("    1. Wipe your forehead with a damp cloth. Skin oil is the single")
        print("       biggest source of contact impedance, and it builds up over a day.")
        print("    2. Slide the headband DOWN so the centre pad presses firmly just")
        print("       above the eyebrows. Loose is worse than slightly too tight.")
        print("    3. Dampen the centre pad and the two frontal sensors with water.")
        print("    4. Push any hair out from under the sensors, including fine hairs.")
        print("    5. Wait 60 s after adjusting before rerunning - the electrodes need")
        print("       to settle and the amplifier to come off the rail.")
        print("    6. Check the battery. A nearly flat Muse behaves erratically.")
        print()
        print("  If all four still rail with the headset firmly on a clean forehead,")
        print("  suspect the hardware rather than the fit.")

    drifts = [q.drift_uv for q in latest if np.isfinite(q.drift_uv)]
    if not railing and drifts and np.median(drifts) > 800.0:
        print(f"\n  Note: raw drift is high (median {np.median(drifts):.0f} uV). That is normal for")
        print("  the first 30-60 s after putting the headset on, while the electrodes")
        print("  polarize and settle. It does not affect the verdicts above, which are")
        print("  computed after filtering, but if contact is borderline, wait a minute")
        print("  and rerun.")

    print()
    if bad_critical:
        print("  DO NOT START A SESSION.")
        print(f"  Unusable: {', '.join(q.name for q in bad_critical)}. "
              f"The {'+'.join(critical)} pair is averaged")
        print("  to produce the arousal index, so a bad channel there corrupts every")
        print("  downstream number. Reseat the headband, push hair aside, and dampen the")
        print("  sensors slightly with water. Then rerun this check.")
        if set(critical) == {"AF7", "AF8"}:
            print()
            print("  If the frontal pair will not hold contact, the temporal pair is a")
            print("  legitimate fallback - rerun with --critical TP9,TP10 and pass")
            print("  --channels TP9,TP10 to alpha_test.py and live_music.py.")
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
