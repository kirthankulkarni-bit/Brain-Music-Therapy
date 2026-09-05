"""
alpha_test.py - the eyes-closed alpha validation, run as a controlled experiment.

WHY THIS IS A SEPARATE SCRIPT

Alpha power rises sharply when the eyes close. It is the oldest and most reliable
finding in EEG (Berger, 1929) and it is the standard way to prove a recording setup
measures cortex rather than amplifier noise. If this does not reproduce, nothing
built on top of the signal means anything.

Doing it by hand inside live_music.py does not work: there is no way to mark when
you opened or closed your eyes, so afterwards you are left eyeballing a trace and
guessing where the boundaries were. This script runs the block sequence itself, so
every window carries the condition it belongs to, and the result is a statistic
rather than an impression.

TWO METHODOLOGICAL DETAILS THAT MATTER

1. WINDOWS STRADDLING A TRANSITION ARE DISCARDED. A 4 s window that starts 2 s
   before you close your eyes contains both conditions. Keeping it would blur the
   contrast toward zero. Any window not fully inside one block is dropped, and the
   count of discarded windows is reported.

2. THE CUES ARE AUDIBLE, NOT PRINTED. You cannot read a terminal with your eyes
   closed. A low tone means close your eyes, a high tone means open them.

Output: per-condition alpha statistics, Welch's t-test, Cohen's d, a verdict, and a
figure saved next to the session log.

Usage:
    python scripts/alpha_test.py                        # 6 blocks x 60 s
    python scripts/alpha_test.py --blocks 4 --block-seconds 45
    python scripts/alpha_test.py --demo                 # synthetic, verifies the logic
    python scripts/alpha_test.py --no-audio-cues        # printed cues only
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from typing import List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402
from session_logger import SessionLogger  # noqa: E402
from stream_utils import get_inlet  # noqa: E402

EYES_OPEN = "eyes_open"
EYES_CLOSED = "eyes_closed"


def beep(frequency: float, duration: float = 0.4) -> None:
    """Audible cue. Falls back to the terminal bell if no audio device is available."""
    try:
        import sounddevice as sd

        sr = 44100
        t = np.arange(int(sr * duration)) / sr
        tone = 0.25 * np.sin(2 * np.pi * frequency * t)
        fade = int(0.02 * sr)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        sd.play(tone.astype(np.float32), sr, blocking=True)
    except Exception:  # noqa: BLE001 - a missing audio device must not stop the test
        print("\a", end="", flush=True)


class DemoInlet:
    """Synthetic stream whose alpha genuinely doubles on a 60 s cycle, to verify the stats."""

    def __init__(self, sampling_rate: float, block_seconds: float):
        self.sampling_rate = sampling_rate
        self.block_seconds = block_seconds
        self._t = 0.0
        self._last = time.time()
        self._rng = np.random.default_rng(23)

    def reset(self) -> None:
        """Align the synthetic block clock with the test loop's start time.

        Without this the generator runs during the pre-test countdown, so its
        blocks sit ~5 s ahead of the condition labels and the demo appears to show
        alpha changing before the cue. Only affects the demo harness.
        """
        self._t = 0.0
        self._last = time.time()

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 256):  # noqa: ARG002
        now = time.time()
        n = min(int((now - self._last) * self.sampling_rate), max_samples)
        if n <= 0:
            time.sleep(0.01)
            return [], []
        self._last += n / self.sampling_rate
        t = self._t + np.arange(n) / self.sampling_rate
        self._t = t[-1] + 1.0 / self.sampling_rate

        # Odd-numbered blocks are "eyes closed" and get 2.2x the alpha amplitude.
        block = (t // self.block_seconds).astype(int)
        gain = np.where(block % 2 == 1, 22.0, 10.0)
        alpha = gain * np.sin(2 * np.pi * 10.0 * t)
        beta = 4.0 * np.sin(2 * np.pi * 20.0 * t)
        rows = np.column_stack([alpha + beta + self._rng.normal(0, 5, n) for _ in range(4)])
        return rows.tolist(), list(t + 1.0e9)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def save_figure(records: List[dict], path: str, block_seconds: float) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    valid = [r for r in records if r["valid"]]
    if not valid:
        return None

    t = np.asarray([r["t"] for r in valid])
    alpha = np.log10(np.asarray([r["alpha"] for r in valid]))

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Shade whole eyes-closed blocks, not per-window slivers.
    blocks = sorted({r["block"] for r in records})
    for b in blocks:
        if b % 2 != 1:
            continue
        ax.axvspan(b * block_seconds, (b + 1) * block_seconds,
                   color="#4a90d9", alpha=0.18, linewidth=0,
                   label="eyes closed" if b == blocks[1] else None)

    ax.plot(t, alpha, color="#00688b", linewidth=1.8, label="log10 alpha power")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("log10 alpha power")
    ax.set_title("Eyes-closed alpha validation  (shaded = eyes closed)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Eyes-open vs eyes-closed alpha validation")
    parser.add_argument("--participant", default="alphatest")
    parser.add_argument("--blocks", type=int, default=6, help="total blocks, alternating")
    parser.add_argument("--block-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0,
                        help="discard this long after each transition")
    parser.add_argument("--channels", default="AF7,AF8",
                        help="electrode pair to measure alpha from, e.g. TP9,TP10")
    parser.add_argument("--reject-p2p", type=float, default=350.0,
                                  help="artifact rejection threshold, uV peak-to-peak per window. "
                                  "Derived from measured blink amplitudes; see FeatureConfig.")
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-audio-cues", action="store_true")
    args = parser.parse_args()

    if args.demo:
        sampling_rate = 256.0
        inlet = DemoInlet(sampling_rate, args.block_seconds)
        print("DEMO MODE - synthetic stream with a genuine 2.2x alpha increase on closed blocks")
    else:
        inlet, sampling_rate = get_inlet()
        if inlet is None:
            print("\nNo stream. Start BlueMuse, connect the headset, hit 'Start Streaming'.")
            return 1

    pair = tuple(c.strip().upper() for c in args.channels.split(","))
    cfg = FeatureConfig(sampling_rate=sampling_rate, window_seconds=args.window,
                        hop_seconds=args.hop, frontal_channels=pair,
                        reject_peak_to_peak_uv=args.reject_p2p)
    print(f"Measuring alpha from: {'+'.join(pair)}")
    extractor = FeatureExtractor(cfg)
    buffers = [collections.deque(maxlen=cfg.window_samples) for _ in range(len(cfg.channels))]

    logger = SessionLogger(participant_id=args.participant,
                           condition="alpha_validation",
                           synthetic=args.demo)
    logger.write_manifest(
        sampling_rate=sampling_rate,
        sampling_rate_source="demo" if args.demo else "lsl_nominal_srate",
        feature_config=cfg.to_dict(),
        protocol="alternating eyes-open / eyes-closed blocks",
        index_channels=list(pair),
        blocks=args.blocks,
        block_seconds=args.block_seconds,
        settle_seconds=args.settle_seconds,
    )

    total = args.blocks * args.block_seconds
    print(f"\n{args.blocks} blocks x {args.block_seconds:.0f} s = {total / 60:.1f} minutes total.")
    print("Low tone  = CLOSE your eyes.   High tone = OPEN your eyes.")
    print("Sit still, relax your jaw, breathe normally. Starting in 5 s...\n")
    time.sleep(5)

    records: List[dict] = []
    if hasattr(inlet, "reset"):
        inlet.reset()
    start = time.time()
    next_hop = start + cfg.window_seconds
    current_block = -1

    try:
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= total:
                break

            block = int(elapsed // args.block_seconds)
            condition = EYES_CLOSED if block % 2 == 1 else EYES_OPEN

            if block != current_block:
                current_block = block
                if not args.no_audio_cues:
                    beep(330.0 if condition == EYES_CLOSED else 880.0)
                label = "CLOSE your eyes" if condition == EYES_CLOSED else "OPEN your eyes"
                print(f"\n  [{elapsed:5.0f}s] block {block + 1}/{args.blocks}: {label}")
                logger.note(f"block {block + 1} start: {condition}", level="info",
                            block=block, condition=condition)

            chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=512)
            if chunk:
                for sample in chunk:
                    for i in range(len(buffers)):
                        buffers[i].append(sample[i])
                logger.log_raw(timestamps, chunk)

            if now < next_hop:
                time.sleep(0.01)
                continue
            next_hop += cfg.hop_seconds

            if len(buffers[0]) < cfg.window_samples:
                continue

            window = np.asarray([list(b) for b in buffers], dtype=np.float64)
            feats = extractor.extract(window, timestamp=now)

            # A window covers [elapsed - window_seconds, elapsed]. Keep it only if
            # that whole span sits inside the current block, past the settle time.
            block_start = block * args.block_seconds
            window_start = elapsed - cfg.window_seconds
            in_block = window_start >= (block_start + args.settle_seconds)

            records.append({
                "t": elapsed,
                "block": block,
                "condition": condition,
                "in_block": in_block,
                "valid": bool(feats.valid),
                "alpha": feats.alpha,
                "log_beta_alpha": feats.log_beta_alpha,
            })
            logger.log_window(condition, feats, block=block, in_block=in_block)

            if feats.valid and in_block:
                print(f"    {elapsed:5.1f}s  {condition:<12} alpha={feats.alpha:9.2f}", end="\r")

    except KeyboardInterrupt:
        print("\n\nInterrupted - analyzing what was collected.")
    finally:
        logger.close()

    # ------------------------------------------------------------------ stats

    usable = [r for r in records if r["valid"] and r["in_block"]]
    closed = np.asarray([r["alpha"] for r in usable if r["condition"] == EYES_CLOSED])
    opened = np.asarray([r["alpha"] for r in usable if r["condition"] == EYES_OPEN])

    print("\n\n" + "=" * 68)
    print("EYES-CLOSED ALPHA VALIDATION")
    print("=" * 68)
    print(f"  windows total          : {len(records)}")
    print(f"  discarded, artifact    : {sum(1 for r in records if not r['valid'])}")
    print(f"  discarded, straddling  : {sum(1 for r in records if r['valid'] and not r['in_block'])}")
    print(f"  usable                 : {len(usable)}  "
          f"({opened.size} open, {closed.size} closed)")

    if closed.size < 5 or opened.size < 5:
        print("\n  Not enough usable windows to conclude. Check electrode contact and rerun.")
        return 1

    ratio = float(closed.mean() / opened.mean()) if opened.mean() > 0 else float("nan")
    log_closed = np.log10(closed)
    log_open = np.log10(opened)
    d = cohens_d(log_closed, log_open)

    try:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(log_closed, log_open, equal_var=False)
    except ImportError:
        t_stat, p_value = float("nan"), float("nan")

    print(f"\n  alpha, eyes OPEN       : {opened.mean():10.2f}  (median {np.median(opened):.2f})")
    print(f"  alpha, eyes CLOSED     : {closed.mean():10.2f}  (median {np.median(closed):.2f})")
    print(f"  ratio closed/open      : {ratio:10.2f}x")
    print(f"  Cohen's d (log alpha)  : {d:+10.2f}")
    print(f"  Welch t                : {t_stat:+10.2f}")
    print(f"  p                      : {p_value:10.2e}")

    print("\n  " + "-" * 64)
    if ratio >= 1.5 and p_value < 0.01:
        print("  PASS. Clear alpha increase on eye closure. The sensing path measures")
        print("  cortex, not amplifier noise. This figure belongs in the paper.")
        verdict = 0
    elif ratio >= 1.2 and p_value < 0.05:
        print("  WEAK BUT PRESENT. Real, smaller than typical. Frontal alpha is weaker")
        print("  than occipital in most people, so this may simply be your anatomy.")
        print("  Worth rerunning once with better contact before accepting it.")
        verdict = 0
    else:
        print("  FAIL. No reliable alpha increase. Most likely causes, in order:")
        print("    1. electrode contact degraded - rerun scripts/contact_check.py")
        print("    2. not relaxed enough; alpha is suppressed by mental effort")
        print("    3. frontal sensors reading muscle rather than cortex")
        print("  Do not proceed to participant sessions until this passes.")
        verdict = 2

    fig_path = save_figure(records, os.path.join(logger.dir, "alpha_validation.png"), args.block_seconds)
    print("\n  session : " + logger.dir)
    if fig_path:
        print("  figure  : " + fig_path)
    print("=" * 68)
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
