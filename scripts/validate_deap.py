"""
validate_deap.py - the control experiment. Run this after the new feature path
is in place, before collecting any fresh data.

The claim being tested: the sampling-rate defect lived in the LIVE path only, and
the new src/eeg_features.py reproduces the offline DEAP results the old code
produced. DEAP was always processed at its true 128 Hz, so if the new extractor
agrees with the old MNE-based pipeline on DEAP, the defect is isolated to the live
path and the paper can say so precisely rather than vaguely.

This is the step people skip, and it is the one that turns "we found a bug" into
"we localized a bug".

It also runs a synthetic-signal test with a known ground truth, which the DEAP
comparison alone cannot give you: a 10 Hz sinusoid must produce alpha-dominant
output at 128 Hz and at 256 Hz alike, and must NOT if the declared rate is wrong.

Usage:
    python scripts/validate_deap.py
    python scripts/validate_deap.py --trials 8 --channel 0
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402

DEAP_RATE = 128.0  # correct for DEAP, and only for DEAP


def legacy_ratio(chunk: np.ndarray, sfreq: float) -> float:
    """The old pipeline, verbatim: MNE PSD, mean over masked bins, linear ratio."""
    import mne

    info = mne.create_info(ch_names=["F3"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(chunk.reshape(1, -1), info, verbose=False)
    psds, freqs = raw.compute_psd(fmin=1, fmax=40, verbose=False).get_data(return_freqs=True)
    alpha = psds[0][(freqs >= 8) & (freqs <= 13)].mean()
    beta = psds[0][(freqs >= 13) & (freqs <= 30)].mean()
    return float(beta / alpha)


def synthetic_check() -> None:
    """Ground-truth test: a 10 Hz alpha sinusoid, seen under correct and wrong rates."""
    print("\n" + "=" * 74)
    print("SYNTHETIC GROUND TRUTH  (10 Hz alpha + 20 Hz beta + 60 Hz mains)")
    print("=" * 74)
    print(f"  {'true rate':>10} | {'declared':>9} | {'peak found':>11} | {'log(b/a)':>9} | verdict")
    print("  " + "-" * 66)

    for true_rate in (128.0, 256.0):
        for declared in (128.0, 256.0):
            seconds = 4.0
            t = np.arange(int(true_rate * seconds)) / true_rate
            rng = np.random.default_rng(3)
            sig = (
                12 * np.sin(2 * np.pi * 10 * t)
                + 4 * np.sin(2 * np.pi * 20 * t)
                + 8 * np.sin(2 * np.pi * 60 * t)
                + rng.normal(0, 5, t.size)
            )
            window = np.vstack([sig] * 4)
            cfg = FeatureConfig(sampling_rate=declared, window_seconds=window.shape[1] / declared)
            feats = FeatureExtractor(cfg).extract(window)
            ok = abs(feats.peak_freq - 10.0) < 1.5
            print(
                f"  {true_rate:>9.0f}  | {declared:>8.0f}  | {feats.peak_freq:>8.2f} Hz | "
                f"{feats.log_beta_alpha:>+9.3f} | "
                f"{'alpha found where expected' if ok else 'ALPHA MISPLACED - rate is wrong'}"
            )


def deap_check(path: str, trials: int, channel: int) -> None:
    print("\n" + "=" * 74)
    print(f"DEAP AGREEMENT CHECK  ({os.path.basename(path)}, channel {channel}, {DEAP_RATE:g} Hz)")
    print("=" * 74)

    with open(path, "rb") as fh:
        deap = pickle.load(fh, encoding="latin1")
    data = deap["data"]

    cfg = FeatureConfig(sampling_rate=DEAP_RATE, window_seconds=5.0, hop_seconds=1.0)
    extractor = FeatureExtractor(cfg)
    n = cfg.window_samples

    print(f"  {'trial':>5} | {'legacy beta/alpha':>18} | {'new beta/alpha':>15} | {'new log10':>10} | {'rank':>5}")
    print("  " + "-" * 66)

    legacy_vals, new_vals = [], []
    for trial in range(min(trials, data.shape[0])):
        chunk = np.asarray(data[trial, channel, :n], dtype=np.float64)
        old = legacy_ratio(chunk, DEAP_RATE)
        feats = extractor.extract(np.vstack([chunk] * 4))
        if not feats.valid:
            print(f"  {trial:>5} | {old:>18.4f} | {'rejected':>15} | {'':>10} | {feats.reject_reason}")
            continue
        new = feats.beta / feats.alpha
        legacy_vals.append(old)
        new_vals.append(new)
        print(f"  {trial:>5} | {old:>18.4f} | {new:>15.4f} | {feats.log_beta_alpha:>+10.3f} | ok")

    if len(legacy_vals) > 2:
        r = float(np.corrcoef(legacy_vals, new_vals)[0, 1])
        rho = _spearman(np.asarray(legacy_vals), np.asarray(new_vals))
        print("\n  Pearson r (legacy vs new) : {:+.4f}".format(r))
        print("  Spearman rho              : {:+.4f}".format(rho))
        print(
            "\n  Absolute values differ by design: the new extractor band-passes and\n"
            "  notches first, and integrates the PSD rather than averaging bins. What\n"
            "  must agree is the ORDERING across trials. A high rho means the new path\n"
            "  measures the same underlying quantity, which localizes the defect to the\n"
            "  live path (wrong declared rate) rather than to the feature definition."
        )
        if rho > 0.8:
            print("\n  PASS: strong rank agreement. The defect is isolated to the live path.")
        else:
            print("\n  INVESTIGATE: rank agreement is weaker than expected. Check the channel,")
            print("  the artifact thresholds, and whether DEAP's pre-applied filtering is")
            print("  interacting with the new band-pass.")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the new feature path against DEAP")
    parser.add_argument("--deap", default=os.path.join(ROOT, "s01.dat"))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--channel", type=int, default=0, help="0 = F3")
    args = parser.parse_args()

    synthetic_check()

    if os.path.exists(args.deap):
        deap_check(args.deap, args.trials, args.channel)
    else:
        print(f"\n  DEAP file not found at {args.deap} - skipping the agreement check.")
        print("  The synthetic ground-truth test above still stands on its own.")

    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
