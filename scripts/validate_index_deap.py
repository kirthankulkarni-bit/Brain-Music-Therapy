"""
validate_index_deap.py - does log(beta/alpha) actually track arousal?

THE GAP THIS FILLS

Two different claims sit under this project's measurement, and only one has ever been
tested:

  1. the electrodes record cortical activity   - tested by the eyes-closed alpha test,
                                                 and see finding_channel_validation.md
                                                 for the channels it actually covered
  2. log(beta/alpha) is a measure of AROUSAL   - never tested here at all

Claim 2 is inherited from the literature and asserted. The whole study steers on it, and
the target is stated in its units, so it is worth testing on data with labels rather than
taking on faith.

DEAP provides exactly that: 40 trials per participant of 32-channel EEG with
self-reported arousal on a 1-9 scale. It is public, so a reviewer can check this without
the headset, the library, or any of this project's own recordings.

WHAT IS TESTED

For each trial, compute log(beta/alpha) on frontal channels with THIS PROJECT'S
extractor - not a reimplementation - and correlate it against the participant's own
arousal rating. A positive correlation is what the index claims; anything else is a
problem for the study, not for DEAP.

Spearman rather than Pearson: the ratings are ordinal on a bounded scale and there is no
reason to expect linearity.

WHAT THIS CANNOT SHOW. One DEAP participant is one participant, and self-report is a
weak criterion. A null result would not disprove the index, and a positive one does not
establish it holds for this project's montage - DEAP uses 32 electrodes on a standard cap,
not four dry sensors on a headband. This is a construct-validity check on a public
dataset, which is more than the project had, and less than a validation.

Usage:
    python scripts/validate_index_deap.py
    python scripts/validate_index_deap.py --file s01.dat --channels AF3 AF4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from scipy import stats as spstats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402

DEAP_FS = 128.0
DEAP_PRETRIAL_S = 3.0        # DEAP trials carry 3 s of pre-trial baseline

# DEAP's 32-channel order (Geneva). Needed because the useful comparison is against
# frontal sites near where the Muse sits, not against an arbitrary index.
DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]


def trial_index(trial: np.ndarray, picks: list[int]) -> float:
    """
    log(beta/alpha) for one trial, via this project's own FeatureExtractor.

    The extractor expects (channels, samples) and applies its own detrend, bandpass and
    notch. Using it rather than a fresh Welch is the point: this tests the deployed
    feature path, not an idealised version of it.
    """
    cfg = FeatureConfig(
        sampling_rate=DEAP_FS,
        window_seconds=4.0,
        hop_seconds=2.0,
        channels=tuple(DEAP_CHANNELS),
        frontal_channels=tuple(DEAP_CHANNELS[i] for i in picks),
        notch_hz=50.0,                      # DEAP was recorded in Europe
    )
    ex = FeatureExtractor(cfg)
    n_win, n_hop = cfg.window_samples, cfg.hop_samples
    start = int(DEAP_PRETRIAL_S * DEAP_FS)
    vals = []
    for s0 in range(start, trial.shape[1] - n_win + 1, n_hop):
        f = ex.extract(trial[:, s0:s0 + n_win])
        if f.valid and np.isfinite(f.log_beta_alpha):
            vals.append(f.log_beta_alpha)
    return float(np.median(vals)) if len(vals) >= 5 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Test log(beta/alpha) against DEAP arousal")
    p.add_argument("--file", default=os.path.join(_ROOT, "s01.dat"))
    p.add_argument("--channels", nargs="+", default=["AF3", "AF4"],
                   help="DEAP channels standing in for the Muse frontal pair")
    p.add_argument("--montage-sweep", action="store_true",
                   help="repeat across seven montages and sign-test the direction")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"DEAP file not found: {args.file}")
        print("This validation needs DEAP s01.dat (not redistributable; request from the")
        print("DEAP authors). Everything else in the project runs without it.")
        return 1

    with open(args.file, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    data = np.asarray(d["data"])[:, :32, :]          # EEG only; 33-40 are peripheral
    labels = np.asarray(d["labels"])

    try:
        picks = [DEAP_CHANNELS.index(c) for c in args.channels]
    except ValueError as exc:
        print(f"Unknown channel: {exc}")
        return 1

    print("=" * 78)
    print("CONSTRUCT VALIDITY: does log(beta/alpha) track self-reported arousal?")
    print("=" * 78)
    print(f"  file     : {os.path.basename(args.file)}")
    print(f"  trials   : {data.shape[0]}, {data.shape[2] / DEAP_FS:.0f} s each at {DEAP_FS:g} Hz")
    print(f"  channels : {', '.join(args.channels)}")
    print(f"  extractor: this project's FeatureExtractor, notch 50 Hz")
    print()

    idx = np.array([trial_index(data[t], picks) for t in range(data.shape[0])])
    ok = np.isfinite(idx)
    if ok.sum() < 12:
        print(f"  only {ok.sum()} trials produced a usable index - cannot test.")
        return 1

    arousal = labels[:, 1]
    valence = labels[:, 0]

    rho_a, p_a = spstats.spearmanr(idx[ok], arousal[ok])
    rho_v, p_v = spstats.spearmanr(idx[ok], valence[ok])

    print(f"  usable trials            : {ok.sum()} of {data.shape[0]}")
    print(f"  index range              : {idx[ok].min():+.3f} to {idx[ok].max():+.3f}")
    print()
    print(f"  vs AROUSAL  (the claim)  : rho = {rho_a:+.3f}, p = {p_a:.3f}")
    print(f"  vs valence  (control)    : rho = {rho_v:+.3f}, p = {p_v:.3f}")
    print()

    if p_a < 0.05 and rho_a > 0:
        print("  The index tracks arousal in the predicted direction on labelled data.")
    elif p_a < 0.05 and rho_a < 0:
        print("  SIGNIFICANT IN THE WRONG DIRECTION. The index moves opposite to reported")
        print("  arousal here. That is a problem for the index, not for DEAP, and it must")
        print("  be resolved before the index is used to steer an intervention.")
    else:
        print("  No significant association. With one participant and 40 trials this is")
        print("  weak evidence either way - the test is underpowered for a small effect -")
        print("  but the index's central claim remains untested on labelled data.")

    if args.montage_sweep:
        # A single correlation at n=40 is underpowered, but the DIRECTION across
        # independent montages is not: under no association each is a coin flip, so a
        # consistent sign across seven is testable even when no single one reaches 0.05.
        print()
        print("  MONTAGE SWEEP - is the direction specific to one channel choice?")
        print(f"    {'channels':<14}{'rho arousal':>13}{'p':>8}{'rho valence':>13}")
        print("    " + "-" * 50)
        rhos = []
        for pair in (("AF3", "AF4"), ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"),
                     ("T7", "T8"), ("P3", "P4"), ("O1", "O2")):
            pk = [DEAP_CHANNELS.index(c) for c in pair]
            v = np.array([trial_index(data[t], pk) for t in range(data.shape[0])])
            m = np.isfinite(v)
            if m.sum() < 12:
                continue
            ra, pa = spstats.spearmanr(v[m], arousal[m])
            rv, _ = spstats.spearmanr(v[m], valence[m])
            rhos.append(ra)
            print(f"    {'/'.join(pair):<14}{ra:>+13.3f}{pa:>8.3f}{rv:>+13.3f}")
        if rhos:
            pos = sum(r > 0 for r in rhos)
            sign_p = spstats.binomtest(pos, len(rhos), 0.5).pvalue
            print()
            print(f"    {pos}/{len(rhos)} montages positive, sign test p = {sign_p:.3f}, "
                  f"median rho {np.median(rhos):+.3f}")
            print("    Individually underpowered; the consistent direction is the result.")

    print()
    print("  Valence is included as a discriminant control. An index claiming to measure")
    print("  AROUSAL should not correlate more strongly with valence than with arousal;")
    print("  if it does, it is measuring something other than what it is named for.")
    print()
    print("  One participant, self-report, and a 32-channel cap rather than four dry")
    print("  sensors. This is a construct-validity check, not a validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
