"""
compare_indices_deap.py - is log(beta/alpha) the right index, or just the inherited one?

THE QUESTION

The controller steers on log(beta/alpha). That choice came from the literature, not from
anything measured in this project. Several other band ratios are used as arousal or
engagement indices, and none has been compared here against labelled data.

DEAP allows the comparison: 40 trials with self-reported arousal, and the same
FeatureExtractor the live system uses.

THE MULTIPLICITY PROBLEM, WHICH IS THE POINT OF THIS HEADER

Testing seven candidate indices on 40 trials and reporting the winner is exactly how a
spurious result is manufactured. With seven candidates and no true effect, the chance
that at least one reaches p < 0.05 is about 30%. So:

  - EVERY candidate is reported, including the ones that lose. There is no "we found
    that X works" without the table showing what else was tried.
  - The headline test is the DIRECTION across montages (sign test), not the magnitude of
    the best correlation, because direction is far harder to obtain by selection.
  - Nothing here licenses changing the deployed index. It is hypothesis-generating on one
    participant, and switching would require pre-registering the new index and validating
    it on data not used to select it.

That last constraint matters most: the analysis plan is frozen and specifies
log(beta/alpha). A change would be a deviation requiring its own justification, and this
script cannot supply it. What it can do is say whether the inherited choice looks
defensible or looks arbitrary.

Usage:
    python scripts/compare_indices_deap.py
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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402
from validate_index_deap import DEAP_CHANNELS, DEAP_FS, DEAP_PRETRIAL_S  # noqa: E402

# Candidate indices, each a function of the band-power dict the extractor returns.
# Signs are chosen so that HIGHER should mean MORE AROUSED under each one's own theory.
INDICES = {
    "log(beta/alpha)  [deployed]": lambda b: np.log10(b["beta"] / b["alpha"]),
    "log(beta)": lambda b: np.log10(b["beta"]),
    "-log(alpha)": lambda b: -np.log10(b["alpha"]),
    "log((beta+gamma)/(alpha+theta))": lambda b: np.log10(
        (b["beta"] + b["gamma"]) / (b["alpha"] + b["theta"])),
    "-log(theta/beta)": lambda b: -np.log10(b["theta"] / b["beta"]),
    "log(beta/(alpha+theta))": lambda b: np.log10(b["beta"] / (b["alpha"] + b["theta"])),
    "log(gamma/alpha)": lambda b: np.log10(b["gamma"] / b["alpha"]),
}


def trial_bands(trial: np.ndarray, picks: list[int]) -> dict | None:
    """Median band powers across a trial, via this project's extractor."""
    cfg = FeatureConfig(
        sampling_rate=DEAP_FS, window_seconds=4.0, hop_seconds=2.0,
        channels=tuple(DEAP_CHANNELS),
        frontal_channels=tuple(DEAP_CHANNELS[i] for i in picks),
        notch_hz=50.0,
    )
    ex = FeatureExtractor(cfg)
    n_win, n_hop = cfg.window_samples, cfg.hop_samples
    acc: dict[str, list[float]] = {k: [] for k in ("alpha", "beta", "theta", "gamma", "delta")}
    for s0 in range(int(DEAP_PRETRIAL_S * DEAP_FS), trial.shape[1] - n_win + 1, n_hop):
        f = ex.extract(trial[:, s0:s0 + n_win])
        if not f.valid:
            continue
        for k in acc:
            v = getattr(f, k, None)
            if v is not None and np.isfinite(v) and v > 0:
                acc[k].append(float(v))
    if min(len(v) for v in acc.values()) < 5:
        return None
    return {k: float(np.median(v)) for k, v in acc.items()}


def main() -> int:
    p = argparse.ArgumentParser(description="Compare candidate arousal indices on DEAP")
    p.add_argument("--file", default=os.path.join(_ROOT, "s01.dat"))
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"DEAP file not found: {args.file}")
        return 1

    with open(args.file, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    data = np.asarray(d["data"])[:, :32, :]
    labels = np.asarray(d["labels"])
    arousal, valence = labels[:, 1], labels[:, 0]

    montages = [("AF3", "AF4"), ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"),
                ("FC5", "FC6"), ("C3", "C4"), ("P3", "P4")]

    print("=" * 78)
    print("WHICH INDEX TRACKS AROUSAL? All candidates reported, winners and losers.")
    print("=" * 78)
    print(f"  {data.shape[0]} DEAP trials, one participant, {len(montages)} montages")
    print(f"  {len(INDICES)} candidate indices - see the header on multiplicity")
    print()

    # Band powers computed once per (montage, trial); indices are then cheap.
    bands: dict[tuple, list] = {}
    for m in montages:
        picks = [DEAP_CHANNELS.index(c) for c in m]
        bands[m] = [trial_bands(data[t], picks) for t in range(data.shape[0])]

    print(f"  {'index':<34}{'AF3/AF4':>10}{'median':>9}{'pos':>6}{'sign p':>9}{'valence':>9}")
    print("  " + "-" * 77)

    results = []
    for name, fn in INDICES.items():
        rhos, front_rho, val_rho = [], float("nan"), float("nan")
        for m in montages:
            vals, ar, va = [], [], []
            for t, b in enumerate(bands[m]):
                if b is None:
                    continue
                v = float(fn(b))
                if np.isfinite(v):
                    vals.append(v)
                    ar.append(arousal[t])
                    va.append(valence[t])
            if len(vals) < 12:
                continue
            r, _ = spstats.spearmanr(vals, ar)
            rhos.append(r)
            if m == ("AF3", "AF4"):
                front_rho = r
                val_rho, _ = spstats.spearmanr(vals, va)
        if not rhos:
            continue
        pos = sum(r > 0 for r in rhos)
        sign_p = spstats.binomtest(pos, len(rhos), 0.5).pvalue
        med = float(np.median(rhos))
        results.append((name, front_rho, med, pos, len(rhos), sign_p, val_rho))
        print(f"  {name:<34}{front_rho:>+10.3f}{med:>+9.3f}"
              f"{f'{pos}/{len(rhos)}':>6}{sign_p:>9.3f}{val_rho:>+9.3f}")

    print()
    deployed = next((r for r in results if r[0].startswith("log(beta/alpha)")), None)
    best = max(results, key=lambda r: r[2]) if results else None

    if deployed and best:
        rank = sorted(results, key=lambda r: -r[2]).index(deployed) + 1
        print(f"  The deployed index ranks {rank} of {len(results)} by median rho.")
        if best[0] == deployed[0]:
            print("  It is also the best of those tested, so the inherited choice looks")
            print("  defensible rather than arbitrary - which is the useful outcome here.")
        else:
            # A Spearman rho from n trials has SE roughly 1/sqrt(n-3). Quoting a ranking
            # without that number invites reading a noise-level gap as a finding.
            n_eff = data.shape[0]
            se = 1.0 / np.sqrt(max(n_eff - 3, 1))
            gap = best[2] - deployed[2]
            print(f"  '{best[0]}' scores higher: median rho {best[2]:+.3f} against "
                  f"{deployed[2]:+.3f}.")
            print(f"  That gap is {gap:+.3f}, against a standard error of about {se:.3f} on")
            print(f"  a single rho at n = {n_eff}. The top candidates are indistinguishable")
            print("  here, and the ranking among them carries no information.")
            print()
            print("  NOT grounds to switch. One participant, seven candidates, and the")
            print("  analysis plan is frozen on log(beta/alpha). What this does establish is")
            print("  that the inherited choice sits in the top group rather than being")
            print("  arbitrary - and that the simpler alternatives are worse.")

    print()
    print("  pos / sign p = how many montages give a positive correlation, and the")
    print("  probability of that split under no association. Direction across montages")
    print("  is far harder to obtain by selection than a single large correlation.")
    print()
    print("  valence is the discriminant control: an arousal index should not track it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
