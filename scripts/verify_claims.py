"""
verify_claims.py - regenerate every number the preprint cites, and check it.

WHY

A preprint's numbers get copied from a terminal into prose once, and then the code
moves. Six weeks later the manuscript says 1.14x and the repository says 1.05x, and
nobody knows which is right or when it changed. That is how honest projects end up
with unreproducible papers.

So every headline claim is recorded here with the value asserted in the manuscript and
the computation that produces it, straight from the artefacts on disk. Running this
before submission tells you whether the paper still describes the code.

A failure here is not necessarily a bug. It means a number moved, and the manuscript
has to move with it - or the claim was wrong. Either way it must be looked at rather
than rounded away, so the tolerances are tight.

Usage:
    python scripts/verify_claims.py
    python scripts/verify_claims.py --verbose
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


def _bench(pattern: str) -> list[dict]:
    return [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(os.path.join(_ROOT, "benchmarks", pattern)))]


def _median_gen(run: dict, precision: str, duration: float) -> float:
    for row in run.get("musicgen") or []:
        if row["precision"] == precision and row["duration_s"] == duration:
            return float(row["median_generation_s"])
    return float("nan")


def _pilot() -> dict:
    from session_logger import load_session
    dirs = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*")))
    if not dirs:
        raise FileNotFoundError("no PILOT session on disk")
    return load_session(dirs[-1])


def _pilot_z() -> np.ndarray:
    z = np.asarray([w["z"] for w in _pilot()["windows"]
                    if w.get("phase") == "intervention" and w.get("valid")
                    and isinstance(w.get("z"), (int, float))], dtype=float)
    return z[np.isfinite(z)]


# ---------------------------------------------------------------- the claims


def claim_t4_best_realtime() -> tuple[float, str]:
    """Best realtime factor across all T4 runs and precisions."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    best = min(row["median_generation_s"] / row["duration_s"]
               for r in runs for row in r["musicgen"])
    return best, f"{len(runs)} T4 runs, all precisions"


def claim_t4_precision_ordering() -> tuple[float, str]:
    """Fraction of run x duration cells where fp32 < fp16-half < fp16."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    cells = ok = 0
    for r in runs:
        for d in (4.0, 8.0):
            a, b, c = (_median_gen(r, "fp32", d), _median_gen(r, "fp16-half", d),
                       _median_gen(r, "fp16", d))
            if all(np.isfinite([a, b, c])):
                cells += 1
                ok += int(a < b < c)
    return ok / max(1, cells), f"{ok} of {cells} cells"


def claim_t4_fp16half_vs_fp32() -> tuple[float, str]:
    """fp16-half speedup over fp32 at 8 s, median across runs."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    a = np.median([_median_gen(r, "fp32", 8.0) for r in runs])
    b = np.median([_median_gen(r, "fp16-half", 8.0) for r in runs])
    return a / b, "median of 3 T4 runs at 8 s"


def claim_t4_between_run_variance() -> tuple[float, str]:
    """Worst max/min across T4 runs for any configuration."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    worst = 0.0
    for p in ("fp32", "fp16", "fp16-half"):
        for d in (4.0, 8.0):
            v = [_median_gen(r, p, d) for r in runs]
            v = [x for x in v if np.isfinite(x)]
            if len(v) > 1:
                worst = max(worst, max(v) / min(v))
    return worst, "worst config across 3 T4 runs"


def claim_pilot_effective_n() -> tuple[float, str]:
    z = _pilot_z()
    rho = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    return z.size * (1 - rho) / (1 + rho), f"AR(1) from {z.size} windows"


def claim_pilot_autocorrelation() -> tuple[float, str]:
    z = _pilot_z()
    return float(np.corrcoef(z[:-1], z[1:])[0, 1]), "lag-1 of PILOT01 intervention z"


def claim_pilot_rejection() -> tuple[float, str]:
    rows = [w for w in _pilot()["windows"] if w.get("phase") == "intervention"]
    return sum(1 for w in rows if not w.get("valid")) / len(rows), f"{len(rows)} windows"


def claim_chatter_before() -> tuple[float, str]:
    """Prompt changes actually logged in PILOT01."""
    audio = _pilot()["audio"]
    return float(sum(1 for i in range(1, len(audio))
                     if audio[i].get("prompt") != audio[i - 1].get("prompt"))), \
        f"{len(audio)} audio events"


def claim_chatter_after() -> tuple[float, str]:
    """Prompt changes when the same z is replayed through the fixed controller."""
    from music_engine import build_prompt
    z = _pilot_z()
    hist, prev, changes = collections.deque(maxlen=20), None, 0
    for v in z:
        hist.append(v)
        tr = (float(np.polyfit(np.arange(len(hist), dtype=float), np.asarray(hist), 1)[0])
              if len(hist) >= 20 else None)
        p = build_prompt(float(v), -1.0, tr, previous_prompt=prev)
        if prev is not None and p != prev:
            changes += 1
        prev = p
    return float(changes), f"replay of {z.size} windows"


def claim_library_clipping_bound() -> tuple[float, str]:
    from library_engine import LibraryConfig
    man = json.load(open(os.path.join(_ROOT, "library", "manifest.json"), encoding="utf-8"))
    peaks = [s["peak"] for e in man["prompts"] for s in e["segments"]]
    return max(peaks) * (2 ** 0.5) * LibraryConfig().output_gain, \
        f"{len(peaks)} segments, gain {LibraryConfig().output_gain}"


def claim_library_dominant_variants() -> tuple[float, str]:
    """Renders available for the prompt that carries a relaxation session."""
    from music_engine import _ENERGY_LADDER
    man = json.load(open(os.path.join(_ROOT, "library", "manifest.json"), encoding="utf-8"))
    for e in man["prompts"]:
        if e["prompt"] == _ENERGY_LADDER[1]:
            return float(len(e["segments"])), "rung 1 base, 96% of PILOT01"
    return float("nan"), "not found"


def claim_latency_budget() -> tuple[float, str]:
    """End-to-end worst case with the library engine."""
    from library_engine import LibraryConfig
    runs = _bench("latency_colab-tesla-t4-run1.json")
    analysis = runs[0]["analysis"][1]["total_analysis_latency_s"]
    return analysis + LibraryConfig().crossfade_seconds, \
        f"{analysis:g} s analysis + {LibraryConfig().crossfade_seconds:g} s crossfade"


def claim_coupling_recovers_lag() -> tuple[float, str]:
    """Ground-truth check: the estimator must return the lag it was given."""
    from analyze_session import coupling_index
    from validate_coupling import build_session
    got = coupling_index(build_session(6.0, retrospective=False, seed=1),
                         n_permutations=120).get("aci_peak_lag_s", float("nan"))
    return float(got), "synthetic session, true lag +6.0 s"


def claim_alpha_validation_ratio() -> tuple[float, str]:
    """Eyes-closed alpha increase - the evidence the rig measures cortex."""
    from session_logger import load_session
    dirs = sorted(glob.glob(os.path.join(_ROOT, "sessions", "alphatest*")))
    session = load_session(dirs[-1])
    rows = [w for w in session["windows"]
            if w.get("phase") in ("eyes_open", "eyes_closed")
            and isinstance(w.get("alpha"), (int, float)) and np.isfinite(w["alpha"])]
    a = np.log10(np.asarray([w["alpha"] for w in rows], dtype=float))
    closed = np.asarray([w["phase"] == "eyes_closed" for w in rows], dtype=bool)
    return float(10 ** (a[closed].mean() - a[~closed].mean())),         f"{closed.sum()} closed / {(~closed).sum()} open windows"


def claim_channel_mismatch_af() -> tuple[float, str]:
    """Eyes-closed alpha ratio on AF7/AF8 - the channels the study actually uses."""
    from eeg_features import FeatureConfig, FeatureExtractor
    from session_logger import load_raw, load_session

    d = sorted(glob.glob(os.path.join(_ROOT, "sessions", "alphatest*")))[-1]
    chans = load_raw(d)[:, 1:].T.astype(float)
    session = load_session(d)
    tl = [(float(w["elapsed_s"]), w["phase"]) for w in session["windows"]
          if w.get("phase") in ("eyes_open", "eyes_closed")]

    def phase_at(t):
        prev = None
        for tt, ph in tl:
            if tt > t:
                return prev
            prev = ph
        return prev

    cfg = FeatureConfig(sampling_rate=256.0, frontal_channels=("AF7", "AF8"))
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    op, cl = [], []
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        f = ex.extract(chans[:, s0:s0 + nw])
        if not (f.valid and np.isfinite(f.alpha) and f.alpha > 0):
            continue
        ph = phase_at((s0 + nw) / 256.0 + 6.25)
        (op if ph == "eyes_open" else cl if ph == "eyes_closed" else []).append(f.alpha)
    if len(op) < 10 or len(cl) < 10:
        return float("nan"), "insufficient"
    return float(10 ** (np.log10(cl).mean() - np.log10(op).mean())),         f"{len(op)} open / {len(cl)} closed windows"


def claim_deap_arousal_rho() -> tuple[float, str]:
    """Spearman rho between log(beta/alpha) and DEAP self-reported arousal, AF3/AF4."""
    import pickle
    from scipy import stats as spstats
    from validate_index_deap import DEAP_CHANNELS, trial_index

    path = os.path.join(_ROOT, "s01.dat")
    if not os.path.exists(path):
        return float("nan"), "s01.dat absent"
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    data = np.asarray(d["data"])[:, :32, :]
    arousal = np.asarray(d["labels"])[:, 1]
    picks = [DEAP_CHANNELS.index(c) for c in ("AF3", "AF4")]
    idx = np.array([trial_index(data[t], picks) for t in range(data.shape[0])])
    ok = np.isfinite(idx)
    rho, _ = spstats.spearmanr(idx[ok], arousal[ok])
    return float(rho), f"{ok.sum()} trials, AF3/AF4"


def claim_streaming_latency_budget() -> tuple[float, str]:
    """Total analysis latency of the low-latency estimator."""
    from eeg_features import FeatureConfig, StreamingBandPower
    est = StreamingBandPower(FeatureConfig(sampling_rate=256.0), tau_seconds=0.25, order=4)
    b = est.latency_budget()
    return float(b["total_analysis_latency_s"]), "order 4, tau 0.25"


# claim -> (function, value asserted in the manuscript, tolerance)
_REPLAY_CACHE: dict = {}


def _replay_ctx():
    """
    The PILOT01 reconstruction, computed once and reused by the claims below.

    Reconstructing z from raw runs the real FeatureExtractor over ~1300 s of 256 Hz
    data, twice (two estimator configurations). Doing that per claim would put minutes
    into a suite that has to stay cheap enough to run before every commit.
    """
    if not _REPLAY_CACHE:
        import controller_replay as cr
        d = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*")))[-1]
        session, chans, ts, pair, base = cr.load(d)
        t0, v0 = cr.reconstruct(chans, ts, pair, cr.DEPLOYED[0], cr.DEPLOYED[1], 0.001)
        offset, r = cr.align(t0, v0, session)
        _REPLAY_CACHE.update(cr=cr, session=session, chans=chans, ts=ts, pair=pair,
                             base=base, offset=offset, r=r)
    return _REPLAY_CACHE


def _replay_z(config):
    c = _replay_ctx()
    key = ("z", config)
    if key not in c:
        c[key] = c["cr"].z_series(c["chans"], c["ts"], c["pair"], c["session"],
                                  c["base"], *config, c["offset"])
    return c[key]


def claim_replay_fidelity() -> tuple[float, str]:
    """
    How well the offline reconstruction reproduces the deployed pipeline.

    Every retuned-estimator number depends on this, because no session log exists for a
    configuration that was never run. Below 0.9 the replay describes a different system
    - a hand-rolled Welch scores 0.05 - so this is the load-bearing assumption behind
    the whole finding, and it is asserted rather than assumed.
    """
    c = _replay_ctx()
    return float(c["r"]), f"offset {c['offset']:+.2f} s against the session log"


def claim_retuned_chatter_no_dwell() -> tuple[float, str]:
    """Switches arriving inside a crossfade at 2 s / 0.5 s / tau 0.5, no dwell."""
    c = _replay_ctx()
    z, t = _replay_z(c["cr"].RETUNED)
    m = c["cr"].replay(z, t, -1.0, 0.0, 0.0, 1.0)
    return float(m["under_crossfade"]), f"{m['changes']} changes over {z.size} windows"


def claim_retuned_chatter_with_dwell() -> tuple[float, str]:
    """
    The same configuration with a dwell of one crossfade.

    This is the result that makes the retuning usable at all: a dwell of at least one
    crossfade is precisely the condition for no switch arriving before the previous
    crossfade finishes, so the count is zero by construction rather than by tuning.
    """
    c = _replay_ctx()
    z, t = _replay_z(c["cr"].RETUNED)
    m = c["cr"].replay(z, t, -1.0, 0.0, 1.0, 1.0)
    return float(m["under_crossfade"]), f"{m['changes']} changes, median gap " \
                                        f"{m['median_gap']:.1f} s"


def claim_ladder_margin_responds() -> tuple[float, str]:
    """
    Prompt changes at the deployed settings with ladder_margin 0.25.

    Asserted because the interesting failure is ZERO. build_prompt used to derive the
    previous rung from the previous PROMPT, which latched the controller on the goal
    rung and produced no changes at all across a whole session. A count above zero is
    the property that broke; the exact value is secondary.
    """
    c = _replay_ctx()
    z, t = _replay_z(c["cr"].DEPLOYED)
    m = c["cr"].replay(z, t, -1.0, 0.25, 0.0, 1.0)
    return float(m["changes"]), f"median gap {m['median_gap']:.1f} s over {z.size} windows"


CLAIMS = {
    "T4 best realtime factor":            (claim_t4_best_realtime,        1.05,   0.01),
    "T4 fp32<fp16-half<fp16 consistency": (claim_t4_precision_ordering,   1.00,   0.001),
    "T4 fp16-half vs fp32 at 8 s":        (claim_t4_fp16half_vs_fp32,     0.952,  0.01),
    "T4 between-run max/min":             (claim_t4_between_run_variance, 1.18,   0.02),
    "PILOT01 lag-1 autocorrelation":      (claim_pilot_autocorrelation,   0.953,  0.005),
    "PILOT01 effective sample size":      (claim_pilot_effective_n,       25.3,   0.5),
    "PILOT01 intervention rejection":     (claim_pilot_rejection,         0.131,  0.005),
    "prompt changes before the fix":      (claim_chatter_before,          491,    1),
    "prompt changes after the fix":       (claim_chatter_after,           24,     2),
    "library clipping bound":             (claim_library_clipping_bound,  0.980,  0.005),
    "renders on the dominant prompt":     (claim_library_dominant_variants, 32,   0),
    "end-to-end budget, library":         (claim_latency_budget,          6.5,    0.05),
    "coupling recovers a +6 s lag":       (claim_coupling_recovers_lag,   6.0,    1.0),
    "eyes-closed alpha ratio":            (claim_alpha_validation_ratio,  2.13,   0.02),
    "same effect on AF7/AF8":             (claim_channel_mismatch_af,     0.91,   0.03),
    "DEAP arousal rho (AF3/AF4)":         (claim_deap_arousal_rho,        0.303,  0.02),
    "streaming estimator budget":         (claim_streaming_latency_budget, 0.413, 0.01),
    "replay fidelity vs the log":         (claim_replay_fidelity,         0.991,  0.005),
    "retuned, no dwell: inside a xfade":  (claim_retuned_chatter_no_dwell, 136,   4),
    "retuned, 1 s dwell: inside a xfade": (claim_retuned_chatter_with_dwell, 0,   0),
    "ladder margin 0.25 still responds":  (claim_ladder_margin_responds,  8,      2),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every number the preprint cites")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("CLAIM VERIFICATION - every headline number, regenerated from artefacts")
    print("=" * 78)
    print(f"  {'claim':<36}{'asserted':>10}{'measured':>11}  status")
    print("  " + "-" * 74)

    failures = []
    for name, (fn, asserted, tol) in CLAIMS.items():
        try:
            measured, source = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<36}{asserted:>10.3f}{'ERROR':>11}  {type(exc).__name__}")
            failures.append(name)
            continue
        ok = np.isfinite(measured) and abs(measured - asserted) <= tol
        print(f"  {name:<36}{asserted:>10.3f}{measured:>11.3f}  "
              f"{'ok' if ok else 'MOVED'}")
        if args.verbose:
            print(f"  {'':<36}{'':<21}  source: {source}")
        if not ok:
            failures.append(name)

    print()
    print("=" * 78)
    if failures:
        print(f"  {len(failures)} CLAIM(S) MOVED - the manuscript no longer matches the code:")
        for name in failures:
            print(f"    {name}")
        print("  Update the manuscript, or find out why the number changed. Do not round.")
    else:
        print(f"  All {len(CLAIMS)} claims reproduce. The manuscript matches the repository.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
