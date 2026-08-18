"""
validate_coupling.py - prove the coupling index recovers a lag it is given.

WHY THIS EXISTS

coupling_index() produces the study's headline number: the lag at which generated
audio and alpha power couple most strongly, and the claim that it is positive with
music leading brain. Everything the adaptive-vs-sham contrast rests on is downstream
of that one estimate.

Until now its sign convention lived only in a docstring. "Positive lag = audio leads
brain" was asserted, never demonstrated, and a sign error there would not look like a
bug - it would look like a clean result pointing the wrong way, and it would survive
review because the number is plausible either way.

That risk is not hypothetical. _stitch_audio_envelope had a real defect: the two
engines log envelopes with opposite time semantics (streaming logs audio about to
play, the library logs audio already heard), and anchoring both as "starts now"
shifted the entire audio timeline by one segment tenure. It produced no error, no
warning, and a biased lag on every session. It was found by reading, not by testing,
which is the wrong way to find something this important.

So: build sessions whose true lag is known by construction, and check the estimator
returns it.

FOUR CHECKS

  1. lag recovery   audio leads brain by a known amount -> estimator must report it
  2. sign           a positive and a negative lag must land on opposite sides of
                    zero, which catches a flip that check 1 alone can pass
  3. null           independent series must NOT come out significant, which is what
                    makes a significant result on real data mean anything
  4. engine parity  the SAME ground truth logged in streaming form and in library
                    retrospective form must give the SAME lag. This is the
                    regression test for the bug above; before the fix these differ
                    by one segment tenure.

Usage:
    python scripts/validate_coupling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from analyze_session import coupling_index  # noqa: E402

HOP = 1.0
SEGMENT_S = 8.0
ENV_RATE = 20.0
DURATION_S = 600.0


def _slow_signal(t: np.ndarray, seed: int) -> np.ndarray:
    """
    A smooth, broadband-but-slow signal, like a real amplitude envelope.

    Deliberately not a single sine: a pure tone makes cross-correlation ambiguous at
    multiples of its period, so a lag test built on one can pass while the estimator
    is picking the wrong peak.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(t)
    for _ in range(6):
        f = rng.uniform(0.005, 0.05)          # 20-200 s periods
        out += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    return out / (out.std() + 1e-12)


def build_session(true_lag_s: float, retrospective: bool, noise: float = 0.6,
                  seed: int = 0, independent: bool = False) -> dict:
    """
    A session whose audio-to-brain lag is known by construction.

    The audio envelope is defined continuously, then chunked into segments and logged
    in whichever engine's format is being tested. Brain alpha is that same envelope
    delayed by true_lag_s, plus noise - so audio LEADS brain by true_lag_s and the
    estimator must report +true_lag_s.
    """
    rng = np.random.default_rng(seed + 7)

    env_t = np.arange(0.0, DURATION_S, 1.0 / ENV_RATE)
    env = _slow_signal(env_t, seed)
    audio_env = env - env.min() + 0.1          # envelopes are non-negative

    brain_t = np.arange(0.0, DURATION_S, HOP)
    if independent:
        driver = _slow_signal(brain_t, seed + 999)
    else:
        # alpha(t) = audio(t - lag): brain repeats what the audio did lag seconds ago.
        driver = np.interp(brain_t - true_lag_s, env_t, env, left=np.nan, right=np.nan)
    alpha_log = driver + noise * rng.standard_normal(brain_t.size)

    windows = [
        {"elapsed_s": float(bt), "phase": "intervention", "valid": True,
         "alpha": float(10.0 ** a)}                     # coupling_index takes log10
        for bt, a in zip(brain_t, alpha_log) if np.isfinite(a)
    ]

    per_segment = int(SEGMENT_S * ENV_RATE)
    audio = []
    for start in range(0, audio_env.size - per_segment + 1, per_segment):
        chunk = audio_env[start:start + per_segment]
        t_start = start / ENV_RATE
        if retrospective:
            # library_engine: elapsed_s marks the END of audio already heard.
            audio.append({"elapsed_s": t_start + SEGMENT_S, "envelope": chunk.tolist(),
                          "envelope_rate_hz": ENV_RATE, "envelope_retrospective": True})
        else:
            # streaming: elapsed_s marks the START of audio about to play.
            audio.append({"elapsed_s": t_start, "envelope": chunk.tolist(),
                          "envelope_rate_hz": ENV_RATE})

    return {"manifest": {"feature_config": {"hop_seconds": HOP}},
            "windows": windows, "audio": audio, "baseline": [], "notes": [], "dir": "synthetic"}


def main() -> int:
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")
        if not ok:
            failures.append(name)

    print("=" * 74)
    print("GROUND-TRUTH VALIDATION OF THE COUPLING INDEX")
    print("=" * 74)
    print(f"  {DURATION_S:.0f} s synthetic sessions, hop {HOP:g} s, "
          f"{SEGMENT_S:g} s segments, envelope {ENV_RATE:g} Hz\n")

    print("1. LAG RECOVERY - the estimator must return the lag it was given")
    for true_lag in (6.0, 3.0, 0.0, -3.0):
        res = coupling_index(build_session(true_lag, retrospective=False, seed=1),
                             n_permutations=200)
        got = res.get("aci_peak_lag_s", float("nan"))
        check(f"true lag {true_lag:+.0f} s recovered",
              abs(got - true_lag) <= HOP,
              f"got {got:+.1f} s, r={res.get('aci_peak_r', float('nan')):+.2f}, "
              f"p={res.get('aci_p_circular_shift', float('nan')):.3f}")

    print("\n2. SIGN CONVENTION - a flip would pass check 1 for symmetric lags only")
    pos = coupling_index(build_session(6.0, retrospective=False, seed=2), n_permutations=200)
    neg = coupling_index(build_session(-3.0, retrospective=False, seed=2), n_permutations=200)
    check("positive and negative lags land on opposite sides of zero",
          pos.get("aci_peak_lag_s", 0) > 0 > neg.get("aci_peak_lag_s", 0),
          f"{pos.get('aci_peak_lag_s'):+.1f} s and {neg.get('aci_peak_lag_s'):+.1f} s")

    print("\n3. NULL - independent series must not come out significant")
    ps = []
    for seed in range(6):
        res = coupling_index(build_session(0.0, retrospective=False, seed=seed, independent=True),
                             n_permutations=200)
        ps.append(res.get("aci_p_circular_shift", float("nan")))
    false_positives = sum(1 for p in ps if p < 0.05)
    check("independent series rarely significant", false_positives <= 1,
          f"{false_positives}/6 runs p<0.05, p values {[round(p, 2) for p in ps]}")

    print("\n4. ENGINE PARITY - same truth, both log formats, same answer")
    print("   (this is the regression test for the envelope time-semantics fix)")
    for true_lag in (6.0, -3.0):
        stream = coupling_index(build_session(true_lag, retrospective=False, seed=3),
                                n_permutations=100)
        libr = coupling_index(build_session(true_lag, retrospective=True, seed=3),
                              n_permutations=100)
        a = stream.get("aci_peak_lag_s", float("nan"))
        b = libr.get("aci_peak_lag_s", float("nan"))
        check(f"streaming and library agree at true lag {true_lag:+.0f} s",
              abs(a - b) <= HOP and abs(a - true_lag) <= HOP,
              f"streaming {a:+.1f} s, library {b:+.1f} s")

    print("\n" + "=" * 74)
    print(f"  {'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' FAILED'}")
    for name in failures:
        print(f"    FAILED: {name}")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
