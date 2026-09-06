"""
validate_event_locked.py - ground truth for the event-locked response estimator.

WHY THIS EXISTS

`analyze_session.event_locked_response` feeds `elr_effect_z` into the study's primary
contrast (`CONTRASTS` in analyze_session.py), where a positive adaptive-minus-sham
difference is read as "adaptive shows more brain-follows-music than sham". Until
2026-09-06 nothing validated it against data whose answer was known.

The coupling index has had that since 8/16, and it earned it the hard way: its sign
convention was asserted in a docstring, never tested, and a real bug in envelope
stitching had inverted it from +6 s to -2 s. The event-locked estimator carried exactly
the same exposure - a signed quantity, a direction-folding step, and a permutation null,
none of it checked against a case with a known answer.

WHAT IS CONSTRUCTED

Four sessions, each with rung changes at known times and z built to a known relationship
with them:

  RESPONDER      z steps the way the music asked after each change, and is otherwise
                 flat. The estimator must report a clearly positive effect.

  ANTI-RESPONDER z steps the OPPOSITE way. The estimator must report a negative effect.
                 This is the half that a sign error breaks, and the half that a
                 "returns positive whenever the loop is closed" implementation passes
                 by accident.

  INDEPENDENT    events at the same times, z an unrelated slow signal. The estimator
                 must report approximately nothing, and the permutation p must not be
                 significant. This is the yoked sham's expected behaviour.

  TRIGGERED      the confound the docstring warns about, made measurable. No audio
                 effect at all: z wanders, and a rung change is EMITTED whenever z
                 crosses a boundary, exactly as the real controller does. The estimator
                 should still report a positive effect, because the event was caused by
                 the excursion that continues after it.

The fourth case is the point. It demonstrates that a positive within-arm effect is not
evidence of an effect, which is the reason the analysis plan refuses to interpret this
number outside the adaptive-minus-sham contrast. Asserting it turns a prose warning into
a property of the code, and it should also show the diagnostic signature the estimator
already reports - a rising pre-window slope, and a curve that peaks at onset.

Usage:
    python scripts/validate_event_locked.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from analyze_session import EVENT_MIN_SEPARATION_S, event_locked_response  # noqa: E402
from music_engine import _ENERGY_LADDER  # noqa: E402

HOP = 1.0
DURATION_S = 900.0
EVENT_EVERY_S = 60.0          # comfortably above EVENT_MIN_SEPARATION_S
RESPONSE_Z = 0.8              # how far z moves when it responds
SETTLE_S = 5.0                # how long the response takes to arrive


def _slow_signal(t: np.ndarray, seed: int) -> np.ndarray:
    """A smooth, autocorrelated wander, so the permutation null is exercised properly."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(t.size)
    k = np.exp(-np.arange(0, 60) / 20.0)
    return np.convolve(x, k / k.sum(), mode="same")


def _session(times: np.ndarray, z: np.ndarray, events: list) -> dict:
    """Package a z series and a list of (onset, rung) into the session dict shape."""
    return {
        "dir": "synthetic",
        "manifest": {"feature_config": {"hop_seconds": HOP}, "target_z": -1.0,
                     "participant_id": "SYN", "condition": "adaptive"},
        "windows": [{"phase": "intervention", "elapsed_s": float(t), "z": float(v),
                     "valid": True, "applied": True}
                    for t, v in zip(times, z)],
        "audio": [{"type": "audio_segment", "elapsed_s": float(onset), "rung": int(rung),
                   "prompt": _ENERGY_LADDER[int(rung)]}
                  for onset, rung in events],
    }


def build(kind: str, seed: int = 0) -> dict:
    """A session whose event-locked answer is known by construction."""
    t = np.arange(0.0, DURATION_S, HOP)
    onsets = np.arange(EVENT_EVERY_S, DURATION_S - 60.0, EVENT_EVERY_S)
    rng = np.random.default_rng(seed)

    if kind == "triggered":
        # No audio effect whatsoever. z wanders; the controller emits a rung change
        # whenever z crosses a half-integer, which is what state_rung actually does.
        z = 2.0 * _slow_signal(t, seed + 3)
        z = (z - z.mean()) / (z.std() or 1.0)
        events, previous, last_t = [], None, -1e9
        for tt, zz in zip(t, z):
            rung = int(np.clip(round(2 + zz), 0, len(_ENERGY_LADDER) - 1))
            if previous is None:
                previous, events = rung, [(float(tt), rung)]
            elif rung != previous and tt - last_t >= EVENT_MIN_SEPARATION_S:
                events.append((float(tt), rung))
                previous, last_t = rung, tt
            else:
                previous = rung
        return _session(t, z, events)

    # Everything else: fixed event times, alternating direction, z built to respond.
    directions = np.where(np.arange(onsets.size) % 2 == 0, +1, -1)
    rungs, rung = [], 2
    for d in directions:
        rung = int(np.clip(rung + d, 1, 3))
        rungs.append(rung)
    events = [(2.0, 2)] + list(zip(onsets.tolist(), rungs))

    z = 0.35 * _slow_signal(t, seed + 11)
    if kind != "independent":
        gain = RESPONSE_Z if kind == "responder" else -RESPONSE_Z
        for onset, d in zip(onsets, directions):
            # A step that arrives over SETTLE_S and then holds until the next event.
            ramp = np.clip((t - onset) / SETTLE_S, 0.0, 1.0)
            hold = (t >= onset) & (t < onset + EVENT_EVERY_S)
            z = z + gain * d * ramp * hold
    else:
        z = z + 0.35 * _slow_signal(t, seed + 77)

    z = z + 0.05 * rng.standard_normal(t.size)
    return _session(t, z, events)


def main() -> int:
    print("=" * 78)
    print("EVENT-LOCKED RESPONSE - ground truth")
    print("=" * 78)
    print("  effect  = z moved the way the music asked, averaged over the post-window")
    print("  pre     = slope of z BEFORE the event; large means the event was triggered")
    print("            by the excursion rather than causing it")
    print()
    print(f"  {'session':<16}{'events':>8}{'effect':>10}{'p':>9}{'pre slope':>12}"
          f"{'peaks at onset':>16}")
    print("  " + "-" * 74)

    results = {}
    for kind in ("responder", "anti-responder", "independent", "triggered"):
        r = event_locked_response(build(kind), n_permutations=300)
        results[kind] = r
        print(f"  {kind:<16}{r.get('elr_n_epochs', 0):>8}"
              f"{r.get('elr_effect_z', float('nan')):>+10.3f}"
              f"{r.get('elr_p_permutation', float('nan')):>9.3f}"
              f"{r.get('elr_pre_slope_z_per_s', float('nan')):>+12.4f}"
              f"{str(r.get('elr_decays_after_onset')):>16}")

    print()
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    resp = results["responder"]["elr_effect_z"]
    anti = results["anti-responder"]["elr_effect_z"]
    indep = results["independent"]["elr_effect_z"]
    trig = results["triggered"]["elr_effect_z"]

    check("a responder gives a positive effect", resp > 0.2, f"{resp:+.3f}")
    check("a responder is significant",
          results["responder"]["elr_p_permutation"] < 0.05,
          f"p = {results['responder']['elr_p_permutation']:.3f}")
    # The sign test. An estimator that reports positive whenever the loop is closed -
    # which is the failure the docstring warns this measure is prone to - passes the
    # responder case and fails here.
    check("an anti-responder gives a NEGATIVE effect", anti < -0.2, f"{anti:+.3f}")
    check("the two are separated by roughly twice the response",
          abs(resp - anti) > RESPONSE_Z, f"{resp:+.3f} vs {anti:+.3f}")
    check("an independent session gives approximately nothing", abs(indep) < 0.2,
          f"{indep:+.3f}")
    check("an independent session is not significant",
          not (results["independent"]["elr_p_permutation"] < 0.05),
          f"p = {results['independent']['elr_p_permutation']:.3f}")

    # The confound, asserted rather than warned about.
    check("a triggered session with NO audio effect still looks positive", trig > 0.2,
          f"{trig:+.3f} - this is why the number is not interpretable within one arm")
    check("and it shows the trigger signature the estimator reports",
          abs(results["triggered"]["elr_pre_slope_z_per_s"]) >
          abs(results["responder"]["elr_pre_slope_z_per_s"]),
          f"pre slope {results['triggered']['elr_pre_slope_z_per_s']:+.4f} vs "
          f"{results['responder']['elr_pre_slope_z_per_s']:+.4f} for a real response")

    print()
    print("=" * 78)
    if all(checks):
        print("  All checks pass. The estimator recovers a known response, reverses with")
        print("  it, stays silent on an unrelated session, and reproduces the trigger")
        print("  confound - so a positive effect in one arm means nothing on its own.")
    else:
        print(f"  {checks.count(False)} CHECK(S) FAILED.")
    print("=" * 78)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
