"""
ladder_policy.py - decision support for the unreachable ladder rungs.

THE PROBLEM

_ENERGY_LADDER has five rungs, and build_prompt can only ever emit three of them.
level is either goal or here +/- 1, and goal = state_rung(target_z) is rung 1 for the
relaxation arm and rung 3 for the focus arm. Reaching rung 0 needs goal == 0, rung 4
needs goal == 4, and neither arm produces those. So the sparsest and most energetic
prompts are dead code.

The sharper consequence is not the wasted prompts. It is that the music NEVER MATCHES
the participant's current state - it always plays one rung toward the target. A
participant at rung 4 hears rung 3 from the first segment onward.

Whether that is correct is a therapeutic question, not an engineering one. The
iso-principle as usually stated is "match the patient's current state, THEN lead them
gradually". The current code only ever leads. Two defensible readings:

  ALWAYS-LEAD (current). Being one rung away is close enough to feel matched while
  still pulling toward the target. Never playing 100 bpm driving music to an anxious
  participant is a feature, not a gap.

  MATCH-THEN-LEAD. Classic practice matches first. Dropping an aroused listener
  straight to music below their state is the failure mode the iso-principle exists to
  avoid, and one rung may be enough of a mismatch to matter.

This script does not answer that. It quantifies what each candidate policy would
actually do against REAL measured z, so the decision is made against occupancy
numbers rather than intuition.

Usage:
    python scripts/ladder_policy.py
    python scripts/ladder_policy.py --sessions sessions/PILOT01_*
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from music_engine import _ENERGY_LADDER, _DEADBAND_Z, _LADDER_CENTRE, state_rung  # noqa: E402
from session_logger import load_session  # noqa: E402

N_RUNGS = len(_ENERGY_LADDER)


# --------------------------------------------------------------- the policies


def policy_always_lead(z: float, target_z: float, first: bool = False) -> int:
    """Current behaviour. Always move exactly one rung toward the target."""
    here, goal = state_rung(z), state_rung(target_z)
    if abs(z - target_z) <= _DEADBAND_Z or here == goal:
        level = goal
    else:
        level = here + (1 if goal > here else -1)
    return int(np.clip(level, 0, N_RUNGS - 1))


def policy_match_then_lead(z: float, target_z: float, first: bool = False) -> int:
    """
    Match on the first segment, lead thereafter.

    The minimal change that makes the extremes reachable: a participant arriving at
    rung 4 hears rung 4 once, then is led down. Everything after the first segment is
    identical to the current policy.
    """
    if first:
        return int(np.clip(state_rung(z), 0, N_RUNGS - 1))
    return policy_always_lead(z, target_z)


def policy_match_when_far(z: float, target_z: float, first: bool = False) -> int:
    """
    Match whenever the participant is more than 2 rungs from the target, else lead.

    Rationale: the mismatch the iso-principle warns about is worst when the gap is
    large. Close to target, leading is safe; far from it, meet them first. Unlike
    match_then_lead this re-matches if they spike mid-session.
    """
    here, goal = state_rung(z), state_rung(target_z)
    if abs(here - goal) > 2:
        return int(np.clip(here, 0, N_RUNGS - 1))
    return policy_always_lead(z, target_z)


def policy_proportional(z: float, target_z: float, first: bool = False) -> int:
    """
    Lead by a step proportional to the error, capped at 2 rungs.

    Keeps the always-lead spirit but lets a large error produce a larger step, which
    incidentally makes the extremes reachable from further away.
    """
    here, goal = state_rung(z), state_rung(target_z)
    if abs(z - target_z) <= _DEADBAND_Z or here == goal:
        return int(np.clip(goal, 0, N_RUNGS - 1))
    step = int(np.clip(round(abs(z - target_z) / 2.0), 1, 2))
    level = here + (step if goal > here else -step)
    # Never overshoot past the goal.
    level = max(level, goal) if goal < here else min(level, goal)
    return int(np.clip(level, 0, N_RUNGS - 1))


POLICIES = {
    "always-lead (current)": policy_always_lead,
    "match-then-lead": policy_match_then_lead,
    "match-when-far": policy_match_when_far,
    "proportional (cap 2)": policy_proportional,
}


# ------------------------------------------------------------------- analysis


def load_z(patterns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Real measured z from the intervention phase of whatever sessions exist."""
    z, used = [], []
    for pattern in patterns:
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d):
                continue
            try:
                s = load_session(d)
            except Exception:  # noqa: BLE001
                continue
            vals = [w["z"] for w in s["windows"]
                    if w.get("phase") == "intervention" and w.get("valid")
                    and isinstance(w.get("z"), (int, float)) and math.isfinite(w["z"])]
            if vals:
                z.extend(vals)
                used.append(f"{os.path.basename(d)} ({len(vals)})")
    return np.asarray(z, dtype=float), used


def occupancy(policy, z: np.ndarray, target_z: float) -> np.ndarray:
    counts = np.zeros(N_RUNGS)
    for i, value in enumerate(z):
        counts[policy(float(value), target_z, first=(i == 0))] += 1
    return counts / max(1, counts.sum()) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare energy-ladder policies")
    parser.add_argument("--sessions", nargs="+",
                        default=["sessions/*"],
                        help="session dirs or globs to draw real z from")
    parser.add_argument("--target", type=float, default=-1.0)
    args = parser.parse_args()

    z, used = load_z(args.sessions)

    print("=" * 78)
    print("LADDER POLICY COMPARISON")
    print("=" * 78)
    if z.size:
        print(f"  real z from: {', '.join(used)}")
        print(f"  n={z.size}  mean={z.mean():+.2f}  sd={z.std():.2f}  "
              f"range {z.min():+.2f} to {z.max():+.2f}")
    else:
        print("  no real intervention z found; using a synthetic N(0, 1.5) stand-in")
        z = np.random.default_rng(0).normal(0.0, 1.5, 2000)
    print(f"  target_z = {args.target:+.1f}  (goal rung = {state_rung(args.target)})")
    print()

    print("  Where the participant actually sat (state_rung of measured z):")
    here_counts = np.zeros(N_RUNGS)
    for v in z:
        here_counts[state_rung(float(v))] += 1
    here_pct = here_counts / here_counts.sum() * 100
    for r in range(N_RUNGS):
        bar = "#" * int(here_pct[r] / 2)
        print(f"    rung {r}  {here_pct[r]:5.1f}%  {bar}")
    print()

    print("  What each policy would PLAY:")
    print(f"    {'policy':<24}" + "".join(f"{'rung ' + str(r):>9}" for r in range(N_RUNGS))
          + f"{'rungs used':>12}")
    print("    " + "-" * (24 + 9 * N_RUNGS + 12))
    for name, fn in POLICIES.items():
        pct = occupancy(fn, z, args.target)
        used_n = int((pct > 0).sum())
        print(f"    {name:<24}" + "".join(f"{p:8.1f}%" for p in pct) + f"{used_n:>9} / {N_RUNGS}")
    print()

    print("  How often the music MATCHES the participant's own rung:")
    for name, fn in POLICIES.items():
        match = sum(1 for i, v in enumerate(z)
                    if fn(float(v), args.target, first=(i == 0)) == state_rung(float(v)))
        print(f"    {name:<24} {match / z.size * 100:5.1f}%")
    print()

    print("  Largest gap ever opened between music and participant (rungs):")
    for name, fn in POLICIES.items():
        gaps = [abs(fn(float(v), args.target, first=(i == 0)) - state_rung(float(v)))
                for i, v in enumerate(z)]
        print(f"    {name:<24} max {max(gaps)}   mean {np.mean(gaps):.2f}")

    print()
    print("=" * 78)
    print("  The choice is therapeutic, not technical. What this can tell you:")
    print("  - if rungs 0/4 stay near 0% under every policy, delete them and document")
    print("    a 3-rung ladder rather than claiming five.")
    print("  - if a policy plays rung 4 to an anxious participant, decide whether that")
    print("    is the iso-principle working or a safety problem, and say which in the")
    print("    protocol BEFORE collecting data.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
