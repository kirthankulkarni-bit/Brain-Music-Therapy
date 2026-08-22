"""
power_analysis.py - how many participants, given what the pilot actually measured.

THE FINDING THAT DRIVES EVERYTHING HERE

PILOT01 logged 1043 valid intervention windows. Its lag-1 autocorrelation is 0.953
and its decorrelation time is 9 s, so the AR(1) effective sample size is 25.3.

A twenty-minute session at a 1 s hop looks like 1200 observations and behaves like
about 25.

Any analysis that treats windows as independent - a t-test over windows, a
correlation p-value, a binomial interval on time-in-band - overstates its evidence
by roughly sqrt(1043/25.3), a factor of 6.4. That is the difference between p = 0.05
and p = 0.4. It is the single easiest way to publish a result that does not
replicate, and it is invisible unless you look for it.

So power here is simulated rather than solved in closed form. Sessions are generated
as AR(1) processes with the pilot's measured rho and sd, participants carry their own
random offset, and the whole study is run thousands of times to count how often it
detects the effect. Nothing assumes independence anywhere.

WHAT IS ASSUMED, AND WHAT THAT MEANS

Within-session structure comes from the pilot and is therefore measured. BETWEEN-
PARTICIPANT variance does not - one participant cannot estimate it - so it is swept
across a plausible range instead of guessed at. Read the table as "if between-
participant SD turns out to be X, you need N", not as a single answer. Narrowing that
range is the main thing the first two or three real participants buy you.

DESIGNS

  independent   different people in each arm. The yoked sham replays participant A's
                schedule to participant B, so this is the design the current code
                implements.
  paired        each participant does both arms on different days, yoked to their own
                earlier session. Removes between-participant variance entirely and is
                dramatically cheaper in n - at the cost of order effects and twice the
                sessions per person.

Usage:
    python scripts/power_analysis.py
    python scripts/power_analysis.py --session sessions/PILOT01_20260822_153652
    python scripts/power_analysis.py --sims 5000
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from session_logger import load_session  # noqa: E402

TARGET_BAND = 0.5
ALPHA = 0.05


def session_stats(session_dir: str) -> dict:
    """Measure the within-session structure the simulation needs."""
    session = load_session(session_dir)
    z = np.asarray([w["z"] for w in session["windows"]
                    if w.get("phase") == "intervention" and w.get("valid")
                    and isinstance(w.get("z"), (int, float))], dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 100:
        raise ValueError(f"{session_dir}: only {z.size} valid intervention windows")

    rho = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    sd = float(z.std(ddof=1))
    n_eff = z.size * (1 - rho) / (1 + rho)
    hop = session["manifest"].get("feature_config", {}).get("hop_seconds", 1.0)
    return {"dir": session_dir, "n_windows": int(z.size), "rho": rho, "sd": sd,
            "n_effective": float(n_eff), "hop": hop,
            "target_z": session["manifest"].get("target_z", -1.0)}


def simulate_sessions(rng, means: np.ndarray, rho: float, sd: float, n: int) -> np.ndarray:
    """
    Many sessions at once as AR(1) processes with the measured autocorrelation.

    Vectorised via lfilter rather than a Python loop: the naive version generated one
    sample at a time for every participant in every simulated study, which put a
    single run of this script past ten minutes. The innovation sd is scaled so the
    stationary sd matches the pilot's, which is what makes a simulated session as
    noisy as a real one rather than smoother.
    """
    from scipy.signal import lfilter

    k = means.size
    innovation = sd * np.sqrt(1.0 - rho ** 2)
    noise = rng.normal(0.0, innovation, size=(k, n))
    noise[:, 0] = rng.normal(0.0, sd, size=k)          # start stationary
    series = lfilter([1.0], [1.0, -rho], noise, axis=1)
    return series + means.reshape(-1, 1)


def session_means(rng, means: np.ndarray, rho: float, sd: float, n: int) -> np.ndarray:
    """
    Session means drawn analytically instead of simulated.

    The mean of an AR(1) run has variance sd^2/n * (1+rho)/(1-rho), i.e. sd^2 over the
    effective sample size, so for a linear outcome the whole time series never needs
    generating. Exact, and thousands of times faster. time_in_band is a nonlinear
    function of the series and still needs the full simulation.
    """
    n_eff = n * (1.0 - rho) / (1.0 + rho)
    return rng.normal(means, sd / np.sqrt(n_eff))


def run_study(rng, n_per_arm: int, effect: float, between_sd: float, stats: dict,
              paired: bool, outcome: str) -> bool:
    """One simulated study. Returns True if it detects the effect at ALPHA."""
    from scipy import stats as sps

    def arm(shift: float, participant_offsets: np.ndarray) -> np.ndarray:
        means = stats["target_z"] + shift + participant_offsets
        if outcome == "time_in_band":
            z = simulate_sessions(rng, means, stats["rho"], stats["sd"], stats["n_windows"])
            return (np.abs(z - stats["target_z"]) <= TARGET_BAND).mean(axis=1)
        return session_means(rng, means, stats["rho"], stats["sd"], stats["n_windows"])

    if paired:
        # Same people in both arms: their offsets cancel in the difference.
        offsets = rng.normal(0.0, between_sd, n_per_arm)
        a, b = arm(0.0, offsets), arm(effect, offsets)
        t, p = sps.ttest_rel(a, b)
    else:
        a = arm(0.0, rng.normal(0.0, between_sd, n_per_arm))
        b = arm(effect, rng.normal(0.0, between_sd, n_per_arm))
        t, p = sps.ttest_ind(a, b)
    return bool(p < ALPHA)


def power_for(rng, n_per_arm: int, effect: float, between_sd: float, stats: dict,
              paired: bool, outcome: str, sims: int) -> float:
    hits = sum(run_study(rng, n_per_arm, effect, between_sd, stats, paired, outcome)
               for _ in range(sims))
    return hits / sims


def required_n(rng, effect: float, between_sd: float, stats: dict, paired: bool,
               outcome: str, sims: int, target_power: float = 0.80,
               max_n: int = 60) -> int:
    """Smallest n per arm reaching target_power. Returns max_n+1 if unreachable."""
    for n in (4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60):
        if n > max_n:
            break
        if power_for(rng, n, effect, between_sd, stats, paired, outcome, sims) >= target_power:
            return n
    return max_n + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulation-based power analysis")
    parser.add_argument("--session", default=None, help="session to take structure from")
    parser.add_argument("--sims", type=int, default=400,
                        help="simulated studies per cell; raise for smoother estimates")
    parser.add_argument("--outcome", default="z_mean", choices=["z_mean", "time_in_band"])
    args = parser.parse_args()

    session_dir = args.session
    if session_dir is None:
        candidates = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*")))
        if not candidates:
            candidates = [d for d in sorted(glob.glob(os.path.join(_ROOT, "sessions", "*")))
                          if os.path.isdir(d)]
        session_dir = candidates[-1]

    stats = session_stats(session_dir)
    rng = np.random.default_rng(0)

    print("=" * 78)
    print("POWER ANALYSIS - simulated, using the pilot's measured autocorrelation")
    print("=" * 78)
    print(f"  structure from : {os.path.basename(stats['dir'])}")
    print(f"  windows        : {stats['n_windows']} at a {stats['hop']:g} s hop")
    print(f"  lag-1 autocorr : {stats['rho']:.3f}")
    print(f"  within-session sd of z : {stats['sd']:.3f}")
    print()
    print(f"  EFFECTIVE SAMPLE SIZE  : {stats['n_effective']:.1f}")
    print(f"  A {stats['n_windows'] * stats['hop'] / 60:.0f}-minute session looks like "
          f"{stats['n_windows']} observations and behaves like {stats['n_effective']:.0f}.")
    print(f"  Treating windows as independent overstates evidence by "
          f"{np.sqrt(stats['n_windows'] / stats['n_effective']):.1f}x.")
    print()
    print(f"  outcome        : {args.outcome}")
    print(f"  alpha {ALPHA}, target power 0.80, {args.sims} simulated studies per cell")
    print()

    effects = [0.2, 0.3, 0.5, 0.8]
    between_sds = [0.3, 0.5, 0.7]

    for paired in (False, True):
        label = "PAIRED (crossover, each participant does both arms)" if paired else \
                "INDEPENDENT (different people per arm - what the code implements)"
        print("-" * 78)
        print(f"  {label}")
        print("-" * 78)
        print(f"    {'effect (z)':>11}" + "".join(f"{'bSD ' + str(b):>12}" for b in between_sds))
        for effect in effects:
            row = f"    {effect:>11.1f}"
            for bsd in between_sds:
                n = required_n(rng, effect, bsd, stats, paired, args.outcome, args.sims)
                row += f"{('>60' if n > 60 else str(n)):>12}"
            print(row)
        print()

    print("=" * 78)
    print("  n is PER ARM, for 80% power at alpha 0.05.")
    print("  bSD is the assumed between-participant SD of the outcome, which the pilot")
    print("  cannot estimate from one person - that is what the first few participants buy.")
    print()
    print("  The paired design is far cheaper in n because it removes between-participant")
    print("  variance entirely. It costs two sessions per person and introduces order")
    print("  effects, which need counterbalancing. The yoked sham as currently written")
    print("  replays one participant's schedule to a DIFFERENT participant, so switching")
    print("  to paired means yoking each person to their own earlier session.")
    print()
    print("  CAVEAT ON THE PAIRED COLUMN: it does not move with bSD because participant")
    print("  offsets cancel EXACTLY in this simulation. Real crossover data has a")
    print("  participant-by-condition interaction - some people respond and some do not -")
    print("  which this does not model, so treat the paired numbers as a floor rather")
    print("  than an estimate. Inflate them if you have any reason to expect")
    print("  heterogeneous response.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
