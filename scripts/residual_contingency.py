"""residual_contingency.py - validate that a sham session is actually decoupled.

WHY THIS EXISTS

A yoked sham is only an informative comparator if the loss of real-time contingency is
demonstrated rather than assumed. The current methodological standard (Rethinking
control conditions in clinical neurofeedback trials, 2026) asks for exactly one thing:
record the participant's true neural signal while false feedback is delivered, then
quantify how often the delivered feedback coincides with what their own signal would
have produced under the same rules.

This project can meet that standard because every session already logs both streams -
`window` events carry the z score the controller would have seen, `audio_segment`
events carry what was actually played. Nothing new needs recording.

It also has a concrete reason to care. A pre-fix build replayed every sham prompt about
seven seconds early, and self-yoking was rejected in the analysis plan on the argument
that a participant's own schedule may still partially track them across days. Both are
residual-contingency failures. This script measures the thing those arguments are about
instead of reasoning about it.

WHAT IT MEASURES

For each session, replay the controller over the participant's own logged z to get the
counterfactual rung they *would* have been served at each moment. Compare against the
rung actually delivered.

    match rate      fraction of time delivered rung == counterfactual rung
    chance rate     the same, under circular time shifts of the delivered schedule,
                    which preserves the schedule's own structure and rung occupancy
                    while destroying any alignment with this participant's signal
    excess          match - chance, in units of the shift-null SD (a z score)

An adaptive session is the positive control: it must score high, because there the
delivered rung IS the counterfactual rung. A properly cross-yoked sham should sit at
chance. Anything in between is residual contingency, and the excess z says how much.

Reporting `excess` in null-SD units rather than raw percentage points matters: raw match
rate is inflated whenever one rung dominates occupancy, and PILOT01 spent 96% of a
session in rung 1.

USAGE

    python scripts/residual_contingency.py                        all sessions
    python scripts/residual_contingency.py --sessions sessions/PILOT01_*
    python scripts/residual_contingency.py --pair A_dir B_dir     simulate cross-yoking
    python scripts/residual_contingency.py --pair A B --lag -7    reproduce the 7 s bug

NOT PART OF THE FROZEN ANALYSIS PLAN. This is a validity check on the sham, not an
outcome analysis, and it is written before any sham session exists so that it cannot be
tuned to a result. If it is later used to justify excluding a session, that decision
belongs in the deviation log.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from music_engine import _ENERGY_LADDER, build_prompt  # noqa: E402

N_SHIFTS = 200          # circular shifts forming the null
MIN_SHIFT_S = 30.0      # ignore tiny shifts, which stay trivially aligned


def session_target_z(path, default=-1.0):
    """The target the session actually ran with, so the counterfactual matches it."""
    try:
        m = json.load(open(os.path.join(path, "manifest.json"), encoding="utf-8"))
        return float(m.get("target_z", default))
    except Exception:
        return default


def load_session(path):
    """Returns (times, z) for windows with a usable z, and (times, rung) delivered."""
    wt, wz, at, ar = [], [], [], []
    with open(os.path.join(path, "events.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            t = e.get("type")
            if t == "window":
                if e.get("valid") and e.get("z") is not None:
                    wt.append(float(e["elapsed_s"]))
                    wz.append(float(e["z"]))
            elif t == "audio_segment":
                if e.get("rung") is not None:
                    at.append(float(e["elapsed_s"]))
                    ar.append(int(e["rung"]))
    return (np.array(wt), np.array(wz)), (np.array(at), np.array(ar))


def delivered_at(seg_t, seg_rung, query_t, lag_s=0.0):
    """Rung playing at each query time. lag_s < 0 makes the schedule arrive early,
    which is what the pre-fix sham bug did."""
    if len(seg_t) == 0:
        return np.full(len(query_t), -1)
    idx = np.searchsorted(seg_t, query_t - lag_s, side="right") - 1
    out = np.where(idx >= 0, seg_rung[np.clip(idx, 0, len(seg_rung) - 1)], -1)
    return out


def counterfactual_rung(z, target_z=-1.0):
    """What the controller would have played, given this participant's own signal.

    NOT state_rung(z). The controller never plays the participant's current rung - it
    leads by one rung toward the target, or sits at the goal inside the deadband. A
    first version of this script used state_rung and the adaptive positive control
    scored only +1.6 sd, which is what caught the error.

    build_prompt is CALLED rather than reimplemented, and the rung read back off the
    returned string, so this cannot drift from the deployed controller. The trend
    suffix is left at None: it affects only the suffix, never the rung, and it is
    calibrated inert in a normal session anyway.
    """
    out = []
    for v in z:
        p = build_prompt(float(v), target_z=target_z)
        out.append(next(i for i, base in enumerate(_ENERGY_LADDER) if p.startswith(base)))
    return np.array(out)


def score(z_t, z, seg_t, seg_rung, lag_s=0.0, target_z=-1.0):
    want = counterfactual_rung(z, target_z)
    got = delivered_at(seg_t, seg_rung, z_t, lag_s)
    ok = got >= 0
    if ok.sum() < 10:
        return None
    want, got, z_t2 = want[ok], got[ok], z_t[ok]
    match = float((want == got).mean())

    span = z_t2[-1] - z_t2[0]
    null = []
    rng = np.random.default_rng(0)
    for _ in range(N_SHIFTS):
        sh = float(rng.uniform(MIN_SHIFT_S, max(MIN_SHIFT_S * 2, span - MIN_SHIFT_S)))
        g = delivered_at(seg_t, seg_rung, ((z_t2 - z_t2[0] + sh) % span) + z_t2[0], lag_s)
        m = g >= 0
        if m.sum() >= 10:
            null.append(float((want[m] == g[m]).mean()))
    null = np.array(null)
    sd = float(null.std()) if len(null) > 2 else float("nan")
    return {
        "n": int(ok.sum()),
        "match": match,
        "chance": float(null.mean()) if len(null) else float("nan"),
        "null_sd": sd,
        "excess_z": (match - float(null.mean())) / sd if sd and sd > 0 else float("nan"),
        "top_rung_frac": float(np.bincount(want, minlength=5).max() / len(want)),
    }


def report(label, r):
    if r is None:
        print("%-30s  (too little overlap to score)" % label[:30])
        return
    flag = ""
    if np.isfinite(r["excess_z"]):
        flag = "  <-- CONTINGENT" if r["excess_z"] > 3 else ""
    print("%-30s n=%4d  match %5.1f%%  chance %5.1f%%  excess %+6.1f sd"
          "  (top rung %4.1f%%)%s"
          % (label[:30], r["n"], 100 * r["match"], 100 * r["chance"],
             r["excess_z"], 100 * r["top_rung_frac"], flag))


def sensitivity(path, fracs=(0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)):
    """How much residual contingency would this check actually catch?

    Builds schedules that are a mixture: a fraction p of the participant's OWN
    contingent schedule, the rest a surrogate with no relationship to them. p = 0 is a
    perfect sham, p = 1 is a fully contingent one. The smallest p that clears +3 sd is
    the detection floor, and it is what licenses any claim that a real sham is clean -
    "no residual contingency" means nothing without it.
    """
    (zt, z), (st, sr) = load_session(path)
    tz = session_target_z(path)
    want = counterfactual_rung(z, tz)
    true_got = delivered_at(st, sr, zt)
    span = zt[-1] - zt[0]
    surro = delivered_at(st, sr, ((zt - zt[0] + span * 0.5) % span) + zt[0])
    ok = (true_got >= 0) & (surro >= 0)
    want, true_got, surro, zt2 = want[ok], true_got[ok], surro[ok], zt[ok]

    rng = np.random.default_rng(1)
    print("Detection floor: mixtures of the participant's own schedule with a surrogate.")
    print("%-10s%12s%12s%14s" % ("contingent", "match", "chance", "excess"))
    floor = None
    for pfrac in fracs:
        take = rng.random(len(want)) < pfrac
        got = np.where(take, true_got, surro)
        match = float((want == got).mean())
        null = []
        for _ in range(N_SHIFTS):
            sh = float(rng.uniform(MIN_SHIFT_S, max(MIN_SHIFT_S * 2, span - MIN_SHIFT_S)))
            g = np.roll(got, int(sh / max(np.median(np.diff(zt2)), 1e-6)))
            null.append(float((want == g).mean()))
        null = np.array(null)
        ez = (match - null.mean()) / (null.std() + 1e-12)
        if floor is None and ez > 3:
            floor = pfrac
        print("%-10.0f%%%11.1f%%%11.1f%%%13.1f sd" % (100 * pfrac, 100 * match,
                                                      100 * null.mean(), ez))
    print("")
    print("detection floor (first mixture clearing +3 sd): %s"
          % ("%.0f%% contingent" % (100 * floor) if floor is not None else "not reached"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--pair", nargs=2, default=None,
                    help="deliver B's schedule against A's signal (simulated yoking)")
    ap.add_argument("--sensitivity", default=None,
                    help="session dir; report the detection floor")
    ap.add_argument("--lag", type=float, default=0.0,
                    help="seconds; negative makes the schedule arrive early")
    args = ap.parse_args()

    if args.sensitivity:
        sensitivity(args.sensitivity)
        return

    if args.pair:
        a, b = args.pair
        (zt, z), _ = load_session(a)
        _, (st, sr) = load_session(b)
        print("signal from %s, schedule from %s, lag %.1f s\n"
              % (os.path.basename(a.rstrip("/")), os.path.basename(b.rstrip("/")),
                 args.lag))
        report("yoked", score(zt, z, st, sr, args.lag,
                              target_z=session_target_z(a)))
        return

    dirs = args.sessions or sorted(glob.glob(os.path.join(_ROOT, "sessions", "*")))
    print("Residual contingency. An adaptive session is the POSITIVE control and must")
    print("score high; a valid cross-yoked sham should sit near zero excess.\n")
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            (zt, z), (st, sr) = load_session(d)
        except FileNotFoundError:
            continue
        if len(zt) < 10 or len(st) < 2:
            print("%-30s  (no usable z or schedule)" % os.path.basename(d)[:30])
            continue
        report(os.path.basename(d), score(zt, z, st, sr,
                                          target_z=session_target_z(d)))


if __name__ == "__main__":
    main()
