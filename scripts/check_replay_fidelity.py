"""check_replay_fidelity.py - did the sham actually replay what it was supposed to?

THE OTHER HALF OF SHAM VALIDITY

`residual_contingency.py` asks whether the sham is decoupled from the participant.
This asks the complementary question: whether it is faithfully coupled to its SOURCE.
A yoked sham is only acoustically matched to the adaptive arm if it reproduces the
source's prompt-decision timeline. Both must hold. Neither implies the other.

This failure mode is not hypothetical here. A pre-fix build normalised replay offsets
against the source's first AUDIO event, while prompts are decided at WINDOW boundaries.
The two timelines have different origins, separated by however long the engine takes to
produce its first segment - 7.06 s on PILOT01, on a loop whose entire latency budget is
6.5 s. Every replayed prompt landed most of a segment early, and a prompt superseded
before the first segment was logged vanished from the replay entirely.

That was found by hand and fixed. The docstring in `live_music._load_yoked_prompts`
records it as "verified to 0.00 s against PILOT01's 492 changes" - a one-off manual
check. This turns it into a standing assertion that runs on every sham session, because
a fidelity bug that returns silently is worth exactly as much as one that was never
fixed.

WHAT IT CHECKS

Given a sham session, recover its source from the manifest's `yoked_from`, rebuild what
the replay was supposed to deliver, and compare against what the sham actually logged:

    count      same number of prompt changes
    sequence   same prompts in the same order
    origin     offset at the first change. In a real sham this is where a wrong
               anchor shows up, because the sham's deliveries are shifted bodily
               against the intended schedule.
    drift      per-change offset error, median and worst case

    A note on which check fires. In the self-test both schedules are normalised to
    their own first element, so the origin difference is 0 by construction and the
    bug surfaces as DRIFT instead (-7.06 s worst, -7.04 s median). That is an
    artefact of reconstructing the bug from one session rather than replaying it
    into another. Do not read "ORIGIN did not fire" as "the anchor was right".

Tolerance is the analysis hop. The replay is reconstructed on the source's hop grid, so
it cannot be finer-grained than that; sub-hop differences are a known and accepted
resolution limit, not a defect.

USAGE

    python scripts/check_replay_fidelity.py sessions/P02_sham_...     check one sham
    python scripts/check_replay_fidelity.py --all                     every sham found
    python scripts/check_replay_fidelity.py --self-test               validate the check

The self-test rebuilds PILOT01's schedule the OLD way and confirms this script flags it.
A checker that has never been shown to fail on known-bad input is not evidence.

Exit code is non-zero if any session fails, so this can gate an analysis run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from live_music import _load_yoked_prompts, load_session  # noqa: E402

DEFAULT_HOP_S = 1.0


def manifest_of(session_dir):
    with open(os.path.join(session_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def hop_of(session_dir, default=DEFAULT_HOP_S):
    try:
        return float(manifest_of(session_dir)["feature_config"]["hop_seconds"])
    except Exception:
        return default


def delivered_schedule(session_dir):
    """What a session actually decided, from its own window log: prompt changes during
    intervention, offset from its first intervention window. Same construction the
    replay loader uses on the source, so like is compared with like."""
    data = load_session(session_dir)
    windows = [w for w in data["windows"]
               if w.get("phase") == "intervention" and w.get("prompt")]
    out, last, t0 = [], None, None
    for w in windows:
        if t0 is None:
            t0 = float(w["elapsed_s"])
        if w["prompt"] != last:
            out.append((float(w["elapsed_s"]) - t0, w["prompt"]))
            last = w["prompt"]
    return out


def buggy_schedule(session_dir):
    """The PRE-FIX construction, kept only so the self-test has known-bad input:
    anchored on the first audio event instead of the first intervention window."""
    data = load_session(session_dir)
    out, t0 = [], None
    for e in data["audio"]:
        if t0 is None:
            t0 = float(e["elapsed_s"])
        out.append((float(e["elapsed_s"]) - t0, e["prompt"]))
    dedup, last = [], None
    for t, p in out:
        if p != last:
            dedup.append((t, p))
            last = p
    return dedup


def compare(intended, actual, hop_s, label=""):
    """Returns (ok, findings). Compares two (offset, prompt) schedules."""
    findings = []
    ok = True

    if len(intended) != len(actual):
        ok = False
        findings.append("COUNT   intended %d changes, delivered %d (%+d)"
                        % (len(intended), len(actual), len(actual) - len(intended)))

    n = min(len(intended), len(actual))
    if n == 0:
        return False, findings + ["EMPTY   nothing to compare"]

    mismatched = [i for i in range(n) if intended[i][1] != actual[i][1]]
    if mismatched:
        ok = False
        findings.append("SEQUENCE %d of %d prompts differ (first at index %d)"
                        % (len(mismatched), n, mismatched[0]))

    diffs = [actual[i][0] - intended[i][0] for i in range(n)]
    origin = diffs[0]
    med = sorted(diffs)[len(diffs) // 2]
    worst = max(diffs, key=abs)

    if abs(origin) > hop_s:
        ok = False
        findings.append("ORIGIN  first change is %+.2f s off (tolerance %.2f s)"
                        % (origin, hop_s))
    if abs(worst) > hop_s:
        ok = False
        findings.append("DRIFT   worst offset error %+.2f s, median %+.2f s "
                        "(tolerance %.2f s)" % (worst, med, hop_s))

    if ok:
        findings.append("origin %+.2f s, median %+.2f s, worst %+.2f s, %d changes "
                        "- within the %.2f s hop" % (origin, med, worst, n, hop_s))
    return ok, findings


def check_session(sham_dir):
    m = manifest_of(sham_dir)
    src = m.get("yoked_from")
    name = os.path.basename(sham_dir.rstrip("/\\"))

    if not src:
        print("%-34s SKIP  no yoked_from in manifest (not a sham)" % name[:34])
        return None
    if not os.path.isdir(src):
        print("%-34s FAIL  source not found: %s" % (name[:34], src))
        return False

    intended = _load_yoked_prompts(src)
    actual = delivered_schedule(sham_dir)
    ok, findings = compare(intended, actual, hop_of(sham_dir))
    print("%-34s %s  (source %s)" % (name[:34], "PASS" if ok else "FAIL",
                                     os.path.basename(src.rstrip("/\\"))))
    for f in findings:
        print("      " + f)
    return ok


def self_test():
    """Show the check fails on the bug it exists to catch."""
    pilots = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT01_*")))
    if not pilots:
        print("self-test needs a PILOT01 session; none found")
        return False
    src = pilots[0]
    hop = hop_of(src)
    correct = _load_yoked_prompts(src)

    print("Self-test on %s (hop %.2f s)\n" % (os.path.basename(src), hop))

    ok, findings = compare(correct, correct, hop)
    print("  identity (correct vs itself)          %s" % ("PASS" if ok else "FAIL"))
    for f in findings:
        print("      " + f)

    bad = buggy_schedule(src)
    ok_bug, findings_bug = compare(correct, bad, hop)
    print("\n  pre-fix audio-anchored schedule       %s"
          % ("PASS (BAD - check is blind)" if ok_bug else "FAIL (correct: caught)"))
    for f in findings_bug:
        print("      " + f)

    good = ok and not ok_bug
    print("\nself-test %s" % ("PASSED - the check accepts correct replays and rejects "
                              "the known bug" if good else
                              "FAILED - do not trust this checker"))
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    dirs = args.sessions
    if args.all or not dirs:
        dirs = [d for d in sorted(glob.glob(os.path.join(_ROOT, "sessions", "*")))
                if os.path.isdir(d)]

    results = [check_session(d) for d in dirs]
    checked = [r for r in results if r is not None]
    if not checked:
        print("\nNo yoked sessions found. Nothing to verify yet.")
        sys.exit(0)
    print("\n%d of %d yoked sessions passed" % (sum(checked), len(checked)))
    sys.exit(0 if all(checked) else 1)


if __name__ == "__main__":
    main()
