"""
run_tests.py - one command that checks everything checkable without hardware.

WHY A RUNNER AND NOT PYTEST

pytest is not in requirements and adding it would mean either pinning a test-only
dependency into an environment that has to stay reproducible for the study, or
maintaining two install paths. The existing validators (verify_library,
validate_coupling) already follow the same shape - print checks, exit non-zero on
failure - so this collects them and adds the regressions that were missing.

WHAT IS COVERED AND WHY EACH ONE EARNED ITS PLACE

Every test here exists because something actually broke, not because it seemed
prudent. In order of the damage the bug would have done:

  CONTROL-LOOP CRASH REPORTING. A NameError killed the worker thread on its first
  intervention window while the finally block still wrote "session complete" - a
  session directory that looked successful and held no data. With a participant in
  the chair you would not find out until analysis.

  PROMPT CHATTER. The trend suffix re-decided every hop from an estimator whose
  noise was 5x its own threshold, producing 477 prompt changes in 20 minutes and
  near-continuous crossfade. Locked at the measured value so a future change to the
  thresholds or the window cannot quietly undo it.

  COUPLING SIGN. The study's headline estimator had its sign convention asserted in
  a docstring and never tested; a real bug in envelope stitching inverted it from
  +6 s to -2 s.

  LIBRARY COVERAGE. If build_prompt emits a prompt the library lacks, the engine
  falls back to a neighbouring rung rather than crashing, so the study runs with the
  wrong music and nothing raises.

  BUILD_PROMPT PURITY. build_library derives the entire prompt space by sweeping
  build_prompt. If it ever becomes stateful, the library silently stops covering
  what the controller can emit.

Usage:
    python scripts/run_tests.py           # everything, no GPU, no headset
    python scripts/run_tests.py --quick   # skip the slow permutation tests
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from music_engine import (  # noqa: E402
    _TREND_ENTER,
    _TREND_EXIT,
    build_prompt,
    state_rung,
)


class Suite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []
        self.skipped: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed.append(name)
        return ok

    def skip(self, name: str, why: str) -> None:
        print(f"  SKIP  {name}  ({why})")
        self.skipped.append(name)

    def section(self, title: str) -> None:
        print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ------------------------------------------------------------------ unit tests


def test_build_prompt_purity(s: Suite) -> None:
    """Same inputs must give the same output - build_library depends on it."""
    cases = [(0.5, -1.0, None, None), (2.0, -1.0, 0.4, None),
             (-2.0, 1.0, -0.4, "flowing melodic electronica, steady gentle rhythm, 85 bpm")]
    stable = all(build_prompt(*c) == build_prompt(*c) == build_prompt(*c) for c in cases)
    s.check("build_prompt is pure", stable, f"{len(cases)} input combinations, 3 calls each")

    # No hidden module state: interleaving different inputs must not change results.
    a1 = build_prompt(2.0, -1.0, 0.4)
    _ = [build_prompt(x, -1.0, 0.4) for x in (-3.0, 0.0, 3.0)]
    s.check("no hidden state between calls", build_prompt(2.0, -1.0, 0.4) == a1)


def test_hysteresis_thresholds(s: Suite) -> None:
    """The band must be a band, and calibrated above the measured noise ceiling."""
    s.check("ENTER above EXIT (a band, not one threshold)", _TREND_ENTER > _TREND_EXIT,
            f"enter {_TREND_ENTER}, exit {_TREND_EXIT}")

    # PILOT01's largest observed 20-hop slope. ENTER sat exactly on it at 0.20,
    # firing zero times by a margin of 0.0002 - luck rather than calibration.
    observed_ceiling = 0.1998
    s.check("ENTER clear of the measured noise ceiling",
            _TREND_ENTER > observed_ceiling * 1.25,
            f"enter {_TREND_ENTER} vs observed max slope {observed_ceiling}")

    base = build_prompt(1.0, -1.0, 0.0)
    s.check("below EXIT asserts nothing", build_prompt(1.0, -1.0, _TREND_EXIT * 0.5) == base)
    s.check("above ENTER asserts a suffix", build_prompt(1.0, -1.0, _TREND_ENTER * 1.5) != base)

    # The band itself: between EXIT and ENTER, hold whatever was already asserted.
    asserted = build_prompt(1.0, -1.0, _TREND_ENTER * 1.5)
    mid = (_TREND_ENTER + _TREND_EXIT) / 2
    s.check("inside the band, a prior assertion is held",
            build_prompt(1.0, -1.0, mid, previous_prompt=asserted) == asserted)
    s.check("inside the band, nothing new is started",
            build_prompt(1.0, -1.0, mid, previous_prompt=None) == base)


def test_chatter_regression(s: Suite, session_z: np.ndarray) -> None:
    """
    Replay real z and assert the prompt does not chatter.

    The threshold is deliberately generous - the measured result was 24 changes and
    this allows 60 - because the point is to catch a return to the old behaviour
    (477 changes, 2.2 s dwell), not to pin an exact number that legitimate tuning
    would break.
    """
    if session_z.size < 200:
        s.skip("chatter regression", "no session with enough intervention windows")
        return

    window = 20
    prompts, prev, hist = [], None, collections.deque(maxlen=window)
    for v in session_z:
        hist.append(v)
        trend = (float(np.polyfit(np.arange(len(hist), dtype=float), np.asarray(hist), 1)[0])
                 if len(hist) >= window else None)
        p = build_prompt(float(v), -1.0, trend, previous_prompt=prev)
        prompts.append(p)
        prev = p

    change_idx = [i for i in range(1, len(prompts)) if prompts[i] != prompts[i - 1]]
    changes = len(change_idx)
    suffix_only = sum(1 for i in change_idx
                      if prompts[i].split(",")[0] == prompts[i - 1].split(",")[0])
    gaps = np.diff(np.asarray(change_idx, dtype=float)) if changes > 1 else np.array([np.inf])
    mean_dwell = len(prompts) / max(1, changes)

    s.check("prompt does not chatter", changes < 60,
            f"{changes} changes over {len(prompts)} windows, mean dwell {mean_dwell:.1f} s")
    s.check("no suffix-only chatter", suffix_only <= 2, f"{suffix_only} suffix-only changes")

    # The safety property, and the one that actually failed in the pilot. Mean dwell
    # is a poor guard because changes cluster - PILOT01 replays to a 43.5 s MEAN and a
    # 4.0 s MEDIAN gap. What must never recur is a switch arriving before the previous
    # crossfade can finish, which is what turns transitions into a continuous blend.
    s.check("no switch faster than the crossfade", float(gaps.min()) >= 1.0,
            f"min gap {gaps.min():.0f} s, median gap {np.median(gaps):.0f} s")


def test_ladder_reachability(s: Suite) -> None:
    """Rungs 0 and 4 are unreachable under both arms - documented, not accidental."""
    reached = set()
    for target in (-1.0, 1.0):
        for z in np.arange(-4.0, 4.05, 0.1):
            for trend in (None, -0.5, 0.0, 0.5):
                p = build_prompt(float(z), target, trend)
                reached.add(p.split(",")[0])
    from music_engine import _ENERGY_LADDER
    idx = {base.split(",")[0]: i for i, base in enumerate(_ENERGY_LADDER)}
    rungs = sorted(idx[r] for r in reached if r in idx)
    s.check("reachable rungs match the documented set", rungs == [1, 2, 3],
            f"reachable: {rungs}; 0 and 4 unreachable by design (see enumerate_prompts.__doc__)")


def test_state_rung_monotonic(s: Suite) -> None:
    zs = np.arange(-5.0, 5.01, 0.05)
    rungs = [state_rung(float(z)) for z in zs]
    s.check("state_rung is monotonic in z", all(b >= a for a, b in zip(rungs, rungs[1:])))
    s.check("state_rung stays in range", min(rungs) == 0 and max(rungs) == 4,
            f"{min(rungs)}..{max(rungs)}")


def test_session_failure_recording(s: Suite) -> None:
    """A crashed worker must record FAILED and must not record complete."""
    import argparse as ap

    import live_music
    from music_engine import build_prompt as real
    from session_logger import SessionLogger

    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:                      # succeed for the initial prompt
            raise RuntimeError("injected fault")
        return real(*a, **k)

    original = live_music.build_prompt
    live_music.build_prompt = flaky
    tmp = tempfile.mkdtemp(prefix="failtest_")
    try:
        args = ap.Namespace(
            participant="RUNTESTS", condition="pilot", target=-1.0, baseline_seconds=20.0,
            duration=0.4, channels="AF7,AF8", reject_p2p=350.0, window=4.0, hop=1.0,
            tau=3.0, segment=8.0, engine="library",
            library=os.path.join(_ROOT, "library"), crossfade=1.0, yoke_from=None,
            mock=True, mock_audio=False, headless=True, out=tmp)
        state = live_music.SessionState(args.target)
        logger = SessionLogger(args.participant, args.condition, root=tmp)
        raised = False
        try:
            live_music.eeg_worker(args, state, logger)
        except RuntimeError:
            raised = True

        msgs = [json.loads(line).get("message")
                for line in open(os.path.join(logger.dir, "events.jsonl"), encoding="utf-8")
                if '"note"' in line]
        s.check("crash re-raised to the caller", raised)
        s.check("crash recorded as FAILED", "session FAILED" in msgs, f"notes: {msgs}")
        s.check("crash NOT recorded as complete", "session complete" not in msgs)
        s.check("phase marked failed", state.phase == "failed", f"phase={state.phase}")
    finally:
        live_music.build_prompt = original
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_baseline_abort(s: Suite) -> None:
    """
    A baseline too short to normalize must abort rather than proceed.

    Without a usable baseline there are no z-scores, so the intervention would be
    steering on an uncalibrated index. This is the one failure the code was already
    handling correctly; it is here so a refactor cannot quietly remove it.
    """
    import argparse as ap

    import live_music
    from session_logger import SessionLogger

    tmp = tempfile.mkdtemp(prefix="baselinetest_")
    try:
        args = ap.Namespace(
            participant="RUNTESTS", condition="pilot", target=-1.0, baseline_seconds=8.0,
            duration=0.3, channels="AF7,AF8", reject_p2p=350.0, window=4.0, hop=1.0,
            tau=3.0, segment=8.0, engine="library",
            library=os.path.join(_ROOT, "library"), crossfade=1.0, yoke_from=None,
            mock=True, mock_audio=False, headless=True, out=tmp)
        state = live_music.SessionState(args.target)
        logger = SessionLogger(args.participant, args.condition, root=tmp)
        live_music.eeg_worker(args, state, logger)

        msgs = [json.loads(line).get("message")
                for line in open(os.path.join(logger.dir, "events.jsonl"), encoding="utf-8")
                if '"note"' in line]
        s.check("short baseline aborts the session",
                any("baseline failed" in (m or "") for m in msgs), f"notes: {msgs}")
        s.check("aborted baseline never reaches intervention",
                state.phase == "baseline failed", f"phase={state.phase}")
        s.check("aborted baseline NOT marked complete", "session complete" not in msgs)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_analysis_runs(s: Suite) -> None:
    """analyze_session must complete on every session on disk."""
    from analyze_session import report

    dirs = [d for d in sorted(glob.glob(os.path.join(_ROOT, "sessions", "*")))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "events.jsonl"))]
    if not dirs:
        s.skip("analysis over real sessions", "no sessions on disk")
        return
    ok, failures = 0, []
    for d in dirs:
        try:
            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                report(d, skip_aci=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{os.path.basename(d)}: {type(exc).__name__}")
    s.check("analyze_session completes on all sessions", not failures,
            f"{ok}/{len(dirs)} ok" + (f", failures: {failures}" if failures else ""))


# ------------------------------------------------------- external validators


def run_validator(s: Suite, name: str, script: str, args: list[str]) -> None:
    cmd = [sys.executable, os.path.join(_ROOT, "scripts", script), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
    s.check(name, proc.returncode == 0, tail[0].strip()[:60])


def load_session_z() -> np.ndarray:
    from session_logger import load_session
    best = np.array([], dtype=float)
    for d in sorted(glob.glob(os.path.join(_ROOT, "sessions", "*"))):
        if not os.path.isdir(d):
            continue
        try:
            sess = load_session(d)
        except Exception:  # noqa: BLE001
            continue
        z = np.array([w["z"] for w in sess["windows"]
                      if w.get("phase") == "intervention" and w.get("valid")
                      and isinstance(w.get("z"), (int, float)) and np.isfinite(w["z"])],
                     dtype=float)
        if z.size > best.size:
            best = z
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every hardware-free check")
    parser.add_argument("--quick", action="store_true",
                        help="skip the slow permutation-based coupling validation")
    args = parser.parse_args()

    s = Suite()

    s.section("1. CONTROLLER - purity, hysteresis, and the chatter regression")
    test_build_prompt_purity(s)
    test_hysteresis_thresholds(s)
    test_state_rung_monotonic(s)
    test_ladder_reachability(s)
    test_chatter_regression(s, load_session_z())

    s.section("2. SESSION - a crash must never look like a success")
    test_session_failure_recording(s)
    test_baseline_abort(s)

    s.section("3. ANALYSIS - the pipeline must complete on real sessions")
    test_analysis_runs(s)

    s.section("4. VALIDATORS - library coverage and the coupling estimator")
    run_validator(s, "library coverage and mixing", "verify_library.py", ["--synthetic"])
    if args.quick:
        s.skip("coupling ground truth", "--quick")
    else:
        run_validator(s, "coupling ground truth", "validate_coupling.py", [])

    print(f"\n{'=' * 74}")
    print(f"  {s.passed} passed, {len(s.failed)} failed, {len(s.skipped)} skipped")
    for name in s.failed:
        print(f"    FAILED: {name}")
    print("=" * 74)
    return 1 if s.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
