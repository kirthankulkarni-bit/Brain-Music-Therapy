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
    python scripts/run_tests.py --quick   # skip everything that recomputes from raw data
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
    cases = [(0.5, -1.0, None), (2.0, -1.0, None),
             (-2.0, 1.0, "flowing melodic electronica, steady gentle rhythm, 85 bpm")]
    stable = all(build_prompt(z, t, previous_prompt=pp) ==
                 build_prompt(z, t, previous_prompt=pp) ==
                 build_prompt(z, t, previous_prompt=pp) for z, t, pp in cases)
    s.check("build_prompt is pure", stable, f"{len(cases)} input combinations, 3 calls each")

    # No hidden module state: interleaving different inputs must not change results.
    a1 = build_prompt(2.0, -1.0)
    _ = [build_prompt(x, -1.0) for x in (-3.0, 0.0, 3.0)]
    s.check("no hidden state between calls", build_prompt(2.0, -1.0) == a1)

    # Everything past target_z is keyword-only, so a call written against the old
    # signature raises instead of silently reinterpreting its third argument. When the
    # trend parameter was removed, build_prompt(z, target, trend) would otherwise have
    # passed the trend as previous_prompt and kept running.
    try:
        build_prompt(1.0, -1.0, 0.4)          # type: ignore[misc]
        rejected = False
    except TypeError:
        rejected = True
    s.check("stale positional call is rejected, not reinterpreted", rejected)


def test_trend_is_not_measurable(s: Suite, session_z: np.ndarray) -> None:
    """
    The evidence for removing the trend suffix, asserted rather than remembered.

    This test replaces test_hysteresis_thresholds, which checked that ENTER sat above
    EXIT and above the measured noise ceiling. Those checks passed for two weeks while
    describing a control that could not work: they verified the threshold was placed
    consistently, never that the quantity underneath it was measurable.

    It is not. The slope the suffix gated on is smaller than the noise of the estimator
    measuring it, so a threshold above the noise can only be crossed by noise, and one
    low enough to catch real drift fires constantly - which is the 8/16 defect, 491
    prompt changes in twenty minutes from a 0.05 threshold against 0.275 of noise.

    If a future estimator ever makes the trend measurable, this test fails and the
    suffix becomes worth reconsidering. That is the intended way to reopen the question.
    """
    if session_z.size < 200:
        s.skip("trend is not measurable", "no session with enough windows")
        return

    W = 20
    idx = np.arange(W, dtype=float)
    slopes = np.array([np.polyfit(idx, session_z[i - W:i], 1)[0]
                       for i in range(W, session_z.size)])
    noise = float(slopes.std(ddof=1))

    # The genuine drift, measured over 60 s stretches rather than the whole session,
    # so a real excursion is not averaged away by a flat session.
    win = 60
    real = max(abs(float(np.polyfit(np.arange(win, dtype=float),
                                    session_z[i:i + win], 1)[0]))
               for i in range(0, session_z.size - win, win // 4))

    s.check("trend noise exceeds the largest genuine drift", noise > real,
            f"noise {noise:.4f} vs largest 60 s drift {real:.4f} "
            f"({noise / real:.1f}x)")

    # Every emitted prompt must be EXACTLY a ladder rung - not "starts with" one, which
    # a suffix would still satisfy. Swept over both arms and a previous_prompt, since the
    # suffix used to be reachable only when one was threaded through.
    import music_engine
    rungs = set(music_engine._ENERGY_LADDER)
    emitted = set()
    for target in (-1.0, 1.0):
        prev = None
        for v in np.arange(-4, 4.01, 0.05):
            prev = build_prompt(float(v), target, previous_prompt=prev)
            emitted.add(prev)
    s.check("every prompt is exactly a ladder rung", emitted <= rungs,
            f"{len(emitted)} distinct prompts, all in the {len(rungs)}-rung ladder"
            if emitted <= rungs else f"outside the ladder: {sorted(emitted - rungs)}")

    gone = [n for n in ("_trend_suffix", "_previous_suffix", "_TREND_ENTER",
                        "_TREND_EXIT", "_ALL_SUFFIXES", "_SUFFIX_HOLDING")
            if hasattr(music_engine, n)]
    s.check("the suffix machinery is gone, not just unused", not gone,
            f"still present: {gone}" if gone else "6 names removed")


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

    prompts, prev = [], None
    for v in session_z:
        p = build_prompt(float(v), -1.0, previous_prompt=prev)
        prompts.append(p)
        prev = p

    change_idx = [i for i in range(1, len(prompts)) if prompts[i] != prompts[i - 1]]
    changes = len(change_idx)
    # Every change is now a rung change by construction - there is no suffix left that
    # could change without one. Kept as a check that that stays true.
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


def test_ladder_hysteresis(s: Suite) -> None:
    """
    Opt-in ladder hysteresis: must be inert by default, effective when asked, and must
    not stick.

    The default matters most. This changes what a participant hears, so it is off unless
    explicitly requested, and the check below is byte-identical output across the whole
    z range rather than a spot check.
    """
    from music_engine import state_rung

    identical = all(build_prompt(float(z), -1.0) ==
                    build_prompt(float(z), -1.0, ladder_margin=0.0)
                    for z in np.arange(-4, 4.01, 0.1))
    s.check("ladder hysteresis is inert by default", identical,
            "byte-identical across z in [-4, 4]")

    # z dithering either side of the 2/3 boundary, which is where round() flips.
    dither = [0.45, 0.55, 0.44, 0.58, 0.47, 0.53, 0.46]

    def walk(margin):
        prev, out = None, []
        for z in dither:
            prev = state_rung(z, previous_rung=prev, margin=margin)
            out.append(prev)
        return sum(1 for a, b in zip(out, out[1:]) if a != b)

    s.check("without hysteresis the rung flips on noise", walk(0.0) >= 5,
            f"{walk(0.0)} flips across a boundary")
    s.check("with hysteresis it does not", walk(0.25) == 0, f"{walk(0.25)} flips")

    # It must still follow a genuine excursion, or it would be a stuck controller.
    prev = None
    for z in np.arange(0.0, 3.01, 0.1):
        prev = state_rung(float(z), previous_rung=prev, margin=0.25)
    s.check("hysteresis still tracks a real move", prev == state_rung(3.0),
            f"walked to rung {prev}, plain gives {state_rung(3.0)}")


def test_ladder_hysteresis_does_not_latch(s: Suite, session_z: np.ndarray) -> None:
    """
    The regression for the latch. This is the test that was missing.

    Everything above exercises state_rung in ISOLATION, called correctly - previous_rung
    fed the previous state estimate. build_prompt wired it differently: it passed
    _rung_of(previous_prompt), the rung being PLAYED, which is always one step toward the
    target. That closes a loop, and with margin > 0 the controller latched on the goal
    rung and stopped responding - zero prompt changes across all 1043 windows of PILOT01,
    while the participant's own rung ranged up to 4.

    Every unit test above passed throughout, because none of them drove the whole
    controller with the margin enabled. A test that cannot fail is not a safeguard.
    """
    from music_engine import PromptGovernor

    if session_z.size < 200:
        s.skip("ladder hysteresis does not latch", "no session with enough windows")
        return

    for margin in (0.25, 0.5):
        gov = PromptGovernor(target_z=-1.0, ladder_margin=margin)
        prompts = [gov.update(float(v), now=float(i)) for i, v in enumerate(session_z)]
        changes = sum(1 for a, b in zip(prompts, prompts[1:]) if a != b)
        s.check(f"margin {margin} still responds to the participant", changes > 0,
                f"{changes} changes over {len(prompts)} windows")

    # The sharper version: a large sustained excursion must move the music, whatever
    # the margin. A latched controller passes a change count and fails this.
    for margin in (0.0, 0.25, 0.5):
        gov = PromptGovernor(target_z=-1.0, ladder_margin=margin)
        for _ in range(40):
            gov.update(0.0, now=0.0)
        settled = gov.prompt
        for i in range(60):
            gov.update(3.0, now=float(i))       # far above target, sustained
        s.check(f"margin {margin} follows a sustained excursion", gov.prompt != settled,
                "prompt moved" if gov.prompt != settled else "prompt never moved")


def test_min_dwell(s: Suite) -> None:
    """
    The dwell must bound the RATE of change, and must not make the music stale.

    A dwell of at least one crossfade is the condition for no switch arriving before the
    previous crossfade finishes - which is what turns transitions into a continuous
    blend of two independent renders. Replayed on PILOT01 at the retuned 2 s / 0.5 s /
    tau 0.5 settings, 136 of 382 changes landed inside a crossfade with no dwell, and 0
    with a 1 s dwell. See docs/finding_ladder_hysteresis.md.
    """
    from music_engine import PromptGovernor, build_prompt as bp

    # Default off: identical to calling build_prompt directly, across the z range.
    gov = PromptGovernor(target_z=-1.0)
    same = all(gov.update(float(z), now=float(i)) ==
               bp(float(z), -1.0, previous_prompt=(gov.prompt if i else None))
               for i, z in enumerate(np.arange(-3, 3.01, 0.25)))
    s.check("governor with no margin and no dwell matches build_prompt", same)

    # A dwell of D means no two changes closer than D, by construction.
    dwell, hop = 1.0, 0.25
    z = np.tile([2.5, -2.5], 200)               # maximally adversarial: flip every hop
    gov = PromptGovernor(target_z=-1.0, min_dwell_seconds=dwell)
    times, prev = [], None
    for i, v in enumerate(z):
        now = i * hop
        p = gov.update(float(v), now=now)
        if prev is not None and p != prev:
            times.append(now)
        prev = p
    gaps = np.diff(np.asarray(times)) if len(times) > 1 else np.array([np.inf])
    s.check("dwell bounds the change rate", float(gaps.min()) >= dwell - 1e-9,
            f"{len(times)} changes, min gap {gaps.min():.2f} s against a {dwell:g} s dwell")
    s.check("dwell without a dwell would have chattered",
            sum(1 for a, b in zip(z, z[1:]) if a != b) > len(times),
            "the input alternates every hop")

    # It is a rate limit, NOT a filter. When the dwell expires the governor must adopt
    # what the controller wants NOW. Holding a stale request would make the music lag by
    # up to a full dwell, which is the latency this project exists to reduce.
    gov = PromptGovernor(target_z=-1.0, min_dwell_seconds=10.0)
    gov.update(0.0, now=0.0)
    gov.update(3.0, now=1.0)                    # requested, blocked by the dwell
    gov.update(-3.0, now=1.5)                   # superseded while still blocked
    after = gov.update(-3.0, now=20.0)          # dwell expired
    s.check("dwell adopts the current request, not the stale one",
            after == bp(-3.0, -1.0, previous_prompt=gov.prompt),
            "no queued backlog of superseded prompts")


def test_retuned_estimator_guard(s: Suite) -> None:
    """
    A retuned estimator without a dwell must refuse to start, not warn.

    The resulting recording is not merely noisy - it is disqualified as a yoke source,
    and that is discovered after the participant has gone home. finding_ladder_
    hysteresis.md exists because this exact configuration was recommended in writing.
    """
    import live_music
    from session_logger import SessionLogger

    tmp = tempfile.mkdtemp(prefix="guardtest_")
    try:
        args = _session_args(tmp, window=2.0, hop=0.5, tau=0.5, baseline_seconds=8.0,
                             duration=0.2)
        state = live_music.SessionState(args.target)
        logger = SessionLogger(args.participant, args.condition, root=tmp)
        refused = False
        try:
            live_music.eeg_worker(args, state, logger)
        except SystemExit:
            refused = True
        s.check("retuned estimator with no dwell refuses to start", refused)

        # And must NOT refuse once the dwell is set, or the guard blocks the very
        # configuration it exists to make safe.
        args2 = _session_args(tmp, window=2.0, hop=0.5, tau=0.5, min_dwell=1.0,
                              baseline_seconds=8.0, duration=0.2)
        state2 = live_music.SessionState(args2.target)
        logger2 = SessionLogger(args2.participant, args2.condition, root=tmp)
        blocked = False
        try:
            live_music.eeg_worker(args2, state2, logger2)
        except SystemExit:
            blocked = True
        except Exception:                        # noqa: BLE001 - any other failure is
            blocked = False                      # not this guard's business
        s.check("the guard does not block a dwell that satisfies it", not blocked)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_ladder_reachability(s: Suite) -> None:
    """Rungs 0 and 4 are unreachable under both arms - documented, not accidental."""
    reached = set()
    for target in (-1.0, 1.0):
        for z in np.arange(-4.0, 4.05, 0.1):
            reached.add(build_prompt(float(z), target).split(",")[0])
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


def test_streaming_estimator(s: Suite) -> None:
    """
    The low-latency alternative must be chunk-size independent and genuinely faster.

    Chunk independence is the property that makes it safe to drive from an LSL pull of
    arbitrary length: filter state persists across calls, so the output cannot depend on
    how the samples happened to arrive. Without it the estimate would vary with network
    timing, which is the kind of fault that only appears under load.
    """
    from eeg_features import FeatureConfig, StreamingBandPower, latency_budget

    cfg = FeatureConfig(sampling_rate=256.0)
    rng = np.random.default_rng(0)
    t = np.arange(2560) / 256.0
    sig = 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 5, t.size)

    a = StreamingBandPower(cfg, tau_seconds=0.25)
    for i in range(0, sig.size, 37):          # ragged chunks, as LSL delivers
        a.push(sig[i:i + 37])
    b = StreamingBandPower(cfg, tau_seconds=0.25)
    b.push(sig)                                # one shot
    rel = abs(a._acc - b._acc) / max(abs(b._acc), 1e-12)
    s.check("streaming estimator is chunk-size independent", rel < 1e-9,
            f"relative difference {rel:.2e}")

    stream = StreamingBandPower(cfg, tau_seconds=0.25).latency_budget()
    windowed = latency_budget(cfg, smoother_tau=3.0)
    s.check("streaming budget beats the windowed path",
            stream["total_analysis_latency_s"] < windowed["total_analysis_latency_s"] / 5,
            f"{stream['total_analysis_latency_s']:.2f}s vs "
            f"{windowed['total_analysis_latency_s']:.2f}s")

    s.check("streaming has no window or hop delay",
            stream["window_centroid_delay_s"] == 0 and stream["hop_quantization_s"] == 0)

    nan_out = StreamingBandPower(cfg).push(np.array([1.0, np.nan, 3.0]))
    s.check("non-finite input is refused, not propagated", not np.isfinite(nan_out))


def _session_args(tmp: str, **overrides):
    """
    A session's args, taken from live_music's OWN parser defaults.

    These used to be hand-built Namespaces, one per test, each carrying a frozen copy of
    the defaults. That meant a new flag broke the suite with an AttributeError, and -
    worse - the tests could keep passing against defaults the real entry point no longer
    used. Taking them from build_parser() means the tests exercise what a session
    actually runs with.
    """
    import live_music

    args = live_music.build_parser().parse_args([])
    args.participant, args.condition = "RUNTESTS", "pilot"
    args.mock, args.headless, args.out = True, True, tmp
    args.engine, args.library = "library", os.path.join(_ROOT, "library")
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_primary_outcome_ground_truth(s: Suite) -> None:
    """
    The registered PRIMARY OUTCOME, tested against a session whose answers are known.

    Until 9/5 the only assertion on analyze_session was that it completes without
    raising on the six sessions on disk. That catches a crash and nothing else: mean z
    is the primary outcome in analysis_plan.md section 3, and a sign error, an
    off-by-one in the applied filter, or a wrong denominator would all "complete" and
    all change the study's answer.

    The coupling index has had ground truth since 8/16 (validate_coupling recovers a
    known +6 s lag) and it is a SECONDARY measure. The primary one had none. Same
    convention, applied where it matters most: test the tool against a known result
    before trusting it on an unknown one.

    The synthetic session below is built so every expected value is computable by hand.
    """
    from analyze_session import HOLD_SECONDS, TARGET_BAND_HALF_WIDTH, basic_metrics

    target = -1.0
    hop = 1.0

    def session(zs, applied=None, valid=None):
        applied = [True] * len(zs) if applied is None else applied
        valid = [True] * len(zs) if valid is None else valid
        return {
            "dir": "synthetic",
            "manifest": {"feature_config": {"hop_seconds": hop}, "target_z": target,
                         "participant_id": "SYN", "condition": "pilot"},
            "audio": [],
            "windows": [{"phase": "intervention", "elapsed_s": float(i) * hop,
                         "z": float(z), "applied": a, "valid": v}
                        for i, (z, a, v) in enumerate(zip(zs, applied, valid))],
        }

    # 1. mean, sd and range over a series whose answers are arithmetic.
    zs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    m = basic_metrics(session(zs))
    s.check("z_mean is the mean of the applied windows", abs(m["z_mean"] - 0.0) < 1e-9,
            f"got {m['z_mean']:+.6f} for {zs}")
    s.check("z_min and z_max are the extremes",
            m["z_min"] == -2.0 and m["z_max"] == 2.0)

    # 2. UNAPPLIED WINDOWS MUST NOT COUNT. This is the filter most likely to rot, and
    #    it silently shifts the primary outcome rather than failing.
    zs = [-1.0, -1.0, 99.0, 99.0]
    m = basic_metrics(session(zs, applied=[True, True, False, False]))
    s.check("unapplied windows are excluded from the primary outcome",
            abs(m["z_mean"] - (-1.0)) < 1e-9,
            f"got {m['z_mean']:+.4f}; two windows at z=99 must not contribute")

    # 3. in-band fraction, with a value exactly on the boundary (inclusive per the code).
    edge = target + TARGET_BAND_HALF_WIDTH
    zs = [target, target, edge, target + 5.0]
    m = basic_metrics(session(zs))
    s.check("time in band counts the boundary and excludes the excursion",
            abs(m["time_in_band_fraction"] - 0.75) < 1e-9,
            f"got {m['time_in_band_fraction']:.3f}, expected 0.750")

    # 4. time_to_target must be nan when the band is never reached. The docstring calls
    #    this out specifically: coding "never" as the session length would turn a
    #    non-response into a slow response, which is a different clinical claim.
    m = basic_metrics(session([target + 5.0] * int(HOLD_SECONDS * 3)))
    s.check("time_to_target is nan when the band is never reached",
            not np.isfinite(m["time_to_target_s"]), f"got {m['time_to_target_s']}")

    # 5. ...and is measured from the first window when it is reached immediately.
    m = basic_metrics(session([target] * int(HOLD_SECONDS * 3)))
    s.check("time_to_target is 0 s when already in band at the start",
            m["time_to_target_s"] == 0.0, f"got {m['time_to_target_s']}")

    # 6. drift is second half minus first half, so a rise is positive.
    m = basic_metrics(session([0.0, 0.0, 2.0, 2.0]))
    s.check("z_drift is positive for a rising session",
            abs(m["z_drift"] - 2.0) < 1e-9, f"got {m['z_drift']:+.3f}, expected +2.000")

    # 7. A session where every window was REJECTED is the realistic degenerate case -
    #    bad contact for twenty minutes - and it must yield nan rather than a number.
    m = basic_metrics(session([1.0, 2.0, 3.0], applied=[False, False, False]))
    s.check("a fully rejected session yields nan, not zero",
            not np.isfinite(m["z_mean"]), f"got {m['z_mean']}")

    # 8. A session with NO windows is a different case: basic_metrics returns an error
    #    dict with none of these keys, and report() used to format the absent target_z
    #    and raise TypeError. An operator running this right after a session that had
    #    already gone wrong would see a traceback and conclude the ANALYSIS was broken.
    #    The control loop's rule is that a crash must not look like a success; this is
    #    the same rule pointing the other way.
    import tempfile as _tf

    empty = _tf.mkdtemp(prefix="emptysess_")
    try:
        with open(os.path.join(empty, "events.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "manifest", "participant_id": "EMPTY",
                                 "condition": "pilot"}) + chr(10))
        from analyze_session import report as _report
        try:
            out = _report(empty, skip_aci=True)
            ok, why = "error" in out, f"reported {out.get('error')!r}"
        except Exception as exc:  # noqa: BLE001
            ok, why = False, f"raised {type(exc).__name__}: {exc}"
        s.check("a session with no windows reports rather than crashes", ok, why)
    finally:
        import shutil
        shutil.rmtree(empty, ignore_errors=True)


def test_sham_path(s: Suite) -> None:
    """
    The sham arm, which is half the registered design and has never run.

    Nothing here was covered before 9/5. check_replay_fidelity carried a --self-test
    that validates it catches the known -7.06 s origin bug, and the suite never ran it -
    so the script that turns a one-off manual check into a standing assertion was itself
    not standing. residual_contingency's positive control was likewise unasserted.

    The third check is new and catches a study-invalidating mistake that nothing would
    have reported: yoking to a schedule the current controller cannot produce. Removing
    the trend suffix shrank the reachable prompt space from 20 strings to 5, which
    retroactively disqualified every session on disk as a yoke source in a way unrelated
    to chatter or to the origin bias. A replay of such a schedule is faithful, the
    library still holds the segments, and the audio sounds fine - the arms simply draw
    from different prompt spaces, which is the one thing yoking exists to prevent.
    """
    import live_music
    from music_engine import build_prompt as bp

    src = sorted(glob.glob(os.path.join(_ROOT, "sessions", "PILOT*")))
    if not src:
        s.skip("sham path", "no PILOT session to yoke from")
        return

    schedule = live_music._load_yoked_prompts(src[-1])
    s.check("a yoke schedule loads and is non-trivial", len(schedule) > 10,
            f"{len(schedule)} entries from {os.path.basename(src[-1])}")

    # The prompt-space guard must fire on a pre-9/5 source and must NOT fire on a
    # schedule built from the current controller. A guard that always fires is noise;
    # one that never fires is decoration.
    n_bad = live_music._warn_if_source_outside_prompt_space(src[-1], schedule)
    s.check("prompt-space guard fires on a pre-suffix-removal source", n_bad > 0,
            f"{n_bad} of {len(schedule)} entries unreachable")

    current = [(float(i), bp(float(z), -1.0))
               for i, z in enumerate(np.arange(-3, 3.01, 0.1))]
    n_ok = live_music._warn_if_source_outside_prompt_space("synthetic", current)
    s.check("prompt-space guard is silent on a current-controller schedule", n_ok == 0,
            f"{n_ok} unreachable of {len(current)}")

    # Lookup must be a step function that holds the last decision, not interpolate or
    # run off the end - the sham plays to its own duration, not the source's.
    first_t = schedule[0][0]
    s.check("yoked lookup holds before the first entry",
            live_music._yoked_prompt_at(schedule, first_t - 10.0) == schedule[0][1])
    s.check("yoked lookup holds past the end",
            live_music._yoked_prompt_at(schedule, schedule[-1][0] + 1e6) == schedule[-1][1])

    # THE POSITIVE CONTROL, asserted rather than printed. residual_contingency.py is a
    # report and always exits 0, so running it as a subprocess validator would have been
    # a check that cannot fail. An adaptive session must register as contingent - if the
    # detector cannot see contingency where it certainly exists, a sham that leaks
    # contingency would pass silently, and that is the measurement the sham arm rests on.
    import residual_contingency as rc

    (z_t, z), (seg_t, seg_rung) = rc.load_session(src[-1])
    if z.size < 100 or seg_rung.size < 5:
        s.skip("residual contingency positive control", "pilot lacks z or a schedule")
        return
    r = rc.score(z_t, z, seg_t, seg_rung, target_z=rc.session_target_z(src[-1]))
    s.check("an adaptive session registers as contingent", r["excess_z"] > 3.0,
            f"excess {r['excess_z']:+.1f} sd, match {100 * r['match']:.1f}% "
            f"vs chance {100 * r['chance']:.1f}%")


def test_session_failure_recording(s: Suite) -> None:
    """A crashed worker must record FAILED and must not record complete."""
    import argparse as ap  # noqa: F401

    import live_music
    from music_engine import PromptGovernor
    from session_logger import SessionLogger

    # The fault is injected at the governor, because that is where the control loop
    # now builds its prompt. It used to be injected by patching live_music.build_prompt,
    # which the loop stopped calling directly when PromptGovernor was introduced - the
    # patch still applied cleanly and intercepted nothing, so the test passed by
    # constructing a session that never crashed. Injecting where the loop actually calls
    # is the difference between testing the failure path and testing nothing.
    calls = {"n": 0}

    class Flaky(PromptGovernor):
        def update(self, *a, **k):
            calls["n"] += 1
            if calls["n"] > 1:                  # succeed once, then fail mid-session
                raise RuntimeError("injected fault")
            return super().update(*a, **k)

    original = live_music.PromptGovernor
    live_music.PromptGovernor = Flaky
    tmp = tempfile.mkdtemp(prefix="failtest_")
    try:
        args = _session_args(tmp, baseline_seconds=20.0, duration=0.4)
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
        live_music.PromptGovernor = original
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
        args = _session_args(tmp, baseline_seconds=8.0, duration=0.3)
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


# Content hash of docs/analysis_plan.md at the freeze, tag preregistration-v1,
# commit e45bd32. Line endings are normalised before hashing because git rewrites
# them on checkout, so the raw hash of a fresh clone on Windows would differ from
# the same file on Linux and the check would fail for the wrong reason.
FROZEN_PLAN_SHA256 = "538328a2dac75fc9bab76fecb7f7cfa11ef88db9b08f6cf7e187bd1fe4fe4ce5"


def test_preregistration_frozen(s: Suite) -> None:
    """
    The analysis plan must not change after the freeze.

    A pre-registration's whole value is that it predates the data. "We did not edit it"
    is an assertion; a hash checked on every test run is a fact. Without this, an
    accidental edit - a typo fix, a reflow, a well-meant clarification - silently
    destroys the guarantee and nothing anywhere would notice.

    If this fails and the change was deliberate, the change does not belong in the file.
    It belongs in section 9 as a dated deviation, and the freeze stands.
    """
    import hashlib

    path = os.path.join(_ROOT, "docs", "analysis_plan.md")
    if not os.path.exists(path):
        s.skip("pre-registration frozen", "analysis_plan.md not found")
        return
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    s.check("pre-registration unchanged since the freeze",
            digest == FROZEN_PLAN_SHA256,
            f"{digest[:16]}... vs frozen {FROZEN_PLAN_SHA256[:16]}...")

    # The deviation log must exist and must live OUTSIDE the hash. Section 9 of the plan
    # says deviations are recorded "here", but "here" is inside the hashed file, so
    # appending a row fails the check above. The first person to log a deviation would
    # find a failing test whose only cause was the log entry, and the obvious fix -
    # updating FROZEN_PLAN_SHA256 - destroys the freeze permanently. A guard that makes
    # disabling itself the reasonable next step is worse than no guard, so the log lives
    # in its own file and this asserts it has not been quietly dropped.
    dev = os.path.join(_ROOT, "docs", "deviations.md")
    s.check("deviation log exists outside the frozen file",
            os.path.exists(dev) and os.path.getsize(dev) > 0,
            "docs/deviations.md")


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
                        help="skip the checks that recompute from raw data: the "
                             "permutation coupling validation and the claims that "
                             "rerun power simulations, DEAP, and the session replay. "
                             "NOT sufficient before a commit.")
    args = parser.parse_args()

    s = Suite()

    s.section("1. CONTROLLER - purity, hysteresis, and the chatter regression")
    test_build_prompt_purity(s)
    test_trend_is_not_measurable(s, load_session_z())
    test_state_rung_monotonic(s)
    test_ladder_reachability(s)
    test_ladder_hysteresis(s)
    test_ladder_hysteresis_does_not_latch(s, load_session_z())
    test_min_dwell(s)
    test_chatter_regression(s, load_session_z())
    test_streaming_estimator(s)

    s.section("2. SESSION - a crash must never look like a success")
    test_primary_outcome_ground_truth(s)
    test_sham_path(s)
    test_session_failure_recording(s)
    test_retuned_estimator_guard(s)
    test_baseline_abort(s)

    s.section("3. PRE-REGISTRATION - frozen before the data")
    test_preregistration_frozen(s)

    s.section("4. ANALYSIS - the pipeline must complete on real sessions")
    test_analysis_runs(s)

    s.section("5. VALIDATORS - library coverage and the coupling estimator")
    run_validator(s, "library coverage and mixing", "verify_library.py", ["--synthetic"])
    # Guards the manuscript against the code moving underneath it. A number copied
    # into prose once and then diverging is how honest projects publish
    # unreproducible papers.
    run_validator(s, "preprint claims still reproduce", "verify_claims.py",
                  ["--quick"] if args.quick else [])
    # Both halves of sham validity: faithfully coupled to its SOURCE, and decoupled
    # from the LISTENER. Neither implies the other, and neither was run by this suite.
    run_validator(s, "replay fidelity catches the origin bug",
                  "check_replay_fidelity.py", ["--self-test"])

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
