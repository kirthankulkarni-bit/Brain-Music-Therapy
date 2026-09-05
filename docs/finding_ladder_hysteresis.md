# The ladder has no hysteresis, and that blocks the retuning

**Found 2026-08-28, testing a recommendation I had already written into
`next_session.md`. It was wrong, and this is the correction.**

## What I recommended, and why it fails

`finding_analysis_latency.md` showed the deployed estimator is a dominated configuration
and recommended `--window 2 --hop 0.5 --tau 0.5` as "a two-parameter change, no new code".

Replaying PILOT01's raw recording through those settings:

| configuration | prompt changes | median gap | switches under 1 s |
|---|---|---|---|
| deployed (4 s / 1 s / tau 3) | 36 | 3.0 s | **0** |
| retuned (2 s / 0.5 s / tau 0.5) | **459** | 1.0 s | **168** |

37% of switches arrive faster than the crossfade can resolve — worse than the 30% that
made PILOT01's audio unusable. **The recommendation as written would have reproduced the
defect it was meant to follow.**

## It is not the trend suffix, and thresholds cannot fix it

The obvious suspect was the trend hysteresis, calibrated for tau = 3. It is not:

| thresholds | changes | under 1 s |
|---|---|---|
| current (0.35 / 0.175) | 459 | 168 |
| recalibrated 5 x sd (0.191) | 459 | 168 |
| above the observed maximum (0.170) | 459 | 168 |

Identical. Suffix-only changes are **zero** in both configurations — the suffix is inert,
exactly as designed after the 8/16 fix.

What chatters is the **rung**:

| configuration | z sd | rung flips | prompt changes |
|---|---|---|---|
| deployed | 0.92 | 217 | 36 |
| retuned | 1.37 | **1122** | 459 |

`state_rung(z)` is `round(2 + z)` clipped to the ladder. It has **no hysteresis at all**.
The rung flips whenever z crosses a half-integer boundary, and `_DEADBAND_Z` only applies
near the *target*, not at rung edges. A noisier z crosses those boundaries constantly.

The deployed configuration survives this only by accident: `build_prompt` maps several
`here` values onto the same output level, so 217 rung flips collapse into 36 prompt
changes. The mapping absorbs the chatter rather than the controller preventing it. Reduce
the smoothing and the absorption stops being sufficient.

## Adding ladder hysteresis: helps, and is not sufficient

A Schmitt trigger on the rung — leave a rung only once z is `margin` past the boundary:

| configuration | changes | median gap | under 1 s |
|---|---|---|---|
| retuned, no hysteresis | 459 | 1.0 s | 168 |
| retuned + margin 0.25 | 296 | 1.5 s | 68 |
| retuned + margin 0.50 | 208 | 2.0 s | **28** |
| **deployed + margin 0.25** | **14** | **16.0 s** | **0** |

Two conclusions, and they point in different directions.

**The retuning is not ready.** Even at margin 0.5 — which is large, half a rung — 28
switches still arrive inside a crossfade. Making those settings safe needs more than a
margin: a minimum dwell time, or an acceptance that this estimator is too noisy for a
five-rung quantiser. That is real controller work, not a flag change.

**Ladder hysteresis is worth having anyway.** On the *deployed* configuration it cuts
prompt changes from 36 to 14 and raises the median gap from 3 s to 16 s, with zero
sub-crossfade switches. It removes chatter the current system is absorbing by luck rather
than by design.

## Consequences

**`next_session.md` is corrected.** PILOT02 runs at the **deployed** settings. A clean
yoke source at known-good settings is the thing that unblocks the study; an unvalidated
speed-up is not.

**The latency finding still stands.** The deployed estimator remains dominated on
information rate, and the analysis path remains 85% of the budget. What changed is the
cost of acting on it: adopting a faster estimator requires giving the ladder hysteresis
and probably a dwell limit, so it is a controller change rather than a configuration
change.

**The n = 7 projection is unaffected in principle and further away in practice.** It
assumed the information rate transfers; that assumption is untested and now sits behind
controller work as well.

## What I got wrong, and why the check caught it

I wrote "a two-parameter change, no new code" from a benchmark that measured the
*estimator* in isolation. The estimator was fine. The controller downstream of it was
tuned — implicitly, by nobody — against the noise level the estimator happened to produce,
and I did not test the pair together until after recommending the change.

The general lesson is in the numbers above: the deployed system absorbs 217 rung flips
into 36 prompt changes. That absorption is not a designed property, it is a coincidence of
the mapping, and any change to the noise level spends it.

---

# Correction and completion, 2026-09-05

The recommendation above — "ladder hysteresis is worth having anyway", margin 0.25 as a
clear win — **was wrong, and enabling it would have shipped a controller that stops
responding.** The controller work it called for is now done. Both of those came out of
committing the replay instead of leaving it in a terminal.

## The measurements above were not reproducible

Every number in the tables above came from code that was never committed. That put them
outside the rule in HANDOFF §9, and the rule earned its place again here: rebuilding the
replay as `scripts/controller_replay.py` disagreed with them, and it is the committed one
that is calibrated.

The harness is validated against a known result before being trusted on an unknown one.
Replaying PILOT01's **logged** z at the deployed settings gives 24 prompt changes, which
is exactly what `verify_claims.py` has asserted since 8/28. Reconstructing z from the raw
recording and replaying that gives 28 — so the reconstruction carries about 17% error,
and that is the error bar on every retuned number, which has no log to check against.

| configuration | committed harness | ad-hoc replay above |
|---|---|---|
| deployed, changes | 28 | 36 |
| deployed, under a crossfade | 0 | 0 |
| retuned, changes | 382 | 459 |
| retuned, under a crossfade | 136 | 168 |
| deployed rung flips | 194 | 217 |
| retuned rung flips | 1006 | 1122 |

The conclusion is unchanged and the magnitudes are not. Two things had to be right before
the reconstruction reproduced the pipeline at r = 0.991:

**The time axis has to come from the LSL timestamps, not the sample index.** PILOT01's
raw stream has two dropouts, the worst 10.67 s. Across a gap, `sample_index / fs` and real
time diverge permanently, so no constant offset can align them — fidelity caps at r = 0.66
and the offset search pins itself to whatever bound it is given. This is trap 3 in a
second costume: the two clocks differ by more than an origin.

**The fidelity check has to compare like with like.** The session logs
`feats.log_beta_alpha`, the raw per-window value; `smoothed_log_beta_alpha` is a separate
field. Correlating a smoothed reconstruction against the unsmoothed log costs ~0.3 of r
and drags the offset with it.

## The margin was latched, not conservative

Enabling `ladder_margin` at 0.25 produces **zero prompt changes across all 1043 windows
of PILOT01** — the music never moves once in twenty minutes, while the participant's own
state rung ranges up to 4. That is not hysteresis. That is a dead controller.

`build_prompt` derived the previous rung with `_rung_of(previous_prompt)`. A prompt
records the rung being **played**, and `build_prompt` always plays one rung toward the
target. Feeding that back as the previous **state** estimate closes a loop: the estimate
is pulled toward the goal, which pulls the output toward the goal, which pulls the
estimate further, until it sticks on the goal rung and nothing can move it.

The "36 → 14, a clear win" figure never exercised this path — it tracked the state rung in
a separate variable, which is the correct algorithm and not the one that was wired up.
**The recommendation was measured on a different controller than the one it recommended
enabling.**

The fix is a parameter, not a heuristic: `build_prompt` now takes `previous_rung`
explicitly, and `PromptGovernor` owns that state. `build_prompt` stays pure, which
`build_library`'s enumeration and the purity test both depend on.

Why the test suite did not catch it: `test_ladder_hysteresis` exercised `state_rung` in
isolation, called correctly, and passed throughout. Nothing drove the whole controller
with the margin enabled. A test that cannot fail is not a safeguard —
`test_ladder_hysteresis_does_not_latch` now does, and fails with 4 checks against the old
wiring.

## The retuning is viable: dwell = crossfade

The unfinished work above was "a minimum dwell time, or an acceptance that this estimator
is too noisy for a five-rung quantiser". It is the dwell, and the right value is not tuned.

**A dwell of at least one crossfade is exactly the condition for no switch arriving before
the previous crossfade completes.** Below that, transitions blend into a continuous mush
of two independent renders; at or above it, sub-crossfade switches are zero by
construction. Measured across every margin, on both estimators:

| estimator | margin | dwell | changes | median gap | under a crossfade | end-to-end |
|---|---|---|---|---|---|---|
| deployed | 0 | 0 | 28 | 3.0 s | 0 | **6.67 s** |
| retuned | 0 | 0 | 382 | 1.0 s | **136** | 2.68 s |
| retuned | 0 | 0.5 | 355 | 1.0 s | 71 | 3.18 s |
| retuned | 0 | **1.0** | 299 | 1.5 s | **0** | **3.68 s** |
| retuned | 0.5 | **1.0** | 167 | 2.0 s | **0** | **3.68 s** |
| deployed | 0.25 | 0 | 8 | 21.0 s | 0 | 6.67 s |

End-to-end is detection latency + dwell + crossfade. The retuned estimator with a 1 s
dwell responds in **3.68 s against the deployed 6.67 s** — a 1.8x improvement that keeps
the information-rate gain and has none of the chatter.

**The dwell is not free, and the earlier framing of it was wrong.** With the library
engine a prompt change takes effect within one crossfade, not one segment — that 8x is the
entire contribution of `library_engine.py`. So a dwell genuinely raises worst-case
responsiveness from one crossfade to `dwell + crossfade`, exactly as the comment in
`library_engine._begin_switch` warned. An 8 s dwell would have consumed the whole retuning
gain and left the system slower than deployed. One crossfade is the smallest dwell that
does the job, which is why it is the right one.

## What is still not answered

**Whether 299 changes in twenty minutes is musically acceptable.** Zero sub-crossfade
switches means no clicks. It does not mean it sounds good — one change every 4 s (or every
7 s at margin 0.5) against the deployed 28 is a different listening experience, and nobody
has heard it. That is open question §7.5 and it is a listening judgement, not a
measurement.

**Whether the information-rate gain survives the real contrast.** Unchanged: the n ≈ 7
projection assumes discriminability measured on eyes-open/closed transfers to
adaptive-vs-sham.

**Whether to enable the margin at all.** It now works, and at the deployed settings it
cuts changes from 28 to 8 with a 21 s median gap. It remains off by default because it
changes what a participant hears.

## Consequences for the next session

`next_session.md` is unchanged where it matters: **PILOT02 still runs at the deployed
settings with no dwell and no margin.** A clean yoke source at known-good settings is what
unblocks the study, and that argument does not depend on any of the above.

What changed is that the retuned settings are no longer categorically unsafe — they are a
measured, guarded option for a later session, rather than a recommendation that had never
been tested end to end.

`live_music.py` now **refuses to start** if the estimator is retuned and `--min-dwell` is
below the crossfade. A warning would not have been enough: the resulting recording is not
merely noisy, it is disqualified as a yoke source, and that is discovered after the
participant has gone home.
