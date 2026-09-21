# Deviations from the frozen pre-registration

`analysis_plan.md` §9 says departures from the plan are recorded "here", with date, reason
and commit. They cannot be, and that is a defect in the freeze mechanism rather than a
choice:

**§9 is inside the hashed file.** `run_tests.py::test_preregistration_frozen` hashes the
whole of `analysis_plan.md` against `FROZEN_PLAN_SHA256`. Appending a row to the
deviations table changes the hash and fails the test. Its docstring resolves the conflict
in the wrong direction — *"the change does not belong in the file. It belongs in section 9
as a dated deviation"* — but §9 **is** in the file, so the sanctioned path is the one the
guard blocks.

That matters because of what someone does when they hit it. The first person to log a
deviation finds a failing test, sees that the only edit they made was to the deviation
log, and updates `FROZEN_PLAN_SHA256` to make it pass. At that point the freeze is gone
and every future edit passes silently. **A guard that forces a wrong action to proceed is
worse than no guard**, because it manufactures a plausible reason to disable itself.

So the log lives here, outside the hash, and the plan stays byte-identical to what was
registered. `analysis_plan.md` is not edited to add a pointer to this file, because that
edit would itself break the freeze. The freeze stands; the log is appendable; neither
property is traded for the other.

`run_tests.py` asserts this file exists, so it cannot be quietly deleted.

---

## Log

| date | deviation | reason | commit |
|---|---|---|---|
| 2026-09-05 | Removed the trend suffix from `build_prompt`. The controller now emits only the five ladder rungs, of which 1–3 are reachable. | The plan already describes the controller this way — §7 records "the trend suffix cannot fire" and "effectively a three-rung ladder" — so this removes a discrepancy between the code and the registered description rather than changing the protocol. Verified to be a no-op: 0 suffixed prompts across all 1212 logged windows on disk. See below. | *(this commit)* |

| 2026-09-20 | Energy ladder widened from 5 rungs at 1.0 SD to **9 rungs at 0.5 SD**, and `--min-dwell` defaulted to **30 s**. | The controller could not produce a stimulus difference for a calm participant: PILOT02 played ONE prompt for twenty minutes, so a yoked sham would be acoustically identical to the adaptive arm and the registered contrast would have nothing to test. See below. | *(this commit)* |

---

## 2026-09-20 — nine rungs at 0.5 SD, and a 30 s dwell

### What the pre-registration says, and what changed

The plan does not fix the rung count or width. §7 records, before the fact, that "only
rungs 1–3 are reachable" and that "PILOT01 used rung 1 for 96% of the session" — i.e. it
registered the problem as a known limitation rather than registering the ladder's
geometry. §3's outcomes and §4's power analysis do not reference it.

So this is a change to the intervention, not to the analysis, and it is logged here
because it changes what a participant hears. It does **not** change the primary outcome,
the target, the deadband, the iso-principle (still match-then-lead by one rung), or the
crossover design.

### Why it was necessary

PILOT02 (2026-09-17, 20 min at deployed settings) produced **0 prompt changes**. One
prompt, 301 segments, 20 minutes.

That was the controller working as designed. With the relaxation target the goal is the
rung at z = −1; the output is always one rung toward it; at 1.0 SD spacing every z below
+0.5 mapped to the same rung. PILOT02 sat at z = **−1.46** (SD 0.67) — at or past target
from the first window — and never came close to leaving that rung. PILOT01 was the same,
less extremely: one rung for 95.9% of its session.

**A yoked sham of a one-prompt session is acoustically identical to the adaptive arm.**
The registered contrast (plan §3, adaptive − sham on mean z) therefore had no stimulus
difference to test for the participant type the intervention is aimed at — someone who
relaxes. That is not a tuning issue; it makes the study unable to answer its own question.

### Why the width, and not the target

Four candidates were replayed against both sessions' real logged z before choosing:

| option | PILOT01 changes | PILOT02 changes |
|---|---|---|
| deployed (5 rungs at 1.0 SD) | 24 | **0** |
| personalised target (baseline-relative) | 109 | **0** |
| match the participant's own rung when in band | 24 | **0** |
| **0.5 SD rungs** | 197 | **102** |

Moving the target only slides the same degenerate mapping down: the participant's z spread
(SD 0.67) still never crosses a 1.0 SD boundary. **The rung width is the binding
constraint.** This matters because the personalised target was the intuitive fix and was
recommended before the simulation was run; it does nothing.

### Why 0.5 SD and a 30 s dwell

A finer ladder alone changes the music every 2–3 s. The dwell added on 9/5 bounds the rate,
and at 30 s the change rate pins at roughly one per 35 s at **any** width — so the width is
purely a choice about variety:

| width | rungs | PILOT01 changes / distinct | PILOT02 changes / distinct |
|---|---|---|---|
| 1.0 SD | 5 | 14 / 2 | **0 / 1** |
| 0.75 SD | 7 | 32 / 4 | 33 / **2** |
| **0.5 SD** | **9** | **33 / 5** | **33 / 3** |
| 0.4 SD | 11 | 33 / 7 | 33 / 4 |

0.75 SD leaves PILOT02 alternating between two prompts, which is thin for a contrast. 0.4
SD buys variety the library would have to pay for. 0.5 SD is the smallest ladder that gives
both recordings at least three distinct pieces of music.

Through the deployed controller, PILOT01 now replays to 33 changes across 5 prompts and
PILOT02 to 26 across 3, with zero switches inside a crossfade.

### The library

The original five prompts are the **even** rungs, unchanged, so the existing 220 segments
still cover five of the nine. Only the four odd rungs are new. `verify_library` fails until
`python scripts/build_library.py` has rendered them — that failure is expected and is the
gate working.

### What this does not establish

That the added variation helps, or that a participant notices it. This is a replay of
logged z through a new mapping: it predicts the **schedule**, not the response. In a real
closed loop the music changes z, which changes the schedule. And variation is necessary for
the adaptive-vs-sham contrast, not sufficient — the sham must still be yoked from a session
recorded with this controller, so **a new yoke-source session is required**.

---

## 2026-09-05 — removing the trend suffix

### Why it is not a protocol change

The registered outcomes (§3) and power analysis (§4) do not reference the suffix. §7
already records it as unable to fire. Replaying every session on disk through
`build_prompt` produces **0 suffixed prompts out of 1212 windows**, so no participant has
heard one and none could have. Deleting a branch that provably never executes changes
nothing a participant experiences.

It is logged anyway, because §9 says to log deviations "including deviations that seem
trivial", and because the intervention code changed between the freeze and data
collection. An unlogged change is indistinguishable from an undisclosed one.

### Why it was removed rather than left alone

Not because it is inert. Because **the quantity it thresholds is not measurable**, so no
threshold can make it work.

Measured on PILOT01, in z units, on the same 20-hop least-squares slope the controller
uses:

| series | hop | largest genuine 60 s drift | trend estimator noise (sd) | ratio |
|---|---|---|---|---|
| logged z, deployed | 1.0 s | 0.0385 | 0.0681 | **noise 1.8× the signal** |
| reconstructed, deployed | 1.0 s | 0.0388 | 0.0660 | noise 1.7× the signal |
| reconstructed, retuned | 0.5 s | 0.0240 | 0.0937 | **noise 3.9× the signal** |

The trend the suffix exists to describe is smaller than the noise of the estimator
measuring it, under every configuration, and it gets **worse** under the retuned
estimator, not better — the trend window shrinks to 10 s while the noise rises.

That closes both escapes:

- A threshold **above** the noise floor (the current `ENTER = 0.35`) can only ever be
  crossed by a noise excursion. Real drift maxes out at 0.0385, roughly 9× below it.
- A threshold low enough to catch real drift (~0.04) sits far below the noise floor and
  fires constantly. That is exactly the 8/16 defect: threshold 0.05 against one-hop noise
  of 0.275, which produced 491 prompt changes in twenty minutes.

There is no setting in between that does the job. The control is not miscalibrated, it is
uncalibratable, and the honest response to a control that cannot work is to remove it
rather than to keep tuning it.

### It was a latent hazard, not merely dead weight

`ENTER = 0.35` was calibrated against the deployed estimator's noise. Under the retuned
estimator that this project now intends to use, the largest observed 20-hop slope rises
from 0.2032 to **0.3023** — a margin of 1.16×, down from 1.72×.

This project has been burned by exactly that margin before. `ENTER` was first set to 0.20
and fired zero times out of 1023 windows by a margin of **0.0002** — recorded at the time
as "an accident rather than a decision". A 1.16× margin measured on a single session is
the same accident with more room, and one session away from repeating.

### What it cost to keep

`build_library.py` enumerates the reachable prompt space by sweeping `build_prompt`, so
the library carried a rendered segment set for every suffixed prompt: **15 of 20 distinct
prompts, 60 of 220 segments, 8.0 minutes of the 29.3-minute library** — 27% of the audio,
maintained for prompts the controller could not request.

### What was kept

The measurement, not the mechanism. `scripts/calibrate_hysteresis.py` still measures the
trend estimator's noise against real recordings; it now reports whether the trend is
measurable at all rather than proposing thresholds for a control that no longer exists.
`verify_claims.py` asserts the noise-to-signal ratio, so the justification is regenerated
rather than remembered.
