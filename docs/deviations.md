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
