# Sham validation: residual contingency

**Written 2026-08-30, before any sham session exists.** `scripts/residual_contingency.py`.

Not part of the frozen analysis plan. This is a validity check on the control
condition, not an outcome analysis. It is written in advance precisely so it cannot be
tuned to a result later.

## Why

A yoked sham is an informative comparator only if the loss of real-time contingency is
*demonstrated* rather than assumed. The current standard — *Rethinking control
conditions in clinical neurofeedback trials* (2026) — asks for one thing: record the
participant's true neural signal while false feedback is delivered, then quantify how
often the delivered feedback coincides with what their own signal would have produced
under the same rules.

This study can meet that standard with data it already logs. `window` events carry the
z the controller would have seen; `audio_segment` events carry what actually played.
Nothing new needs recording.

There is also a specific reason to care here. Self-yoking was rejected in §2 of the
analysis plan on the argument that a participant's own schedule may still partially
track them across days. That is a residual-contingency argument made from reasoning.
This measures it.

## Method

Replay `build_prompt` over the participant's own logged z to get the rung they *would*
have been served, and compare against the rung actually delivered.

- **match** — fraction of windows where delivered rung equals counterfactual rung
- **chance** — the same under circular shifts of the delivered schedule, which
  preserves its structure and rung occupancy while destroying alignment with this
  participant's signal
- **excess** — (match − chance) in units of the shift-null SD

`build_prompt` is *called*, not reimplemented, and the rung read back off the returned
string, so the check cannot drift from the deployed controller.

## Results

**Positive control passes.** PILOT01 is adaptive, so it must show strong contingency:

| session | n | match | chance | excess |
|---|---|---|---|---|
| PILOT01 (adaptive) | 1043 | 97.7% | 92.2% | **+10.1 sd** |

**Chance is 92.2%.** Raw match rate is almost uninformative here, because one rung
dominates occupancy. Reporting excess in null-SD units is not a stylistic choice — it
is what makes the number mean anything.

**Detection floor: 20% contingent.** Mixing the participant's own schedule with a
surrogate at varying fractions:

| contingent fraction | match | chance | excess |
|---|---|---|---|
| 0% | 92.6% | 92.9% | −0.7 sd |
| 5% | 92.7% | 92.9% | −0.4 sd |
| 10% | 93.4% | 93.2% | +0.5 sd |
| **20%** | 94.2% | 92.8% | **+3.6 sd** |
| 50% | 94.8% | 92.8% | +6.2 sd |
| 100% | 97.7% | 92.4% | +9.3 sd |

A sham that is 20% contingent gets caught. One that is 10% contingent does not.

## What that means, and the finding inside it

**The check is sound but blunt, and it is blunt for a reason that is itself a result.**
PILOT01 spent 96% of the session in one rung. When the counterfactual is almost always
the same value, there is very little room for contingency to express itself, and the
null sits at 92.9% with a small SD. The rung-reachability collapse documented in
`scripts/ladder_policy.py` therefore *directly limits how well the sham can be
validated*. Those two facts are the same fact.

Practical consequences:

- **Never report "no residual contingency" without the detection floor beside it.**
  Unqualified, that claim would overstate what this data can support.
- The floor should be recomputed per participant. If real participants spread across
  rungs more than PILOT01 did, it improves; if they concentrate further, it worsens.
- If a policy change ever widens rung occupancy (the MATCH-THEN-LEAD option in
  `ladder_policy.py`), sham validation gets sharper as a side effect. That is a genuine
  argument for that policy which had not been on the list.

## One correction worth recording

The first version of this script used `state_rung(z)` as the counterfactual and the
adaptive positive control scored only +1.6 sd. That was wrong: the controller never
plays the participant's current rung, it leads by one rung toward the target. The
positive control is what caught it, which is the reason to always have one.

## Still open

- No sham session exists yet, so nothing has been validated in anger — only the
  machinery and its sensitivity.
- The pre-fix seven-second sham offset is a *replay-fidelity* bug, not strictly a
  residual-contingency one, and this check does not target it. Worth a separate
  assertion that replayed segment onsets match the source session's timeline.
- Only one session has usable z plus a schedule, so a real cross-yoked pair could not
  be tested. Run it on the first two real sessions.

---

# Replay fidelity

**Added 2026-08-30.** `scripts/check_replay_fidelity.py`.

Residual contingency asks whether the sham is decoupled from the *participant*. This
asks the complementary question: whether it is faithfully coupled to its *source*. A
yoked sham is acoustically matched to the adaptive arm only if it reproduces the
source's prompt-decision timeline. Both must hold, and neither implies the other.

## Why it is a standing check rather than a one-off

The pre-fix loader anchored replay offsets on the source's first **audio** event while
prompts are decided at **window** boundaries. On PILOT01 those origins are 7.06 s apart,
so every replayed prompt landed most of a segment early on a loop with a 6.5 s latency
budget — and a prompt superseded before the first segment was logged disappeared from
the replay entirely.

That was caught by hand and `_load_yoked_prompts` records it as "verified to 0.00 s
against PILOT01's 492 changes" — a single manual check. A fidelity bug that returns
silently is worth as much as one never fixed, so this makes the verification run on
every sham session, with a non-zero exit code so it can gate an analysis.

## Self-test

The checker is validated against the bug it exists to catch:

| input | result |
|---|---|
| correct schedule vs itself | **PASS** — origin +0.00 s, median +0.00 s, worst +0.00 s, 492 changes |
| pre-fix audio-anchored schedule | **FAIL** — worst −7.06 s, median −7.04 s |

The −7.06 s is recovered independently and matches the value in the fix commit.

**Read the self-test carefully.** Both schedules there are normalised to their own first
element, so the origin difference is zero by construction and the bug surfaces as DRIFT
rather than ORIGIN. In a real sham a wrong anchor shifts deliveries bodily and ORIGIN is
where it appears. Do not read "ORIGIN did not fire" as "the anchor was right".

## Current status

No sham sessions exist, so `--all` correctly reports nothing to verify. Both checks are
in place before the first one is recorded, which is the point.

## Incidental finding

Running the check surfaced the chatter guard: PILOT01 has a median gap of 2.00 s between
prompt changes across 492 changes, which flags it as a pre-fix chattering session and
therefore **unusable as a yoke source**. The seeding plan in §2 of the analysis plan
nominates the operator's post-fix pilot (PILOT02) for this, and that session does not
exist yet. Recording it is a prerequisite for the first sham-first participant.
