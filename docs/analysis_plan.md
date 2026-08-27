# Analysis plan (pre-registration draft)

**Status: DRAFT. Two decisions are still open and are marked `[DECIDE]`. Finalise them,
then freeze this file and record its commit hash before the first participant.**

The point of writing this before data exists is that every choice below becomes
harder to make honestly afterwards. Which outcome is primary, what counts as an
exclusion, and how the autocorrelation is handled all move a p-value, and all of them
look like reasonable judgement calls when made with the data in front of you.

Every number cited comes from PILOT01 (2026-08-22) or the benchmark set. Sources are
named so a reviewer can check them.

---

## 1. Hypotheses

**H1 (primary).** Participants in the adaptive arm reach a lower mean arousal index
than participants in the yoked-sham arm, where arousal is `z`, the frontal
log(beta/alpha) ratio normalised against each participant's own resting baseline, and
the target is z = −1.0.

**H2 (secondary, mechanistic).** Audio-neural coupling is stronger in the adaptive arm
than in the yoked sham. Because the sham reproduces the same acoustic sequence with the
brain-music contingency broken, a difference cannot be produced by regression to the
mean, by the passage of time, or by the music alone.

**Directional and one-sided in intent, tested two-sided.** The design predicts a sign;
the tests do not assume it. A significant effect in the wrong direction is a result,
not a null.

---

## 2. Design

`[DECIDE]` **Independent-groups or within-participant crossover.** This changes
recruitment, consent language, and session count, so it must be settled before
submission. The evidence:

| | independent | crossover |
|---|---|---|
| n per arm for a 0.3 z effect | ~60 | **8** |
| sessions per participant | 1 | 2 |
| between-participant variance | must be absorbed | cancels |
| order and carryover effects | none | must be counterbalanced, needs washout |

Source: `scripts/power_analysis.py`, simulated from the pilot's measured
autocorrelation.

The crossover is dramatically cheaper and is the only version feasible at this
project's scale. Its costs are real: the paired sample sizes are a **floor**, because
the simulation cancels participant offsets exactly while real crossover data carries a
participant-by-condition interaction it does not model. Carryover is the specific
threat — a relaxation session on day 1 plausibly shifts the day-2 resting baseline, and
counterbalancing controls order without controlling carryover. If crossover is chosen,
specify a washout of at least 48 hours and test for a period effect before pooling.

**Blinding.** Participants are not told which arm a session is. The operator cannot be
blinded, because the sham requires passing `--yoke-from`. Record this as a limitation;
it is not solvable with the current implementation.

**Yoking.** Each sham session replays a prior adaptive session's prompt schedule on its
original timeline. Known fidelity limit, measured rather than assumed: the replay
reproduces the source's prompt sequence exactly but runs **0.4–1.0 s early throughout**,
and the source's first prompt is skipped. Describe the arms as *matched in sequence and
duration with a known sub-hop lead*, never as identical.

---

## 3. Outcomes

### Primary

**Mean z across valid intervention windows.** One number per session.

Chosen over time-in-band deliberately. Powering on time-in-band costs roughly a factor
of three in detectable effect: it dichotomises, discarding how far inside or outside the
band a participant is, and it saturates, since someone well outside the band scores near
zero in *both* arms. Source: `scripts/power_analysis.py --outcome time_in_band`.

### Secondary

| outcome | what it is | reported how |
|---|---|---|
| time in band | fraction of windows with \|z − target\| ≤ 0.5 | descriptive; the interpretable clinical quantity |
| time to target | seconds to first reach and hold the band 30 s | descriptive; `nan` when never reached, never coded as session length |
| ACI | lagged audio-neural coupling, peak r and lag | H2 |
| event-locked response | z change time-locked to rung changes | H2, **contrast only** |

**The event-locked response is not interpretable within a single arm.** Rung changes are
*triggered* by z moving, so a positive effect is what a closed loop with no therapeutic
effect also produces. PILOT01 shows the signature plainly: z rises from −0.45 to +1.0 in
the ten seconds *before* the change, peaks at onset, then decays. Only adaptive minus
sham removes it. `analyze_session.py` prints `elr_pre_slope_z_per_s` and
`elr_decays_after_onset` beside the effect for exactly this reason.

### Not outcomes

Rejection rate and buffer underruns are **signal-quality metrics**, reported for both
arms because reviewers will ask and because a between-arm difference would itself be a
confound. They are not dependent variables and will not be tested for an effect.

---

## 4. Sample size and stopping

n is set by `scripts/power_analysis.py` for 80% power at α = 0.05, given the design
chosen in §2 and a smallest effect of interest of **0.3 z**.

`[DECIDE]` **Smallest effect of interest.** 0.3 z is used above as a working figure and
is not yet justified clinically. Justify it or replace it before freezing — a sample
size computed from an arbitrary effect is not a sample size.

**No optional stopping.** The n is fixed in advance. Data will not be inspected for
significance and then extended. If recruitment falls short, the shortfall is reported
and the analysis runs on what was collected, reported as underpowered.

**Between-participant SD is unknown** and is swept in the power table rather than
assumed. After the first three complete pairs, recompute the power table with the
observed SD and record the update here — as a documented revision, not a silent one.

---

## 5. Analysis

### Windows are not independent

PILOT01: lag-1 autocorrelation **0.953**, decorrelation time **9 s**, so 1043 valid
windows carry an effective sample size of **25.3**. Any test treating windows as
independent overstates its evidence by about **6.4×** — the distance between p = 0.05
and p = 0.4.

**Therefore:** no test is ever run across windows. Each session collapses to one number
per outcome, and tests run across *sessions*. Where a within-session p-value is
unavoidable, it comes from a permutation null that preserves the autocorrelation
(circular shift for the ACI, shuffled onsets for the event-locked response), never from
a parametric test.

### Primary test

- Crossover: paired t-test on per-participant differences, with a period effect tested
  first; if significant, first-period data only.
- Independent: Welch's two-sample t-test.

Effect size as Cohen's d with a 95% CI. **The CI is the result**, not the p-value.

### Multiplicity

One primary outcome, one test. Secondary outcomes are reported with CIs and are
explicitly **not** corrected, because they are not being used to claim an effect. No
secondary outcome is promoted to primary after seeing the data.

### Exclusions, defined now

A session is excluded if any of the following, all checked automatically by
`analyze_session.py`:

| criterion | threshold | why |
|---|---|---|
| baseline failed | <10 valid windows | no baseline means no z-scores at all |
| intervention rejection rate | >40% | leaves too few windows for the coupling analysis |
| `session FAILED` recorded | any | the control loop crashed; the run is incomplete |
| buffer underruns | >0 | audio dropped out, so the participant did not receive the intervention |
| `audio_chattering` true | any | switches faster than a crossfade; not acoustically representative |

Excluded sessions are reported with their reason. **No session is excluded on the basis
of its outcome value.**

`audio_chattering` also disqualifies a session as a `--yoke-from` source, which
currently rules out every session recorded before 2026-08-16.

---

## 6. What would falsify H1

Stated in advance so a null is a result rather than a disappointment:

- No difference in mean z between arms, with a CI tight enough to exclude 0.3 z.
- A difference in the wrong direction.
- A difference that disappears once excluded sessions are handled as specified.
- In crossover, a significant period effect, which would mean the arms are not
  comparable and the design has failed regardless of the outcome.

**Both arms improving equally is the most likely null and the most informative one.** It
would indicate the music helps and the contingency does not — which is what the yoked
sham exists to detect, and a publishable finding.

---

## 7. Known limitations, recorded before the fact

- **The ladder is narrower than described.** Only rungs 1–3 are reachable, and PILOT01
  used rung 1 for 96% of the session. Do not describe the controller as five graded
  levels.
- **The trend suffix cannot fire.** After the hysteresis fix it is calibrated above the
  measured noise ceiling. The controller is effectively a three-rung ladder.
- **Frontal alpha only.** The Muse has no occipital electrodes, so the arousal index
  rests on frontal channels where alpha is weaker.
- **Operator not blinded.** See §2.
- **Single site, one headset, one operator.**

---

## 8. Deviations

Any departure from this plan after freezing is recorded here with its date, reason, and
commit — including deviations that seem trivial. An unlogged deviation is
indistinguishable from an undisclosed one.

| date | deviation | reason |
|---|---|---|
| — | — | — |
