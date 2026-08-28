# Analysis plan (pre-registration draft)

**Status: DRAFT. Design is decided (§2). One decision remains, marked `[DECIDE]`.
Settle it, then freeze this file and record its commit hash before the first
participant.**

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

**DECIDED: within-participant crossover, cross-yoked.** Each participant completes both
an adaptive and a sham session; their sham replays a *different* participant's adaptive
schedule.

| | independent | **crossover (chosen)** |
|---|---|---|
| n per arm for a 0.3 z effect | ~60 | **8** |
| sessions per participant | 1 | 2 |
| between-participant variance | must be absorbed | cancels |
| order and carryover | none | counterbalanced, washout required |

Source: `scripts/power_analysis.py`, simulated from the pilot's measured
autocorrelation. 60 participants per arm is not feasible at this project's scale; 8 is.

**Cross-yoked, not self-yoked, and this matters.** The obvious shortcut is to yoke each
participant's sham to their own earlier adaptive session. It is acoustically matched and
the music no longer responds to them, so it looks valid. It is not: that schedule was
generated *by their own brain dynamics*, and people are consistent enough between days
that it may still partially track them. The contingency is then broken only partially,
the arms differ by less than intended, and the bias runs toward the null — the direction
that quietly wastes a study rather than producing a false positive.

Cross-yoking keeps the crossover's whole statistical advantage, since participant offsets
cancel in the paired difference regardless of whose schedule the sham replays.
`live_music.py` warns loudly if a session is self-yoked.

**Order and washout.**
- Order counterbalanced: half adaptive-first, half sham-first, assigned before enrolment.
- **Washout ≥ 48 hours** between a participant's two sessions.
- A **period effect is tested before pooling**. If significant, the arms are not
  comparable and the analysis falls back to first-period data only — which is an
  independent-groups comparison at n = 8 per arm, and therefore underpowered. Report it
  as such rather than pooling anyway.

**Seeding.** A sham needs a prior adaptive session from someone else, so the first
sham-first participant has no source. Seed from the operator's own post-fix pilot
(PILOT02), or run the first participant adaptive-first. Record which was done.

**The paired sample sizes are a floor.** The simulation cancels participant offsets
exactly, while real crossover data carries a participant-by-condition interaction it does
not model — some people respond and some do not. Treat 8 as a minimum, not a target.

**Blinding.** Participants are not told which arm a session is. The operator cannot be
blinded, because the sham requires passing `--yoke-from`. Record this as a limitation;
it is not solvable with the current implementation.

**Yoking.** Each sham session replays a prior adaptive session's prompt schedule on its
original timeline, reproducing the prompt-decision timeline exactly — verified to 0.00 s
across PILOT01's 492 changes. Sham resolution is bounded by the source's analysis hop
(1 s by default), which is well inside a crossfade.

Sessions recorded before 2026-08-16 replayed against the wrong origin and ran 7.06 s
early. That bias is baked into their logs and cannot be corrected retrospectively, so
those sessions are disqualified as yoke sources — as are any flagged `audio_chattering`.

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

n is set by `scripts/power_analysis.py` for 80% power at α = 0.05, for the crossover
design chosen in §2. The smallest effect of interest is justified below, and it is
**0.15 z**, not the 0.3 z used as a working figure while the design was open.

### Smallest effect of interest: 0.15 z, and what that costs

**One z unit is defined by the participant's own baseline.** On PILOT01 the baseline SD
of log10(beta/alpha) was 0.1912, so:

| effect | log10 units | change in beta/alpha |
|---|---|---|
| 0.15 z | 0.029 | 6.8% |
| 0.30 z | 0.057 | 14.1% |
| 1.00 z | 0.191 | 55.3% |

For scale, the eyes-open to eyes-closed manipulation in `fig0` is **1.7 z** (+113% alpha).
A therapeutic contrast should be expected far below a gross physiological one.

**Literature anchors.** Two classes, and the distinction matters:

| source | effect | in z here |
|---|---|---|
| Music vs *silence*: frontal beta −23% (monaural beats) | d = 1.0 | 0.59 |
| Music vs *silence*: frontal beta −36% (pentatonic) | d = 1.56 | 1.01 |
| Neurofeedback **neural modulation**, pooled | g = 0.34 (0.23–0.44) | — |
| Same, pre- to post-training | g = 0.26 (0.03–0.50) | — |
| Same, audio-only feedback subgroup | SMD = 0.28 (−3.19 to 3.75) | — |

The music-vs-silence effects are **upper bounds, not estimates**. Both arms here hear
music; the contrast isolates *contingency alone*, which must be smaller. The
neurofeedback neural-modulation figures are the right comparison class, and they cluster
at g ≈ 0.26–0.34.

**The unit conversion that decides the number.** `z` is standardised by *within*-participant
baseline SD; meta-analytic g is standardised by *between*-participant SD. They are not
the same currency, and conflating them is the easy mistake here. At a between-participant
SD of 0.5 z, a g of 0.3 corresponds to **0.15 z**, not 0.30 z.

So **0.15 z** is the literature-matched target. The previous 0.3 z working figure was
twice the effect the closest comparable literature reports.

### What that costs, stated plainly

| effect (z) | as d (bSD 0.5) | participants for 80% power |
|---|---|---|
| 0.59 (music vs silence — upper bound) | 1.18 | 4 |
| 0.30 (the old working figure) | 0.60 | 8 |
| 0.20 | 0.40 | 15 |
| **0.15 (literature-matched)** | **0.30** | **30** |

**At n = 10, power to detect 0.15 z is 38%.** A study with 38% power is more likely to
miss a real effect than find it.

Splitting the work differently does not rescue it. Effective sample size scales with time
on task, so averaging K sessions per arm shrinks each participant's SE by √K — and the
**total number of sessions stays near 60 regardless**:

| design | participants | sessions each | total sessions |
|---|---|---|---|
| 1 × 20 min per arm | 30 | 2 | 60 |
| 2 × 20 min per arm | 15 | 4 | 60 |
| 3 × 20 min per arm | 10 | 6 | 60 |

In a paired design the information is proportional to total time on task. Trading
participants for sessions-per-participant changes recruitment difficulty, not power.

### `[DECIDE]` The consequence, which is a scope decision rather than a statistical one

Three honest options:

1. **Power the study properly.** ~60 sessions, e.g. 15 participants × 2 sessions per arm.
   At ~45 min per session including setup, roughly 45 hours of lab time.
2. **Run it as an explicit feasibility study.** n = 10, powered only for effects ≥ 0.3 z.
   Report the **confidence interval and an estimate of the between-participant SD**, and
   state that it is not powered to test H1. This is legitimate and publishable — and the
   between-participant SD it yields is exactly what a properly powered follow-up needs,
   since that quantity is currently swept rather than known.
3. **Reduce the noise.** n_eff is limited by the 9 s decorrelation time. A shorter
   smoother or a longer session raises it, but re-validating the arousal index would be
   required first.

**Option 2 is the honest fit for the current scale**, and it changes what the paper claims
rather than weakening it: a feasibility study reporting a CI is a contribution, whereas an
underpowered study reporting p > 0.05 as evidence of no effect is not. Whichever is
chosen, the choice and its power must be stated in the manuscript before collection.

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

Paired t-test on per-participant adaptive−sham differences in mean z.

A period effect is tested first (adaptive-first vs sham-first participants). If
significant, the crossover assumption has failed and the analysis uses first-period data
only, as an independent-groups comparison — reported as underpowered rather than
presented as the planned test.

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

## 8. Sources

Literature anchors for the smallest effect of interest (§4). Accessed 2026-08-28.

1. **Neurofeedback neural modulation, meta-analysis.** *Systematic review and meta-analysis
   of the relationships between real-time neurofeedback training parameters and
   acquisition of neural modulation.* Frontiers in Human Neuroscience (2025).
   Pooled Hedges' g = 0.34 (95% CI 0.23–0.44) first-to-last session; g = 0.26 (0.03–0.50)
   pre- to post-training. Feedback-modality subgroups: complex 0.50 (0.29–0.71), simple
   audiovisual 0.59 (−0.01 to 1.19), simple visual 0.16 (0.07–0.25), simple audio 0.28
   (−3.19 to 3.75). 55 groups from 39 studies, mean N = 16.2.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12426165/

   *This is the primary anchor* — it measures neural modulation, which is this study's
   outcome class, rather than clinical symptom change. Note the audio-only subgroup's
   confidence interval is uninformative; the pooled estimate carries the weight.

2. **Relaxation audio, frontal beta reduction.** *Pentatonic sequences and monaural beats
   to facilitate relaxation: an EEG study.* Frontiers in Psychology (2024), N = 31.
   Frontal beta fell 0.13 → 0.10 µV²/Hz with monaural beats (d = 1.0, p = 0.04) and
   0.11 → 0.07 with pentatonic sequences (d = 1.56, p = 0.02). Alpha showed no significant
   change in either condition.
   https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1369485/full

   *Upper bound only.* This is audio versus silence; both arms here hear music.

3. **Relaxation training, clinical outcomes.** Ten-year systematic review with
   meta-analysis: d = 0.57 within-group, 0.51 between-group for anxiety.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2427027/

   Context only — these are symptom scales, not EEG, and are not commensurable with z.

---

## 9. Deviations

Any departure from this plan after freezing is recorded here with its date, reason, and
commit — including deviations that seem trivial. An unlogged deviation is
indistinguishable from an undisclosed one.

| date | deviation | reason |
|---|---|---|
| — | — | — |
