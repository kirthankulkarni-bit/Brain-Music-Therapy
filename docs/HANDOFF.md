# Handoff: everything a new session needs

**Written 2026-08-28. Updated 2026-09-05** — trap 1 was partly wrong and is now
resolved; see the update inside it. Read this first in a fresh conversation. It carries the state,
the numbers, the open decisions, and — most importantly — the traps, because several of
them cost hours and look invisible.

Verify the state is still current before trusting anything below:

```bash
python scripts/run_tests.py && python scripts/verify_claims.py
```

Expected: **45 passed, 0 failed** and **21 claims reproduce**. If either disagrees,
something changed after this was written and the numbers here are stale.

---

## 1. What this project is

Closed-loop EEG music therapy. A Muse 2 headband streams 4 channels at 256 Hz over LSL;
`log(beta/alpha)` on frontal electrodes is z-scored against the participant's own 120 s
baseline; a controller maps z to one of a small set of text prompts; an audio engine
plays a matching precomputed segment.

**It is a systems and methods paper, not an efficacy study.** Evidence base, counted:

| | count |
|---|---|
| benchmark runs | 9 |
| closed-loop sessions | 1 (self-administered, unblinded) |
| adaptive/sham pairs | **0** |
| participants | **0** |

No claim about whether the intervention helps anyone can appear anywhere.

---

## 2. Current state

| | |
|---|---|
| tests | 34 passing (`scripts/run_tests.py`) |
| verified claims | 17 (`scripts/verify_claims.py`) |
| figures | 7, at 300 dpi (`docs/figures/`) |
| pre-registration | **FROZEN**, tag `preregistration-v1`, commit `e45bd32` |
| plan sha256 (LF) | `538328a2dac75fc9bab76fecb7f7cfa11ef88db9b08f6cf7e187bd1fe4fe4ce5` |
| preprint | §1–§8 drafted (`docs/preprint_draft.md`) |
| library | 220 segments, 29.3 min audio, gitignored (manifest is committed) |

**The pre-registration is hash-checked by the test suite.** Editing
`docs/analysis_plan.md` fails `run_tests.py`. That is deliberate. Post-freeze changes go
in §9 as dated deviations, or become `preregistration-v2`.

`preregistration-v1` is an **annotated** tag, so `git rev-parse preregistration-v1`
returns the tag object (`7a94667`), not the commit. Use `preregistration-v1^{commit}` to
get `e45bd32`. This is not a discrepancy; it looks like one.

Retrieve the frozen text with `git show preregistration-v1:docs/analysis_plan.md`.

---

## 3. The findings, with numbers

### Latency (the paper's core)

- Nothing reaches realtime. Best across 2 GPU tiers, 3 precision modes, 9 runs: **1.05×**
- The bound is the **sequential decode loop**, not arithmetic. A T4 gives fp16 ~8× its
  fp32 throughput on tensor cores; the workload got *slower*. `fp32 < fp16-half < fp16`
  in 6 of 6 cells.
- Between-run variance is a machine property: laptop up to **1.96×**, T4 **1.02–1.18×**.
- End-to-end worst case **6.5 s** (5.5 s analysis + 1.0 s crossfade). **85% is analysis.**

### The analysis path is a dominated configuration

Measured against labelled ground truth (`scripts/estimator_sweep.py`):

| estimator | detect | d | ind/min | info/min |
|---|---|---|---|---|
| **deployed** 4 s win, 1 s hop, tau=3 | 5.67 s | 1.99 | 1.2 | 2.14 |
| 2 s win, 0.5 s hop, tau=0.5 | 1.68 s | 1.14 | 13.6 | **4.20** |
| streaming o4, tau=0.25 | 0.17 s | 0.70 | 30.9 | 3.91 |

8 of 10 alternatives beat the deployed setting **on both axes**. The smoother is
simultaneously the largest latency term and the dominant cause of the autocorrelation
that collapses statistical power — shortening it pays twice.

**But see trap #1 below: you cannot simply retune.**

### Statistics

- PILOT01: 1043 windows, lag-1 rho **0.953**, **effective n = 25.3**. Treating windows as
  independent overstates evidence **6.4×**.
- The continuous coupling index fails *when the intervention works* — a participant near
  target keeps the controller on one rung, so there is nothing to correlate.
  PILOT01: r = −0.054, p = 0.795, on a validated estimator.
- Event-locked measures are confounded by their own trigger. PILOT01 gives +0.412 z
  (p = 0.104) — but z rises **before** the event and decays after, which is a trigger
  signature. Only adaptive − sham is interpretable.
- Dichotomised outcomes (time-in-band) cost ~3× in detectable effect.

### The index

- **Construct validity** (`scripts/validate_index_deap.py`, DEAP `s01.dat` is in the repo):
  rho = **+0.303** with self-reported arousal (p = 0.058), +0.082 with valence.
  **7/7 montages positive**, sign test p = 0.016, frontal-to-posterior gradient.
- **Seven candidate indices compared**: deployed ranks 3rd, but the gap to best is 0.020
  against SE 0.164 — indistinguishable. The inherited choice is defensible; the simpler
  alternatives are worse.

---

## 4. TRAPS — read these before changing anything

### Trap 1: do NOT retune the estimator without controller work

I recommended `--window 2 --hop 0.5 --tau 0.5` and it was **wrong**. Replayed against
PILOT01:

| | changes | median gap | under a crossfade |
|---|---|---|---|
| deployed | ~~36~~ 28 | 3.0 s | 0 |
| retuned | ~~459~~ **382** | 1.0 s | ~~168~~ **136** |

*Struck values are the original uncommitted measurement; the corrected ones come from
`scripts/controller_replay.py`. See the update at the end of this trap.*

36% sub-crossfade, worse than the original defect. **It is not the trend thresholds** —
recalibrating them changes nothing.

`state_rung(z)` is `round(2 + z)` with **no hysteresis**, so the rung flips whenever z
crosses a half-integer: 194 times deployed, 1006 retuned. The deployed system survives by
*accident* — `build_prompt` maps several `here` values onto one level, absorbing most
flips. Less smoothing spends that absorption.

**UPDATE 2026-09-05 — half of this trap is resolved, and the `ladder_margin` advice that
used to sit here was dangerous.** Read
[finding_ladder_hysteresis.md](finding_ladder_hysteresis.md) before acting on any of it.

Three things changed:

**The numbers above are not reproducible and are ~20% too high.** They came from
uncommitted code. `scripts/controller_replay.py` is the committed replacement, calibrated
against a known result: replaying PILOT01's *logged* z at deployed settings gives 24
changes, matching `verify_claims.py` exactly. It reads deployed 28 / retuned 382 / 136
sub-crossfade, not the 36 / 459 / 168 struck above. Two things had to be right first — the time axis must
come from the LSL timestamps (PILOT01 has two dropouts, worst 10.67 s) and the fidelity
check must compare unsmoothed against unsmoothed. Getting either wrong caps r at 0.66.

**`ladder_margin` was latched, and turning it on would have frozen the music.** At margin
0.25 it produced **zero prompt changes in twenty minutes** while the participant's rung
ranged to 4. `build_prompt` fed the rung being *played* back in as the previous *state*
estimate, and since play is always one rung toward the target, that closed a loop onto the
goal rung. `build_prompt` now takes `previous_rung` explicitly and `PromptGovernor` owns
it. Fixed, tested, and the test fails against the old wiring.

**The retuning is viable now.** The missing piece was a minimum dwell, and the value is
structural rather than tuned: **a dwell of one crossfade is exactly the condition for no
switch arriving before the previous crossfade completes.** Retuned + `--min-dwell 1.0`
gives 299 changes, 1.5 s median gap, **0 sub-crossfade**, and 3.68 s end-to-end against
the deployed 6.67 s. The dwell is not free — the library engine acts within one crossfade,
not one segment, so worst case is `dwell + crossfade`; an 8 s dwell would have eaten the
entire gain.

`live_music.py` **refuses to start** on a retuned estimator with `--min-dwell` below the
crossfade. Still unjudged: whether 299 changes in twenty minutes sounds acceptable. It is
click-free, which is not the same as good.

### Trap 2: the alpha validation used the WRONG CHANNELS

Figure 0 was recorded with `frontal_channels: TP9/TP10`. The live index uses **AF7/AF8**.

| pair | ratio | d | p | rejected |
|---|---|---|---|---|
| TP9/TP10 (validated) | 1.85× | 1.23 | 1.8e−25 | 3% |
| **AF7/AF8 (the index)** | **0.91×** | −0.17 | **0.26** | **48%** |

AF7 alone shows a **significant reversal** (p = 0.009) — it tracked blinks. **This gates
participant data.** See `docs/finding_channel_validation.md`.

### Trap 3: raw sessions do not align with logs the way you expect

- Raw columns are `[lsl_ts, ch0..ch3]`, and `MUSE_CHANNELS = (TP9, AF7, AF8, TP10)`.
  So AF7/AF8 are **raw columns 2 and 3**, not 1 and 2. Getting this wrong inverts the
  eyes-closed effect.
- Session `elapsed_s` and raw sample time have **different origins**. On the alphatest the
  offset is **+6.25 s**. Find it by correlating, never assume zero.
- Always read the channel pair from `manifest["index_channels"]`, never from memory.

### Trap 4: do not reimplement the feature extractor

The pipeline detrends, applies a **zero-phase** bandpass and 60 Hz notch *inside* each
window, then runs Welch. A hand-rolled Welch correlates **r = 0.05** with the logged
output. Use `FeatureExtractor` and verify **r > 0.9** against the session log before
trusting any comparison. `estimator_sweep.py` refuses to report below 0.9.

### Trap 5: units

`build_prompt` operates on **z** (normalised by baseline SD, 0.1912 on PILOT01). Raw
`log_beta_alpha` is ~8× smaller in scale. Thresholds computed in log units are wrong by
that factor. `calibrate_hysteresis.py` divides by the baseline SD and refuses sessions
without one.

### Trap 6: alpha is intermittent

Computing spectral prominence over a whole recording averages the peak away. Compute per
~10 s segment and take the **median**. A whole-session Welch scores a validated channel
as having no alpha peak.

### Trap 7: every session on disk is disqualified as a yoke source

Pre-fix ones for chatter; **all** of them for a 7.06 s replay-origin bias (fixed 8/28, but
the bias is baked into existing logs). **The sham arm cannot run until PILOT02 exists.**

---

## 5. Decisions already made — do not relitigate

| decision | value | where |
|---|---|---|
| design | within-participant crossover, **cross-yoked** (never self-yoked) | plan §2 |
| n | 10 participants, both arms, ≥48 h washout, counterbalanced | plan §2, §4 |
| smallest effect of interest | **0.15 z**, from neurofeedback meta-analytic g ≈ 0.3 | plan §4 |
| study type | **feasibility** — no p-value for the primary contrast | plan §1 |
| primary outcome | mean z (not time-in-band) | plan §3 |
| paper framing | measurement + methodology, **not** a system paper | `related_work.md` |

At n = 10, power for 0.15 z is **38%**. The primary outputs are a confidence interval
(±0.20 z) and two variance components.

---

## 6. What is blocked on what

```
SRC approval ──────────────────────────► participant data
                                              ▲
AF7/AF8 alpha validation ─────────────────────┤  (gates it)
                                              │
PILOT02 (clean yoke source) ──────────────────┘  (sham arm impossible without it)
```

**Next hardware session is fully specified in `docs/next_session.md`** — every command
verified to exist and take the flags quoted. Summary: contact gate on AF7/AF8 → alpha
validation on AF7/AF8 → estimator sweep on that recording → PILOT02 **at deployed
settings**.

---

## 7. Open questions

1. **Does the eyes-closed effect hold on AF7/AF8 with good contact?** Gates everything.
2. **Should the inert trend suffix be deleted?** It cannot fire after calibration.
3. **Should `ladder_margin` be turned on?** It *works* now — as of 8/28 it did not, and
   enabling it would have frozen the controller (trap 1). At deployed settings it gives
   28 → 8 changes with a 21 s median gap. Still off by default, still a therapeutic call.
7. **Does the retuned controller sound acceptable?** `--min-dwell 1.0` makes the retuned
   estimator click-free at 299 changes in twenty minutes, against 28 deployed. Click-free
   is a measurement; acceptable is a listening judgement, and nobody has made it. This
   gates whether the 1.8x latency gain is actually collectable.
4. **Does the information-rate gain survive the real contrast?** The n ≈ 7 projection
   assumes discriminability measured on eyes-open/closed transfers to adaptive-vs-sham.
5. **Do the crossfades sound acceptable?** Never judged on non-chattering audio.
6. ~~**Novelty check**~~ **DONE 9/5, and it changed the claim.** All six references in
   `related_work.md` §1 resolved individually. Ehrlich et al. (2019) is a **partial
   counterexample** — 4 s window, 0.5 s update rate, explicit reasoning about filter delay,
   but no end-to-end figure — so the claim narrowed to "no *measured, decomposed*
   end-to-end budget" and Ehrlich must be cited rather than found by a reviewer. Two
   citations were wrong in detail, Neurophone had no citation at all (it runs on a
   **Muse S**), and a 2026 near-neighbour was missing (Monroy-D'Croz et al.,
   arXiv:2606.01473 — prefrontal EEG over LSL into Ableton; no latency, no control
   condition, and their frontal alpha asymmetry explained **0.40%** of variance, which is
   published support for trap 2). **Still open:** *Mind to Music* is paywalled and unread,
   and its title advertises real-time operation.

---

## 8. File map

| path | what it is |
|---|---|
| `docs/HANDOFF.md` | this file |
| `docs/next_session.md` | **the next hardware session, step by step** |
| `docs/analysis_plan.md` | **frozen pre-registration — do not edit** |
| `docs/preprint_draft.md` | §1–§8 |
| `docs/related_work.md` | the arXiv sweep and what survives it |
| `docs/finding_channel_validation.md` | trap 2 in full |
| `docs/finding_analysis_latency.md` | the dominated-configuration result |
| `docs/finding_ladder_hysteresis.md` | trap 1 in full |
| `docs/results_latency.md`, `docs/results_pilot.md` | benchmark and pilot results |
| `src/eeg_features.py` | extractor, smoother, `StreamingBandPower` |
| `src/music_engine.py` | `build_prompt` — the controller |
| `src/library_engine.py` | the default audio path |
| `src/analyze_session.py` | outcomes, coupling, event-locked |
| `scripts/run_tests.py` | 45 checks, one command |
| `scripts/verify_claims.py` | regenerates all 17 manuscript numbers |
| `scripts/estimator_sweep.py` | latency vs information rate, needs a labelled session |
| `scripts/controller_replay.py` | replays a recording through the real controller; chatter counts |
| `scripts/signal_quality.py` | per-channel: cortex or ocular artefact |
| `scripts/calibrate_hysteresis.py` | trend thresholds for a given estimator |
| `scripts/validate_index_deap.py`, `compare_indices_deap.py` | index validity on DEAP |
| `s01.dat` | DEAP participant 1, gitignored, not redistributable |

---

## 9. Working conventions that earned their place

- **Test a tool against a known result before trusting it on an unknown one.** Six of my
  errors this project were caught that way and none any other way.
- **Read labels off the data, not off memory** — `manifest["index_channels"]` is how trap 2
  surfaced.
- **A test that cannot fail is not a safeguard.** The pre-registration hash check was
  verified by editing the file and watching it fail.
- **Commit messages carry the reasoning**, including what was wrong and how it was found.
  `git log` is the project's real notebook.
- Every number in the manuscript is in `verify_claims.py`. Add new ones there rather than
  letting prose drift from code.
