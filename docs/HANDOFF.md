# Handoff: everything a new session needs

**Written 2026-08-28. Updated 2026-09-05, and after the hardware session of 2026-09-17**
— start with §0, which is what that session found. Read this first in a fresh conversation. It carries the state,
the numbers, the open decisions, and — most importantly — the traps, because several of
them cost hours and look invisible.

Verify the state is still current before trusting anything below:

```bash
python scripts/run_tests.py && python scripts/verify_claims.py
```

Expected: **110 passed, 0 failed, 1 skipped** and **44 claims reproduce**. The skip is
the residual-contingency positive control, which needs a session recorded with the current
ladder; none exists yet. `verify_library` fails until the four new prompts are rendered —
see §0.5. Takes ~100 s; `--quick` cuts it to ~65 s by skipping everything that recomputes from raw data, and says so rather than looking clean. If either disagrees,
something changed after this was written and the numbers here are stale.

---

## 0. The 2026-09-17 hardware session — what it found

The session in `next_session.md` was run. Two recordings, both clean enough to use:
`alphatest_20260917_214726` (AF7/AF8 alpha validation) and `PILOT02_20260917_220343`
(20 min at deployed settings). The headline is not either recording — it is two findings
that change what the study can be.

### 1. PILOT02 played ONE prompt for twenty minutes

0 prompt changes, 301 segments of "calm ambient piano", 0 underruns, 0 sub-crossfade
switches, 7.5% rejection. Technically perfect, and useless as a yoke source.

This is the controller working as designed. With the relaxation target (z = −1), the goal
is rung 1 and the controller always moves one rung toward it, so **every z from −∞ up to
+0.5 outputs rung 1**. The music only changes when z rises 1.5 SD *above* target. PILOT02
sat at z = **−1.46** (SD 0.67), at or past target from the first window. PILOT01 was the
same, less extremely (rung 1 for 95.9%).

Consequence: for any participant who relaxes, the adaptive arm is a fixed one-prompt
playlist, and a yoked sham replaying it is **acoustically identical**. The adaptive-vs-sham
contrast has no stimulus difference to test. **This is the study's biggest open design
problem**, it is a therapeutic/design decision, and any fix is a deviation from the frozen
plan. See §7 question 8.

(PILOT02 also sat 1.46 SD below its own baseline from the first second of music. n = 1,
no sham, could be sound onset itself — descriptive only.)

### 2. The AF7/AF8 alpha validation is mixed, not a clean pass

| | 9/17 AF7/AF8 | Aug AF7/AF8 | Aug TP9/TP10 |
|---|---|---|---|
| alpha_test ratio (arithmetic) | 1.84× | 0.91× | 1.85× |
| logged-window ratio (geometric) | **1.34×** | — | 2.13× |
| d (log alpha) | 0.61 | −0.17 | 1.23 |
| p | 5 × 10⁻⁷ | 0.26 | 2 × 10⁻²⁵ |
| rejected | 6.8% | 48% | 3% |
| **eye-closure prominence ratio** | **AF7 1.09, AF8 1.13 — FAIL (< 1.2)** | AF7 0.75 | TP9 2.46, TP10 1.91 (9/17) |

Alpha *power* rises significantly on eye closure, with good contact and balanced windows
(151 open / 143 closed). But the validated diagnostic — does a distinct alpha *peak* rise —
fails on both frontal channels while passing clearly on TP9/TP10 in the same recording, and
the median ratio is only 1.23×. Classification against `next_session.md` step 2: **"effect
present but weak"** — usable, reporting frontal SNR as a measured limitation. The script's
own "PASS … belongs in the paper" banner overstates it.

**Figure 0 has NOT been replaced.** The manuscript's alpha claims are now pinned to the
August recording (`PINNED_ALPHA` in `verify_claims.py`); the 9/17 numbers are asserted as
separate claims. Replacing Figure 0 with a weaker, mixed result is a decision about the
paper — §7 question 9.

### 3. The dominated-configuration result holds on AF7/AF8

Sweep on the 9/17 recording (reproduction r = 0.982): **7 of 9** alternatives beat the
deployed setting on both axes (8 of 9 on TP9/TP10). Everything is roughly halved — deployed
d 1.06 (vs 1.99), info/min 1.14 (vs 2.14) — and the latency floor gives **1.94×** the
deployed information rate (1.83× on TP9/TP10). Detection latencies on this recording are
noisy (5 transitions, weak signal, non-monotonic across τ); don't quote them.

### 4. Three recording bugs, all found at the headset, all fixed

- **BlueMuse duplicated packets** — trap 9. It is why the first contact checks read
  "RAILING". Filtered in `stream_utils.DedupInlet`; the first version of that filter was
  itself wrong.
- **Raw timestamps lost precision with laptop uptime** — trap 10.
- **A syntax error in `alpha_test.py` passed all 105 tests**, because nothing in the suite
  parsed the headset-only scripts. The suite now compiles every file.

The estimator sweep also counted samples to keep time, which the two 9/17 dropouts (11.75 s,
20.25 s) broke; it now uses `session_logger.sample_clock`. And eight tests plus Figures 0 and
6 were quietly selecting "the newest" recording — the same bug as the manuscript claims,
further in. All pinned; a test guards it.

### 5. The one-prompt controller is fixed (2026-09-20)

**Nine rungs at 0.5 SD, and `--min-dwell` now defaults to 30 s.** Replayed through the
deployed controller, PILOT01 gives 33 prompt changes across 5 distinct prompts and PILOT02
26 across 3, with zero switches inside a crossfade — against 24/3 and **0/1** before.

The rung WIDTH was the binding constraint, not the target. A personalised target — the
intuitive fix, and the one recommended here before the simulation was run — produces **0
changes on PILOT02**, because sliding the target down leaves the same mapping. Full
reasoning, the four options and the width/dwell sweep: `deviations.md`, 2026-09-20.

The original five prompts are the **even** rungs, so the existing library still covers five
of nine. **`build_library.py` must be run to render the four odd rungs** — it needs the GPU,
takes roughly half an hour, and `verify_library` fails until it is done. That failure is the
gate working, not a regression.

### What to do next

1. **Render the four new prompts:** `python scripts/build_library.py`. Then
   `python scripts/verify_library.py` should pass.
2. **Record a new yoke source** with the current controller. PILOT02 cannot serve as one
   (0 prompt changes), and neither can anything older. This is now the only thing blocking
   the sham arm.
3. ~~Decide whether Figure 0 is replaced (§7 q9).~~ **DONE 2026-09-20: report both**, the
   August TP9/TP10 validation and the 9/17 AF7/AF8 one, rather than substituting either.
   Written into **§5** of the draft (not §6 — §5 is where Figure 0 lives), the §8
   limitation updated to match, and `make_figures.py` now renders both panels as
   `fig0_alpha_validation.png` and `fig0b_alpha_validation_af78.png`.
3. The link: duplicates were 47% during the alpha test and 0.7% during PILOT02 after a
   BlueMuse restart; two dropouts in the first recording, none in the second. Restart
   BlueMuse fully before every session and keep the laptop within a metre, in front.

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
| closed-loop sessions | 2 (self-administered, unblinded; PILOT02 had 0 prompt changes) |
| adaptive/sham pairs | **0** |
| participants | **0** |

No claim about whether the intervention helps anyone can appear anywhere.

---

## 2. Current state

| | |
|---|---|
| tests | 110 passing, 1 skipped (`scripts/run_tests.py`) |
| verified claims | 44 (`scripts/verify_claims.py`) — 39 manuscript, pinned to PILOT01 and the August alpha test; 5 describing 9/17 |
| figures | 7, at 300 dpi (`docs/figures/`) |
| pre-registration | **FROZEN**, tag `preregistration-v1`, commit `e45bd32` |
| plan sha256 (LF) | `538328a2dac75fc9bab76fecb7f7cfa11ef88db9b08f6cf7e187bd1fe4fe4ce5` |
| preprint | §1–§8 drafted (`docs/preprint_draft.md`) |
| library | 220 segments, 29.3 min audio, gitignored (manifest is committed). **Covers 5 of the 9 rungs — 4 need rendering, see §0.5** |

**The pre-registration is hash-checked by the test suite.** Editing
`docs/analysis_plan.md` fails `run_tests.py`. That is deliberate. Post-freeze changes go
in `docs/deviations.md` as dated deviations (not §9 — §9 is inside the hashed file), or
become `preregistration-v2`.

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

8 of the **9** alternatives beat the deployed setting **on both axes** (10 configurations were tested; one of them is the deployed setting). The smoother is
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

### Trap 8: a `--demo` or `--mock` run lands in `sessions/` and outranks the real one

Every selector took `sorted(glob(...))[-1]`, so the NEWEST match wins. Running
`alpha_test.py --demo` for thirty seconds to smoke-test the hardware gate wrote
`sessions/alphatest_<today>`, which sorts after the August recording — and **six
manuscript claims silently recomputed against a signal generator**: the eyes-closed
alpha ratio, the AF7/AF8 mismatch, and the entire estimator sweep.

`verify_claims` caught it, but only because those numbers had been locked hours earlier.
Before that, running the demo would have rewritten the alpha validation and Figure 6
from synthetic data with nothing to notice.

The marker always existed — `alpha_test` writes `sampling_rate_source: "demo"`,
`live_music` writes `"mock"`. **Nothing read it.** `session_logger.real_sessions()` does
now, and every selector feeding the manuscript goes through it. Use it rather than
`glob` for anything that reaches a number.

Since `sessions/` **is committed** (only `raw_eeg.f32` is ignored), a demo run followed
by `git add -A` would have put synthetic data in the study's dataset. Two further
defences, because a marker nothing reads is not a design to trust twice:

- a synthetic run is now **named** `DEMO_<participant>_<ts>`, so the mistake is visible
  in `ls` rather than only in a manifest field
- `.gitignore` excludes `sessions/DEMO_*`, so it cannot be committed

`is_synthetic()` checks the name first, so a demo interrupted before its manifest flushed
still reads as synthetic — the wrong way round to fail would be reading as real.

---

### Trap 9: BlueMuse can deliver every packet two to four times

Found 2026-09-17. `verify_sample_rate` counted **1048 samples/s against a median spacing of
3.906 ms (256 Hz)** — both cannot be true. Each 12-sample packet arrived, timestamps stepped
back 43 ms, and the same packet arrived again: 8292 received in 8 s, **2052 unique**. The
EEG was fine. Four copies at first; two after restarting BlueMuse *and* resetting Windows
Bluetooth; 0.7% during PILOT02.

Every script counts samples, not time, so duplicates read as the signal jumping backwards
every 12 samples — which is what `contact_check` reported as AF7/AF8 **"RAILING"** on a
headset that was probably fine. Nothing downstream checked.

`stream_utils.get_inlet` now returns a `DedupInlet`, so all four live scripts are covered.
**The first version was wrong**: it dropped any sample whose timestamp did not advance, and
lost 111 of 2568 *genuine* samples (4.3%), because copies interleave and a real packet can
arrive after a copy of a later one. Checked by comparing against the true unique set rather
than accepting "249 Hz, close enough". The fix dedups by timestamp *identity* and holds
samples 50 ms to reorder (measured lateness: max 13.9 ms). Sessions log `stream_*` counts.

**Before any session:** end BlueMuse *and* LSL Bridge in Task Manager, reopen, Start
Streaming **once**, then check the stream is duplicate-free before trusting a contact check.

### Trap 10: raw timestamps depended on how long the laptop had been up

`SessionLogger.log_raw` cast LSL time straight to float32. LSL time is seconds since boot,
so precision was ~8 ms for PILOT01 (clock ~70,000 s) and **0.25 s for both 9/17 recordings**
(clock ~2,339,000 s, 27 days up). Fixed: timestamps are stored relative to the first sample
(~0.1 ms over a session) and the absolute origin is logged as a `raw time origin` note.

The two 9/17 recordings keep their quarter-second timestamps. Use
`session_logger.sample_clock` — sample counting, corrected for dropouts located from the
timestamps — for anything that aligns them by time. It returns exactly `arange(n)/fs` when
there are no dropouts, so nothing already correct moves.

---

### Trap 7: every session on disk is disqualified as a yoke source

Pre-fix ones for chatter; **all** of them for a 7.06 s replay-origin bias (fixed 8/28, but
the bias is baked into existing logs). **The sham arm cannot run until PILOT02 exists.**

**A third reason, added 9/5, and it is structural.** Removing the trend suffix shrank the
reachable prompt space from 20 strings to 5. PILOT01's schedule carries a suffix on
**368 of its 492 entries**, so a sham yoked to it would play music the adaptive arm
**cannot reach at all**. That is worse than an acoustic mismatch: the arms would differ in
their prompt vocabulary, which is the one thing yoking exists to control for.

Nothing would have reported it. The replay would be faithful, the library still holds
those segments, the audio sounds fine, and every existing check passes.
`_warn_if_source_outside_prompt_space` now catches it, and it is asserted in both
directions — it must fire on a pre-9/5 source and stay silent on a current one.

The consequence for PILOT02: **a yoke source must be recorded with the current
controller.** Any session predating 9/5 is unusable as one regardless of its quality, so
this is not a bar PILOT02 has to clear so much as a reason no earlier session can.

**Update 2026-09-17: PILOT02 exists and is also unusable, for a new reason.** It is clean,
current-controller, chatter-free — and has **zero prompt changes**, so a sham replaying it
is identical to the adaptive arm. See §0.1. The sham arm is still blocked, now on a design
decision rather than on a recording.

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
SRC approval ─────────────────────────────────────► participant data
                                                         ▲
AF7/AF8 alpha validation   DONE 9/17 — mixed ────────────┤  (usable with a stated limitation)
                                                         │
controller produces a stimulus difference   DONE 9/20 ───┤  (9 rungs at 0.5 SD + 30 s dwell)
        │                                                │
        ├──► render 4 new library prompts ───────────────┤  (build_library.py, ~30 min GPU)
        │                                                │
        └──► a yoke source recorded with THIS controller ┘  (sham arm impossible without it)
```

`docs/next_session.md` was run on 2026-09-17; its results are in §0. **The next hardware
session cannot be specified until §7 question 8 is decided** — recording another
deployed-settings pilot would reproduce PILOT02.

---

## 7. Open questions

1. ~~**Does the eyes-closed effect hold on AF7/AF8 with good contact?**~~ **Answered
   9/17: partly.** Alpha power rises (1.34× geometric, d = 0.61, p = 5 × 10⁻⁷, 6.8%
   rejected), but the eye-closure prominence check fails on both frontal channels (AF7
   1.09×, AF8 1.13×) while TP9/TP10 pass (2.46×, 1.91×). "Effect present but weak." §0.2.
2. ~~**Should the inert trend suffix be deleted?**~~ **DONE 9/5 — deleted.** Not for being
   inert, but because the quantity it thresholded is **not measurable**: the largest
   genuine 60 s drift in PILOT01 is 0.0385 z/hop against slope-estimator noise of 0.0681
   (1.8x, and 3.9x under the retuned estimator). A threshold above the noise can only be
   crossed by noise; one low enough to catch real drift fires constantly, which is the
   8/16 defect. It was also a hazard rather than dead weight — ENTER 0.35 sat only 1.16x
   above the retuned estimator's largest slope. Verified a no-op: **0 suffixed prompts
   across all 1212 logged windows**. Freed 15 of 20 library prompts and 60 of 220 segments
   (8.0 of 29.3 min). Logged in [deviations.md](deviations.md).
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
8. ~~**The controller cannot produce a stimulus difference for a calm participant.**~~
   **RESOLVED 9/20 — ladder widened to 9 rungs at 0.5 SD, dwell default 30 s.** The width
   was the binding constraint: a personalised target and in-band matching both give **0
   changes on PILOT02**, while 0.5 SD rungs give 102 (33 with the dwell). See
   `deviations.md`. **Still outstanding:** render the four new prompts, and record a yoke
   source with this controller — no existing session qualifies.
9. ~~**Replace Figure 0 with the AF7/AF8 recording?**~~ **DECIDED 9/20: report both.**
   The August TP9/TP10 validation and the 9/17 AF7/AF8 one, side by side — the first shows
   the rig measures cortex, the second shows what the study's own channels do, including
   the failed eye-closure check. `PINNED_ALPHA` stays on August; the 9/17 numbers are
   already asserted as their own claims. **Not yet written into §6 of the draft.**

---

## 8. File map

| path | what it is |
|---|---|
| `docs/HANDOFF.md` | this file |
| `docs/next_session.md` | **the next hardware session, step by step** |
| `docs/analysis_plan.md` | **frozen pre-registration — do not edit** |
| `docs/deviations.md` | the deviation log. Deliberately OUTSIDE the hash — §9 of the plan says to log deviations in the plan, which would break the freeze |
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
| `scripts/run_tests.py` | 106 checks, one command |
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
