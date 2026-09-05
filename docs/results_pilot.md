# Pilot session: what a dress rehearsal found

PILOT01, 2026-08-22. One 20-minute closed-loop session, self-administered, headset +
LSL + library engine running together for the first time. Raw data in
`sessions/PILOT01_20260822_153652/`, metrics in that directory's `metrics.json`.

**This is a system-validation run, not data.** Single arm, unblinded, and the operator
was the participant and knew the target. Nothing here is evidence about whether the
intervention works, and the outcome numbers below are reported only because they
describe how the instrument behaved.

The rehearsal passed on everything it was designed to check and failed on three
things nobody was looking for. Those three are the content of this document.

---

## 1. The sensing path is sound

| | result | concern threshold |
|---|---|---|
| baseline rejection | 3.5% | — |
| intervention rejection | **13.1%** | >40% would have been a problem |
| buffer underruns | **0** | any is a bug |
| library prompts missing | **0** | any means coverage drift |
| segments played | 629 | — |

The coupling index needs ≥40 valid intervention windows; at 13.1% rejection a
20-minute session yields over 1000. Signal quality is not the limiting factor.

Outcome numbers, for completeness and not for interpretation: time in band 39.8%,
time to target 167 s, z mean −0.94 against a target of −1.0.

---

## 2. The controller was chattering, and the library engine exposed it

**629 audio events in 1200 s. 491 were prompt changes, not segment exhaustion.**
Median gap between switches 1.35 s; **30% arrived faster than the 1.0 s crossfade
could finish**, so the output was a near-continuous blend of two independent renders
rather than music with transitions in it.

The ladder was not the problem. The rung was stable — rung 1 for 95.9% of the
session. What flapped was the **trend suffix**, cycling through all four variants of
that one rung:

| suffix | share of session |
|---|---|
| holding steady, minimal variation | 35.8% |
| gradually more present | 23.2% |
| softer and slower, receding | 21.3% |
| (base, no suffix) | 15.6% |

### Root cause

`trend` was `z - previous_z`, a raw one-hop difference. Measured on this session:

| quantity | value |
|---|---|
| trend estimator sd (1 hop) | 0.275 |
| decision threshold | 0.05 |
| genuine drift being described | 0.00088 per hop |
| **signal-to-noise** | **0.003** |

**The threshold was five times smaller than the noise it was thresholding.** Sign
flipped on 29% of hops, threshold crossed on 24%, and every crossing rewrote the
prompt string.

### Why it had never been seen

Under the streaming engine a prompt was sampled once per 8 s segment, so chatter was
absorbed by the segment boundary. The library engine responds in 1 s and therefore
followed a signal that was mostly noise faithfully. The 8× latency improvement did
not create this defect; it removed the thing hiding it.

### The fix, and its honest limit

`trend` is now a least-squares slope over 20 hops (estimator sd 0.275 → 0.068), and
`build_prompt` applies a dual-threshold hysteresis band. Replayed against the pilot's
own z:

| | changes | rung | suffix-only | mean dwell | median gap | gaps < crossfade |
|---|---|---|---|---|---|---|
| before | 477 | 24 | 453 | 2.2 s | 1.4 s | **30%** |
| after | **24** | 24 | **0** | 43.5 s | 4.0 s | **0%** |

**The suffix was removed entirely on 2026-09-05, and this table is why the second fix
was not the end of it.** Look at the `suffix-only` column: the fix took it from 453 to 0.
A control contributing nothing to 24 remaining changes is either perfectly calibrated or
inoperative, and this table cannot tell the two apart. It turned out to be the second —
the slope the suffix gated on is smaller than the noise of the estimator measuring it
(1.8x on this session), so no threshold could ever have worked. Calibrating it above the
noise did not make it correct, it made it silent. See [deviations.md](deviations.md).

The numbers below stand as the record of what the calibration achieved; the control they
describe no longer exists.

Read the two dwell columns together. Mean dwell rises to 43.5 s, but the remaining
rung changes **cluster** — the median gap between consecutive changes is 4.0 s, and 13
of 23 gaps still fall inside a single 8 s segment. What the fix guarantees is the
property that actually failed: **no switch arrives before the previous crossfade can
finish**, down from 30% of switches. Clustered genuine changes are the controller
tracking a participant who is genuinely moving; sub-crossfade switching was the
controller tracking noise.

The fix had a consequence I did not check for at the time. With the suffix inert, the
controller reaches only the base prompts — and the library had been built with 4
renders of each of 20 prompts on the assumption all 20 were live. A 17-minute session
therefore reached **12 of 80 segments**: 32 seconds of unique audio, looping. That is
arguably a worse failure than the chatter it replaced. The library was rebuilt with
renders allocated by actual use (32 per base prompt, 220 total), taking the dominant
prompt from 32 s to 256 s of unique audio, each segment heard about 4 times in a
20-minute session and entered at a random offset.

**The suffix is now inert in a normal session, and that is a finding rather than a
workaround.** No plausible excursion reaches a 20-hop slope of 0.35: the pilot's
entire z range was 4.8 units over 20 minutes, and a 3-unit move compressed into 20 s
only reaches 0.15. Whether a branch that cannot fire should stay in the code is an
open design question.

The first calibration attempt set ENTER at 0.20, which turned out to sit on the
largest slope the session ever produced (0.1998). It fired zero times out of 1023 by
a margin of 0.0002 — an accident another session would cross arbitrarily. 0.35 is
clear of the observed ceiling, so "does not fire" is a property of the calibration.

---

## 3. A 20-minute session is worth 25 observations, not 1200

| quantity | value |
|---|---|
| valid intervention windows | 1043 |
| lag-1 autocorrelation | **0.953** |
| decorrelation time (1/e) | 9 s |
| **AR(1) effective sample size** | **25.3** |

Any analysis treating windows as independent — a t-test over windows, a correlation
p-value, a binomial interval on time-in-band — overstates its evidence by
**√(1043/25.3) ≈ 6.4×**. That is the distance between p = 0.05 and p = 0.4.

This is invisible unless looked for, and it is the easiest available route to
publishing something that does not replicate. Every inferential number in this
project must either use the effective sample size or use a permutation null that
preserves the autocorrelation. `coupling_index` and `event_locked_response` both use
the latter.

---

## 4. The coupling index fails exactly when the therapy works

The continuous ACI returned **r = −0.054 at p = 0.795** — a clean null. The estimator
is not at fault: `scripts/validate_coupling.py` shows it recovering known lags at
r ≈ 0.85 with p < 0.001, and catching a sign inversion that a real bug had introduced.

The reason is structural. `coupling_index` cross-correlates the whole session, so it
needs the audio to vary. A participant who reaches target and stays keeps the
controller on one rung — 95.9% here — so there is no controller-driven variation left
to correlate against.

**The primary mechanism measure is weakest precisely when the intervention is working
best.** That is not fixable by improving the estimator, and it is a bad property for a
dependent variable.

### The companion measure

`event_locked_response()` conditions on rung changes instead: discrete, timestamped
moments when the music demonstrably changed. On PILOT01 it returns **+0.412 z at
p = 0.104** from 10 usable events, where the continuous index found nothing.

**This number is not causal on its own, and must never be reported as though it
were.** A rung change happens *because* z moved; z then keeps moving because it is
autocorrelated and whatever drove it did not stop. A positive effect is exactly what a
closed loop with no therapeutic effect produces — the music follows the brain, the
brain continues, and time-locking to the follow makes it look like a lead.
Baseline-correcting does not remove this, because the pre-window movement is what
triggered the event.

The yoked sham is what makes it causal: the same changes at the same times, driven by
a different brain, is an estimate of exactly this confound. **Adaptive minus sham is
the only interpretable quantity.**

---

## 5. Sample size, and a design recommendation

Simulated from the pilot's measured autocorrelation — never solved in closed form,
because nothing here is independent. n **per arm** for 80% power at α = 0.05, outcome
`z_mean`:

| effect (z) | independent, bSD 0.3 | bSD 0.5 | bSD 0.7 | **paired** |
|---|---|---|---|---|
| 0.2 | 50 | >60 | >60 | **15** |
| 0.3 | 25 | 60 | >60 | **8** |
| 0.5 | 10 | 20 | 40 | **6** |
| 0.8 | 6 | 10 | 15 | **4** |

**The independent design the code currently implements needs roughly 60 per arm — 120
people — to detect a 0.3 z effect. A within-participant crossover needs 8.** At this
project's scale that is the difference between infeasible and a fortnight of sessions.

Switching means yoking each participant to their own earlier session rather than a
different person's. `--yoke-from` already supports it; what changes is the protocol
and the counterbalancing, not the code.

### Outcome choice costs more than design tuning

n per arm, between-participant SD 0.5, 3000 simulated studies per cell:

| effect (z) | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|
| `z_mean` | >60 | 60 | 20 | 10 |
| `time_in_band` | >60 | >60 | >60 | 20 |

**Powering on time-in-band costs roughly a factor of three in detectable effect.** It
dichotomises — "inside the band or not" discards how far inside or outside, and
thresholding a continuous measure always loses power — and it saturates, because a
participant sitting well outside the band scores near zero in *both* arms.

Saturation also removes most of the paired design's advantage: offsets cancel exactly
in a difference of means and do not cancel through a nonlinear function. So the
crossover recommendation above is specific to `z_mean`.

Report time-in-band descriptively, since it is the interpretable clinical quantity.
Power the study on mean z.

Two caveats that belong next to the table:

- **Between-participant SD is swept, not estimated.** One participant cannot estimate
  it. Narrowing that range is the main thing the first two or three real participants
  buy.
- **The paired column is a floor, not an estimate.** It does not move with bSD because
  participant offsets cancel exactly in the simulation. Real crossover data has a
  participant-by-condition interaction — some people respond and some do not — which
  this does not model. Inflate those numbers if there is any reason to expect
  heterogeneous response.

---

## 6. What the pilot licenses

**Established:**

- The sensing path produces usable data at 13.1% rejection over 20 minutes
- The library engine sustains a full session with zero underruns and zero coverage misses
- The analysis pipeline completes end to end on real closed-loop data
- Windows are not independent, and every inferential number must account for it

**Not established, and not addressable by more pilots:**

- Whether the intervention does anything. That needs both arms, blinding, and
  participants.
- Whether the crossfades sound acceptable. The pilot's audio was chattering, so it was
  not representative — that judgement needs a session run after the fix.

**Open design questions this raised:**

- Should the inert trend suffix be removed rather than left unable to fire?
- Should the ladder be documented as three rungs, or two, given the measured occupancy?
- Independent or crossover design, given the 60-vs-8 gap?

---

## Reproducing

```bash
python src/analyze_session.py sessions/PILOT01_20260822_153652
```

```bash
python scripts/power_analysis.py --sims 3000
```

```bash
python scripts/ladder_policy.py --sessions "sessions/PILOT01_*"
```
