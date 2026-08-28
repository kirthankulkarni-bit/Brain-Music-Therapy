# Preprint outline: what this project can currently claim

Written before drafting, because the hardest editorial decision is which paper this is,
and that decision is made badly once prose exists to defend.

Every claim below is mapped to the artefact that supports it. Run
`python scripts/verify_claims.py` to regenerate all 14 headline numbers from the data
on disk; all currently reproduce.

---

## 1. What this paper is

**A systems and methods paper.** The contribution is a characterised latency floor for
closed-loop generative audio on consumer hardware, an architecture that works within it,
and a set of statistical cautions that apply to any closed-loop EEG study of this shape.

**What it is not: an efficacy study.** The evidence base, counted rather than
remembered:

| | count |
|---|---|
| benchmark runs | 9 |
| closed-loop sessions | **1** (PILOT01, self-administered, unblinded) |
| adaptive/sham arm pairs | **0** |
| participants | **0** |

No claim about whether the intervention helps anyone can appear in this manuscript.
Every outcome number from PILOT01 is reported as instrument behaviour, never as effect.

### Working title

> *Closed-loop EEG-driven music on consumer hardware: a latency characterisation and a
> precomputed-library architecture*

Names both contributions and promises no clinical result. Resist adding "for anxiety" or
"therapy" to the title — the study that would license either has not run.

---

## 2. Contributions, in the order they should be argued

**C1. Live generative audio cannot close the loop, and the reason is the decode loop.**
Not "our GPU was small". A T4 runs fp16 on tensor cores at ~8× its fp32 throughput;
giving the workload that made it *slower*, which rules out an arithmetic bound. Batch-1
autoregressive decoding is bound by the sequential token dependency, so no faster GPU
and no numeric format fixes it. This generalises beyond MusicGen to the workload class.

**C2. A precomputed library is a complete solution, not a fallback.** `build_prompt` is a
pure function with a finite range, so a library covering that range covers the
controller's entire output space exactly. This is the argument that makes the
architecture principled rather than expedient.

**C3. The win is removing commitment, not removing latency.** Streaming had to play a
queued segment to completion, so a prompt change waited a full segment regardless of GPU
speed. A resident library abandons mid-segment. 8.0 s → 1.0 s worst case.

**C4. Four statistical cautions for closed-loop EEG.** Each was found the hard way and
each is a trap the next group will hit:
- windows are not independent (effective n 25 from 1043)
- a continuous coupling index loses power exactly when the intervention works
- event-locked measures are confounded by the trigger in any closed loop
- dichotomised outcomes cost ~3× in detectable effect

**C5. A validated, reproducible instrument.** Ground-truth-tested estimator, 25 automated
checks, every headline number regenerable by one command.

---

## 3. Section plan, with evidence

### §1 Introduction
Closed-loop EEG neurofeedback with generative audio; the gap is that nobody reports the
latency budget. Frame the paper as characterising it.

### §2 System
Muse 2 → LSL → frontal β/α → z against own baseline → `build_prompt` → audio.

| claim | evidence |
|---|---|
| analysis-path latency 5.5 s, decomposed | `benchmarks/latency_probe.py`, any result JSON |
| DSP is not the bottleneck (sub-ms, 190× headroom) | same |
| controller emits a finite prompt set | `scripts/build_library.py::enumerate_prompts` |

**State plainly in §2:** only rungs 1–3 are reachable, and PILOT01 used rung 1 for 96% of
the session. Do not describe the ladder as five graded levels.

### §3 Latency characterisation — the core

| claim | number | source |
|---|---|---|
| nothing reaches realtime | best 1.05× | 3 T4 runs, all precisions |
| ordering fp32 < fp16-half < fp16 | 6 of 6 cells | `verify_claims.py` |
| fp16-half is *slower* on tensor cores | 0.952× at 8 s | T4 runs |
| between-run variance is a machine property | laptop 1.96× vs T4 1.18× | both run sets |

The T4 result is the pivot: it refutes the tensor-core explanation and forces the decode
loop explanation. **Report the refuted hypothesis explicitly** — it is the strongest
evidence that the conclusion was not assumed.

### §4 Architecture
Finite prompt space → complete coverage. Commitment removal → 8×. Equal-power crossfade
between independent renders, with the honest cost: a blend rather than a musical
development.

Include the clipping bound as a worked design constraint: two uncorrelated segments under
equal-power ramps sum to at most √2 × the louder peak, so `output_gain ≤ 1/(0.99·√2)`.
Small, concrete, and shows the architecture was engineered rather than assembled.

### §5 Instrument validation
Ground-truth recovery of known lags (r ≈ 0.85, exact recovery at ±0, 3, 6 s), null
calibration (0/6 false positives), and engine parity. This is what licenses §6.

### §6 Pilot: what a dress rehearsal found
One session, framed as instrument shakedown. Sensing path sound (13.1% rejection, zero
underruns). Then the three defects, each generalisable:
- a crash that recorded itself as success
- a controller thresholding noise five times smaller than the noise
- a fix that traded chatter for repetition

### §7 Statistical cautions
C4 in full, with `fig2`. The strongest transferable content in the paper.

### §8 Limitations
Frontal alpha only; single site, headset, operator; operator not blinded; **no efficacy
data and none planned at this scale**; the ladder narrower than designed; the trend
suffix inert; the feasibility study underpowered for efficacy **by design and stated as
such** rather than discovered afterwards.

### §9 Planned study — a feasibility study, stated as one
**Within-participant crossover, cross-yoked, n = 10**, counterbalanced, ≥48 h washout.
**Explicitly a feasibility study**: at this n, power to detect the literature-matched
0.15 z effect is 38%, so no efficacy test is run and **no p-value is reported for the
primary contrast**. Full specification in `docs/analysis_plan.md`.

The primary output is an interval (±0.20 z at n = 10 under modest heterogeneity) plus
estimates of the **between-participant** and **participant × condition** SDs — the two
quantities that currently force the power analysis to sweep a range instead of naming a
number, and therefore the two that a properly powered trial needs most.

Progression criteria are fixed in advance, including what result would *not* justify a
larger trial.

**Justifying 0.15 z is worth a paragraph**, because the unit conversion is where this
kind of study usually goes wrong. `z` is standardised by *within*-participant baseline
SD; meta-analytic Hedges' g by *between*-participant SD. At a between-participant SD of
0.5 z, the pooled neurofeedback neural-modulation effect (g ≈ 0.26–0.34) maps to 0.15 z,
not the 0.3 z that a naive reading of "both are about 0.3" would suggest.

Two design details are worth a sentence each in the paper, because both are easy to get
wrong and neither is obvious:

- **Cross-yoked, not self-yoked.** A participant's own earlier schedule was generated by
  their own brain dynamics and may still partially track them, breaking the contingency
  only halfway. The bias runs toward the null.
- **The simulated n is a floor.** The power simulation cancels participant offsets
  exactly; real crossover data carries a participant-by-condition interaction it does not
  model — which is precisely one of the quantities this study exists to estimate.

Stating the design in advance, in a preprint that cannot report its outcome, is a
credibility asset rather than an admission.

---

## 4. Figures

| fig | file | caption |
|---|---|---|
| 0 | `fig0_alpha_validation.png` | Sensing-path validation. Frontal alpha rises **2.13×** with eyes closed (d = 1.55, p = 2.8×10⁻²³, 151/76 windows), within the 1.5–3× range expected for frontal channels. Evidence the rig records cortex, not amplifier noise. |
| 1 | `fig1_session_trajectory.png` | Closed-loop session. Arousal index z over 20 min with the target band and rung changes marked. 40% of windows in band; the controller changes rung 10 times, which motivates §7. |
| 2 | `fig2_autocorrelation.png` | Autocorrelation of z. Lag-1 ρ = 0.953, decorrelation 9 s: 1043 windows carry an effective sample size of 25, so treating them as independent overstates evidence 6.4×. |
| 3 | `fig3_switch_intervals.png` | Audio switch intervals before and after hysteresis. 30% of switches arrived faster than the 1.0 s crossfade, blending rather than transitioning. |
| 4 | `fig4_event_locked.png` | Event-locked response. z rises *before* the rung change and decays after, the signature of a trigger rather than a response. Only adaptive minus sham is interpretable. |
| 5 | `fig5_power.png` | Required n against effect size. Crossover needs 8 per arm where independent groups need 60. |

All six are generated by `scripts/make_figures.py` at 300 dpi in one visual style, from
the session logs rather than from saved PNGs, so they regenerate if the data changes.

Figure 0 replaces the plot `alpha_test.py` saves inside its own session directory. That
one predates this figure set and does not match it; the claim is too load-bearing to
present in a different visual language from everything it supports.

---

## 5. Before drafting

| task | why |
|---|---|
| ~~Decide independent vs crossover~~ | **DONE** — crossover, cross-yoked, n = 10 |
| ~~Justify the smallest effect of interest~~ | **DONE** — 0.15 z, anchored to the neurofeedback meta-analytic g with the unit conversion made explicit |
| ~~Decide scope given 38% power~~ | **DONE** — explicit feasibility study; estimation, not testing |
| ~~Freeze `analysis_plan.md` and record its commit hash~~ | **DONE** — tag `preregistration-v1`, commit `e45bd32`, hash checked by the test suite |
| Choose a venue | arXiv `eess.AS` or `q-bio.NC`; a systems venue may want a demo |
| Decide on releasing the library | 109 MB of audio; a Zenodo DOI is the usual route |

## 6. Before submitting

```bash
python scripts/verify_claims.py    # all 13 numbers still match the manuscript
python scripts/run_tests.py        # 24 checks
python scripts/make_figures.py     # regenerate from current data
```

Record the commit hash in the manuscript. A preprint whose numbers cannot be traced to a
specific commit is not reproducible, whatever the repository contains.

Cite the pre-registration as tag **`preregistration-v1`** (commit `e45bd32`, frozen
2026-08-28), and state in §9 that it predates all participant data. The content hash is
in the README and enforced by the test suite, so the claim is checkable by a reader
rather than taken on trust.
