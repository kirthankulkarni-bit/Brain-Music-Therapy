# Closed-loop EEG-driven music on consumer hardware: a latency characterisation and a precomputed-library architecture

**DRAFT — sections 1–8.** §9 (planned study) is specified in
[analysis_plan.md](analysis_plan.md), frozen at tag `preregistration-v1`.

Numbers in this draft are checked by `python scripts/verify_claims.py` (14/14 reproduce).
Citations marked `[CITE: …]` are placeholders where a reference is needed and has not yet
been selected — they are not claims awaiting support, they are known gaps.

---

## Abstract

*(Write last. Should state: latency floor measured across two GPU tiers and three
precision modes; nothing reaches realtime; the bound is the autoregressive decode loop,
not arithmetic; a precomputed library covers the controller's finite prompt space exactly
and cuts worst-case response 8×; and four statistical cautions for closed-loop EEG. State
explicitly that no efficacy data are reported.)*

---

## 1. Introduction

Closed-loop EEG neurofeedback pairs a continuously estimated neural state with a
stimulus that responds to it. Generative audio models make an appealing feedback
channel: instead of a bar or a tone, the participant hears music that changes with their
own brain state, and the change can be graded rather than binary. Several groups have
proposed such systems `[CITE: prior EEG-driven generative music systems]`.

What is missing from those proposals is a latency budget. A closed-loop system is
defined by the delay between a neural event and the stimulus that answers it, and that
delay determines whether the loop is closed in any meaningful sense. The neurofeedback
literature treats it as decisive: comparing 0 s, 1 s and 20 s feedback delays,
real-time feedback improved performance where delayed feedback did not, and delays of
about one second may already be sufficient to disturb the effect (Sato et al., 2022).
The proposed mechanism — sense of agency, and the forward model relating intention to
outcome — is not specific to operant protocols and would apply to any closed loop in
which the participant is meant to perceive the contingency.

Generative audio models sit awkwardly inside that constraint. They are autoregressive
and produce audio a token at a time, so the delay is not an implementation detail to be
optimised away but a property of the model class. Whether such a system can close a loop
at all is an empirical question that, as far as we are aware, has not been reported
`[CITE: confirm no prior latency characterisation exists]`.

This paper reports that measurement for MusicGen on two hardware tiers, gives the
architecture we adopted once the measurement ruled out live generation, and documents
four statistical properties of closed-loop EEG data that we found the hard way and that
we expect to affect any study of this shape.

**We report no efficacy data.** The system has run one closed-loop session, on the
author, unblinded. Every session-level number in §6 describes the behaviour of the
instrument, not of an intervention. The planned feasibility study is pre-registered
(§9) and has not been run.

### Contributions

1. A latency characterisation of MusicGen for closed-loop use across two GPU tiers,
   two backends, three precision modes and nine runs. **Nothing reaches realtime**, and
   the binding constraint is the sequential decode loop rather than arithmetic
   throughput — established by giving the workload roughly 8× the arithmetic and
   observing it get *slower*.
2. An architecture that is complete rather than approximate: the controller's prompt
   space is finite, so a precomputed library can cover it exactly.
3. The observation that the audio-side bottleneck was **commitment**, not generation
   time, and that removing it cuts worst-case response from 8.0 s to 1.0 s.
4. Four statistical cautions, each with a worked example: window autocorrelation,
   the structural failure of continuous coupling indices, the trigger confound in
   event-locked measures, and the cost of dichotomised outcomes.

---

## 2. System

A Muse 2 headband streams four EEG channels at 256 Hz over LSL. Frontal channels
(AF7, AF8) are averaged, band powers computed by Welch's method over a 4 s window at a
1 s hop, and an arousal index formed as log(β/α). The index is smoothed with an
exponential filter (τ = 3 s) and z-scored against a mandatory 120 s eyes-open baseline
recorded at the start of every session, so values are comparable across participants.

A controller maps z onto a text prompt, and an audio engine renders or selects music for
that prompt.

### 2.1 The analysis path is 5.5 s before any audio exists

| term | value | source |
|---|---|---|
| window centroid delay | 2.0 s | half the 4 s window |
| hop quantisation | 0.5 s | half the 1 s hop |
| smoother group delay | 3.0 s | τ = 3 s |
| **total** | **5.5 s** | |

None of this is computation. DSP costs under a millisecond per window, roughly 190×
headroom against the hop, so the analysis latency is structural: it is the price of
estimating band power from a finite window and smoothing the result. Reducing it means
accepting a noisier index, not writing faster code.

This term matters because it bounds what any audio improvement can achieve, and because
it is invisible in systems that report only model inference time.

### 2.2 The controller emits a finite set of prompts

`build_prompt(z, target_z, trend)` is a pure function. It selects one of five energy
rungs and optionally appends one of three trend suffixes, so its range is at most twenty
strings, and it cannot emit anything else. §4 depends on this.

The design applies the iso-principle: the prompt is not set to the target state but to
one rung from the participant's current state in the direction of the target `[CITE:
iso-principle in music therapy]`.

**The reachable set is smaller than the designed set, and we report the measured figure
rather than the intended one.** Because the controller always moves exactly one rung
toward a goal that is itself a rung of the target, only rungs 1–3 are reachable under
the two therapeutic targets used here; the sparsest and most energetic prompts cannot be
selected. In the one session run, rung 1 accounted for 96% of the intervention. The
controller is therefore better described as a three-level ladder that behaves like a
two-level one, not the five-level design it was drawn as.

---

## 3. Latency characterisation

All measurements from `benchmarks/latency_probe.py`. Raw results and per-run JSON are in
the repository.

### 3.1 Nothing reaches realtime

| machine | best configuration | realtime factor |
|---|---|---|
| GTX 1650 Ti (laptop, 4 GB) | fp16-half, 4 s | 2.08× |
| Tesla T4 (Colab, 15.6 GB) | fp32, 4 s | **1.05×** |

Streaming requires < 1.0×. The T4 approaches it at the shortest duration tested, and
the margin shrinks as segments lengthen: 1.05× at 4 s against 1.13× at 8 s. Crossing the
threshold at one duration would not constitute viable streaming, because segment length
is not free — it trades against musical coherence.

### 3.2 The bound is the decode loop, not arithmetic

We initially attributed the absence of an fp16 speedup on the laptop to its lack of
tensor cores (the GTX 16-series reports compute capability 7.5 while omitting them). The
T4 tested that directly, and refuted it.

Tesla T4, three independent runs, medians:

| mode | 4 s | 8 s | vs fp32 |
|---|---|---|---|
| fp32 | **4.21 s** | **9.00 s** | — |
| fp16-half (`.half()`) | 4.51 s | 9.45 s | 0.94× / 0.95× |
| fp16 (autocast) | 5.61 s | 11.47 s | 0.75× / 0.79× |

**On the T4, fp32 is fastest and both fp16 modes lose.** The ordering
fp32 < fp16-half < fp16 held in 6 of 6 run × duration cells.

This is decisive because of what a T4 offers: fp16 on tensor cores at roughly 8× its
fp32 throughput. A matmul-bound workload given 8× the arithmetic should accelerate
dramatically. This one *decelerated*. It was therefore never arithmetic-bound.

What remains is the sequential structure: MusicGen decodes 50 tokens per second of
audio, each conditioned on the last, so per-step launch overhead and the serial
dependency dominate. Reduced precision adds conversion cost and changes kernel
selection without shortening the chain.

The two machines even disagree in direction — `fp16-half` is 6–8% *faster* on the
laptop and 5–6% *slower* on the T4 — which fits: the 1650 Ti is bandwidth-starved
enough that halving bytes moved recovers a little, while the T4 has bandwidth to spare
and pays only the conversion.

**This generalises past MusicGen.** Any batch-1 autoregressive decoder faces the same
structure, so a faster GPU is not the lever. Changing the decode loop is — batching,
speculative decoding, or a non-autoregressive model.

We report the refuted hypothesis explicitly because it is the strongest evidence the
conclusion was measured rather than assumed.

### 3.3 Between-run variance is a property of the machine

| machine | between-run max/min | within-run p95/median |
|---|---|---|
| GTX 1650 Ti | **up to 1.96×** | 1.01–1.11 |
| Tesla T4 | **1.02–1.18×** | 1.01–1.09 |

On the laptop, repeating the same configuration hours apart changed the result by up to
1.96×, an order of magnitude more than the spread within a run. On the T4 the two are
the same order.

The practical consequence: a single benchmark run is close to a measurement in the
cloud and is not one on a thermally limited laptop. Reporting a point estimate from one
laptop run is not defensible, and we report ranges throughout. Adding an untimed warm-up
made the laptop *slower* (12.80 s → 18.05 s at fp32 4 s), because on a thermally limited
part the warm-up mostly adds heat — warm-up practice borrowed from server benchmarking
is counterproductive there.

---

## 4. Architecture: a precomputed library that covers the controller exactly

### 4.1 Finite prompt space

Because `build_prompt` is pure with a finite range (§2.2), a library containing renders
of every reachable prompt covers the controller's entire output space. This is not an
approximation of live generation restricted to a subset; it is exact coverage of what
the controller can ask for. What is given up is novelty *within* a prompt, which is
recovered by rendering K independently seeded variants and selecting among them at
runtime.

The prompt set is derived by sweeping `build_prompt` over a dense grid of its inputs
rather than by enumerating the ladder by hand, so it cannot drift from the function it
mirrors. Coverage is asserted by a test.

### 4.2 The bottleneck was commitment, not generation

The obvious benefit of a library is that selection costs microseconds where generation
cost seconds. That alone would only fix queue starvation.

The larger effect is different. In the streaming implementation, a segment placed in the
playback queue had to play to completion, so a prompt change could not become audible
for up to `queue_depth × segment_seconds` — 8.0 s — **regardless of GPU speed**. A
resident library removes that commitment: playback can abandon the current segment
mid-way and crossfade into a segment of the new prompt from any offset.

| | streaming | library |
|---|---|---|
| worst case, prompt → audible | 8.0 s | **1.0 s** |
| measured median | up to 8.0 s | 13 ms |
| underruns | permanent starvation | 0 |

The worst case is now one crossfade, and `crossfade_seconds` is the latency knob exactly
as `queue_depth` was before.

### 4.3 Crossfade design and a clipping bound

Library segments are independent renders, so at a seam the two signals are uncorrelated.
Equal-power (sin/cos) ramps are used rather than linear, which would sum to an audible
dip mid-transition. Segments are RMS-matched at build time, so the ramp handles shape
only.

Two uncorrelated segments summed under equal-power ramps reach at most √2 × the louder
peak, which bounds the output gain:

> gain ≤ 1 / (peak_max · √2)

With a build-time peak ceiling of 0.99 this gives gain ≤ 0.714. This constraint is not
academic: enlarging the library from 80 to 220 segments raised the maximum segment peak
from 0.482 to 0.990 and pushed the worst case to 1.120 at the then-current gain of 0.8,
while a 90 s verification render still peaked at 0.792 and clipped nothing. The render
did not clip; it *could* have. The bound, not the observation, is what is now asserted.

### 4.4 What the architecture gives up

Streaming conditioned each segment on the tail of the previous one, so the music
developed as a single piece. Library playback blends between independent renders:
coherent within a segment, a crossfade rather than a development across a seam. That is
the cost of the 8× reduction in worst-case response, and it is a musical judgement
rather than a measurable one.

---

## 5. Instrument validation

Two claims in §6 and §7 depend on estimators being correct, so both were validated
against synthetic data with known ground truth before being applied to real sessions.

**Sensing path.** Frontal alpha rose **2.13×** with eyes closed (Cohen's d = 1.55,
p = 2.8 × 10⁻²³, 151 closed / 76 open windows), within the 1.5–3× range expected for
frontal channels, which lack the occipital dominance of classic alpha demonstrations.
This is evidence the montage records cortical activity rather than amplifier noise
(Figure 0).

**Coupling estimator.** The lagged audio-neural coupling index was tested on synthetic
sessions with a lag imposed by construction. It recovered +6, +3, 0 and −3 s exactly
(r ≈ 0.85, p < 0.001), placed positive and negative lags on opposite sides of zero, and
returned no significant result in 0 of 6 independent-series controls.

That validation was not ceremonial. It caught a real defect: the two audio engines log
amplitude envelopes with opposite time semantics — streaming logs audio about to play,
the library logs audio already heard — and treating both as forward-anchored shifted the
audio timeline by one segment. On synthetic data with a true lag of +6.0 s, the defect
returned **−2.0 s**. It did not degrade the estimate; it **inverted its sign**, turning
"audio leads brain" into "brain leads audio" silently and on every library session.

---

## 6. Pilot session

One 20-minute closed-loop session, self-administered and unblinded, run as an instrument
shakedown rather than as data. Outcome values are reported only insofar as they describe
how the instrument behaved.

### 6.1 The sensing path is adequate

| | result |
|---|---|
| baseline rejection | 3.5% |
| intervention rejection | 13.1% |
| buffer underruns | 0 |
| library prompts missing | 0 |

At 13.1% rejection a 20-minute session yields over 1000 valid windows, so signal quality
is not the limiting factor for the analyses in §7.

### 6.2 A crash that recorded itself as success

A `NameError` in the control loop terminated the worker thread on its first intervention
window while the cleanup path still wrote a "session complete" record. The baseline
logged normally, the manifest was written, and the session directory was
indistinguishable from a successful run while containing no intervention data.

With a participant present this is the worst available failure mode, because it is
discovered at analysis when the session can no longer be repeated. The loop now records
an explicit failure with its phase and re-raises; the distinction is asserted by a test
that injects a fault at the same position.

### 6.3 A controller thresholding noise

The session produced 629 audio events in 1200 s, of which 491 were prompt changes rather
than segment exhaustion. The median gap between switches was 1.35 s and **30% of
switches arrived faster than the 1.0 s crossfade could complete**, so the output was a
near-continuous blend of two renders rather than music with transitions in it.

The ladder was not at fault — the energy rung was stable for 95.9% of the session,
exactly as the graded design intends. The trend suffix was: it cycled through all four
variants of that one rung.

The cause was a signal-to-noise failure. `trend` was computed as a single-hop difference
of z, whose standard deviation was 0.275 against a decision threshold of 0.05: **the
threshold was five times smaller than the noise it was thresholding**. The sign flipped
on 29% of hops.

It had not been visible before because the streaming engine sampled the prompt once per
8 s segment, absorbing the chatter at the segment boundary. The library engine responds
in 1 s and therefore followed a signal that was mostly noise. The 8× improvement did not
create this defect; it removed what had been hiding it.

Replacing the one-hop difference with a 20-hop least-squares slope and adding a
dual-threshold hysteresis band reduced prompt changes from 477 to 24 on replay of the
same data, with no switch faster than a crossfade.

**The fix has an honest limit.** Calibrated above the measured noise ceiling, the trend
suffix can no longer fire during ordinary drift: no plausible excursion reaches the
required slope. The controller is effectively the energy ladder alone.

### 6.4 A fix that traded chatter for repetition

Because the suffix became inert, the controller now reaches only the suffix-free
prompts. The library had been built with four renders of each of twenty prompts on the
assumption all twenty were live. Replaying the session through the corrected controller,
a 17-minute session reached **12 of 80 segments** — 32 seconds of unique audio, looping.

For a relaxation intervention that is arguably worse than the chatter it replaced. The
library was rebuilt with renders allocated by measured use (32 per reachable base prompt,
220 total), taking the dominant prompt from 32 s to 256 s of unique audio.

We report this sequence because it illustrates a general hazard: a fix validated against
the metric it targets can degrade a property nobody was measuring.

---

## 7. Four statistical cautions for closed-loop EEG

### 7.1 Windows are not independent

| quantity | value |
|---|---|
| valid intervention windows | 1043 |
| lag-1 autocorrelation | 0.953 |
| decorrelation time (1/e) | 9 s |
| **AR(1) effective sample size** | **25.3** |

A 20-minute session at a 1 s hop *looks like* 1200 observations and *behaves like* 25.
Any analysis treating windows as independent — a t-test across windows, a parametric
correlation p-value, a binomial interval on time-in-band — overstates its evidence by
approximately √(1043/25.3) ≈ **6.4×**, the difference between p = 0.05 and p = 0.4
(Figure 2).

This follows directly from the smoother that produces the index, so it is a property of
the design rather than of this participant. Every inferential quantity in this project
either uses the effective sample size or a permutation null preserving the
autocorrelation.

### 7.2 A continuous coupling index fails when the intervention works

The lagged coupling index returned r = −0.054, p = 0.795 — a clean null on a session
where the estimator is validated to recover known lags at r ≈ 0.85 (§5).

The reason is structural. The index cross-correlates the whole session, so it requires
the audio to vary. A participant who reaches the target and stays keeps the controller
on one rung — 95.9% here — leaving no controller-driven variation to correlate against.

**The measure is weakest precisely when the intervention is most successful.** This is
not fixable by improving the estimator, and it is a poor property for a primary outcome.

### 7.3 Event-locked measures are confounded by their own trigger

Conditioning on rung changes instead recovers signal where the continuous index finds
none: +0.412 z, p = 0.104, from 10 events.

That number is not causal, and the shape of the response shows why. z rises from −0.45
to +1.0 in the ten seconds **before** the change, peaks at onset, and decays to +0.18
after (Figure 4). A rung change happens *because* z moved; z then continues because it
is autocorrelated. A positive effect is exactly what a closed loop with no therapeutic
effect produces — the music follows the brain, the brain continues, and time-locking to
the follow makes it look like a lead. Baseline-correcting does not remove it, because
the pre-window movement is the trigger.

**In any closed loop, the stimulus is a function of the signal, so event-locked
contrasts within a single arm are uninterpretable.** A yoked control that reproduces the
same trigger structure with the contingency broken is required, and the difference
between arms is the only interpretable quantity.

### 7.4 Dichotomised outcomes are expensive

Time-in-band is the clinically interpretable quantity, and powering on it costs roughly
a factor of three in detectable effect relative to mean z. It discards how far inside or
outside the band a participant sits, and it saturates: someone well outside scores near
zero in both arms, contributing nothing to the contrast. Saturation also removes most of
a crossover design's advantage, because participant offsets cancel exactly in a
difference of means and do not cancel through a nonlinear function.

Report it descriptively; power on the continuous measure.

---

## 8. Limitations

**No efficacy data.** One session, on the author, unblinded, single arm. Nothing here
speaks to whether the intervention benefits anyone. The planned study (§9) is a
feasibility study and is likewise not powered to establish efficacy.

**The end-to-end latency is far outside the neurofeedback regime.** With the library
engine the worst case is 6.5 s — 5.5 s analysis plus one crossfade. The literature
finds that delays near 1 s already disturb neurofeedback learning (Sato et al., 2022).
Our budget is several times that, and the 8× audio improvement does not change the
conclusion because **analysis lag is now 85% of the budget**.

We do not think this invalidates the approach, but the argument must be made rather than
assumed. Operant neurofeedback asks the participant to perceive a contingency and learn
self-regulation, which is where the delay constraint is derived. This intervention asks
nothing of the participant, who listens passively while the music tracks their state;
the proposed mechanism is the iso-principle, mood induction by matching and gradually
leading affect, which does not obviously require perceived agency `[CITE: iso-principle
mechanism]`. **Whether a therapeutic effect survives a 6.5 s delay is an open empirical
question, and it is not one this work answers.** It is the most important threat to the
approach and should be tested directly — for example by manipulating delay
experimentally.

**The controller is narrower than designed.** Only rungs 1–3 are reachable, one rung
accounted for 96% of the pilot, and the trend suffix cannot fire after calibration. The
system is a three-level ladder behaving as a two-level one.

**Frontal channels only.** The Muse 2 has no occipital electrodes, and frontal alpha is
weaker and more contaminated by ocular and muscle artefact than occipital alpha. The
index inherits that.

**Single site, single headset, single operator.** The operator cannot be blinded, since
running the sham arm requires supplying the yoked source explicitly.

**Between-participant variance is unmeasured.** With one participant it cannot be
estimated, which is why the sample-size analysis sweeps a range rather than naming a
number. Estimating it is the primary objective of the planned study.

**Benchmark scope.** One model (`musicgen-small`), two GPUs, two backends. The decode-loop
argument should generalise to other autoregressive audio models, but that is an inference
from mechanism rather than a measurement.

---

## References

Sato, Y. et al. (2022). *Real-Time Detection and Feedback of Canonical Electroencephalogram
Microstates: Validating a Neurofeedback System as a Function of Delay.* Frontiers in
Systems Neuroscience. https://doi.org/10.3389/fnsys.2022.786200
*(Verify author list and year before submission.)*

`[CITE: prior EEG-driven generative music systems]`
`[CITE: iso-principle in music therapy]`
`[CITE: confirm no prior latency characterisation of generative audio for closed-loop use]`

Additional sources supporting the pre-registered effect size are listed in §8 of
[analysis_plan.md](analysis_plan.md).
