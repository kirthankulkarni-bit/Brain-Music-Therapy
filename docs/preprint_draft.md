# Closed-loop EEG-driven music systems do not report their latency: a measurement, and what it costs

*Working title. The earlier framing — "a latency characterisation and a precomputed-library
architecture" — described the work but positioned it as a system paper, which the related
work below rules out. The contribution is the measurement and the methodology, not the
system.*

**DRAFT — sections 1–8.** §9 (planned study) is specified in
[analysis_plan.md](analysis_plan.md), frozen at tag `preregistration-v1`.

Numbers in this draft are checked by `python scripts/verify_claims.py` (94/94 reproduce).
**Since 2026-11-22 every EEG result here comes from public data (PhysioNet EEGBCI, HEROIC,
DEAP) or synthetic signals.** Recordings made during development, the author's pilot session
included, are not used for any claim; the self-recorded claims still run with
`verify_claims.py --self-data` but are not cited. The submitted version is
[arxiv_short.md](arxiv_short.md); this draft keeps the longer argument and must agree
with it on every number.

All `[CITE: …]` placeholders were resolved on 2026-09-05/07 and all author lists verified
against the papers; one was wrong (reference 6 was attributed to "Sato et al." and is Asai
et al.). Two items remain outstanding and are marked in the reference list: *Mind to
Music* is paywalled and unread, and Lowe-Brown et al. (2026) has not been read in full.

The paragraph below is retained from the drafting stage. Citations marked `[CITE: …]`
were placeholders where a reference was needed and had not yet
been selected — they are not claims awaiting support, they are known gaps.

---

## Abstract

*(Write last. Should state: latency floor measured across two GPU tiers and three
precision modes; nothing reaches realtime; the bound is the autoregressive decode loop,
not arithmetic; a precomputed library covers the controller's finite prompt space exactly
and cuts worst-case response 8×; on 109 public subjects the deployed estimator loses to
all 9 alternatives on both latency and information rate, and the 0.439 s floor is 1.50× as
informative; the same 9 of 9 holds on Muse 2 recordings (HEROIC, 13 subjects, real
transitions), where the index channels AF7/AF8 show almost no eye-state alpha; and four statistical cautions for closed-loop EEG. State explicitly that no
efficacy data are reported and that all EEG results use public or synthetic data.)*

---

## 1. Introduction

Closed-loop EEG-driven music systems are an established and active line of work. Systems
have paired affect decoding with generative audio backbones (Zhang et al., 2026), built
real-time emotional music generation from EEG (Ran et al., 2024), and used music as a
feedback channel for emotion mediation in a BCI (Ehrlich et al., 2019); a 2023 review
surveys the field (Dash & Agres, 2023). Building such a system is
no longer novel, and this paper does not claim to.

**What these systems do not report is how long their loop takes to close.**

That omission is specific and checkable. MindMelody (Zhang et al., 2026) conditions
MusicGen-medium on a continuously updated EEG-derived affect state and describes itself as
"closed-loop real-time", while reporting no end-to-end latency, no per-generation
inference time, and no hardware specification. It is not unusual in this respect: across
the EEG-driven affective music literature, work is reported in terms of architecture,
decoding accuracy, and audio-quality or subjective-alignment metrics rather than timing.

The omission is not universal in real-time music research. Gesture2Music, a
gesture-driven rather than EEG-driven system, reports 25–30 ms inference and 60–70 ms
full-loop latency, explicitly against a 100 ms threshold for interactive use
(Jeyaraj, Subramanian, Gangadharan & Paul, arXiv:2511.00793). The practice exists; it has
not crossed into the EEG case.

It matters more in the EEG case, not less, for two reasons.

**The analysis path costs seconds before any model runs.** Estimating band power requires
a window, and smoothing a noisy index requires a filter; both are delays, and neither is
computation that a faster GPU addresses. A gesture system reads a sensor; an EEG system
must integrate over time to get a usable estimate at all.

**Generative audio models are autoregressive.** They emit audio a token at a time, so
generation latency is a property of the model class rather than an implementation detail.

And delay is not a cosmetic concern. Training **parietal alpha** — the same rhythm this
system measures — with visual neurofeedback delivered immediately, at +250 ms, or at
+500 ms, sustained post-training change was negatively correlated with latency and
**absent at 500 ms** (Belinskaia, Smetanin, Lebedev & Ossadtchi, 2020). On EEG
microstates the shape is similar: comparing 0, 1 and 20 s delays, the trained microstate
rose significantly over baseline only when feedback was undelayed, although another
microstate class rose under the delayed conditions too, and controllability ratings did
not differ across delays, which the authors read as participants being naive to the delay
(Asai, Hamamoto, Kashihara & Imamizu, 2022). The mechanism offered — feedback arriving
while the oscillation is still ongoing, and the forward model linking intention to
outcome — is not specific to operant protocols, and §8 takes the threat seriously.

So the question this paper asks is not whether such a system can be built. It is: **when
one is measured rather than described, what is the number, and what follows from it?**

We measure it for MusicGen across two GPU tiers, two backends, three precision modes and
nine runs, and report an end-to-end worst case of **6.5 s** — roughly 65× the threshold
Gesture2Music holds itself to, and several times the delay at which neurofeedback
degrades. The dominant term is not the model. We then give the architecture this forces,
and four statistical properties of closed-loop EEG data that we expect to affect any
study of this shape.

**We report no efficacy data.** No participants were enrolled and no EEG was recorded
for this paper. Every EEG number below comes from the PhysioNet EEG Motor
Movement/Imagery Dataset (109 subjects), from DEAP, or from synthetic sessions with known
ground truth, so it describes the instrument and not an intervention. The planned
feasibility study is pre-registered (§9) and has not been run.

### Contributions

1. **A latency budget for closed-loop EEG-driven generative audio**, which this
   literature does not currently report. Measured across two GPU tiers, two backends,
   three precision modes and nine runs. **Nothing reaches realtime**, and the binding
   constraint is the sequential decode loop rather than arithmetic throughput —
   established by giving the workload roughly 8× the arithmetic and observing it get
   *slower*.
2. **The decomposition**, which is the actionable part: after removing the audio-side
   bottleneck, **85% of the remaining budget is the analysis path**. Work aimed at
   faster generation is optimising the smaller term. On labelled eye-closure transitions
   from 109 public subjects, the deployed estimator is beaten on both detection latency
   and information rate by **all 9** alternatives, and the architecture's 0.439 s floor is
   1.50× as informative as the deployed setting.
3. **Four statistical cautions** for closed-loop EEG, each with a worked example:
   window autocorrelation (effective n of 51 from 1200 windows), the structural failure
   of continuous coupling indices, the trigger confound in event-locked measures, and
   the cost of dichotomised outcomes. We have not found these addressed in this
   literature, which reports subjective and audio-quality metrics rather than neural
   time-series statistics.
4. **An architecture that follows from the measurement** rather than preceding it: the
   controller's prompt space is finite, so a precomputed library covers it exactly, and
   removing playback *commitment* — not generation time — cuts worst-case response from
   8.0 s to 1.0 s.

### Relation to prior work

We are not proposing a better EEG-to-music system, and make no comparison of audio
quality or decoding accuracy against MindMelody, Mind to Music, or others. Those systems
are the motivation: they are the reason a latency budget is worth having, and the reason
its absence is worth pointing out. Our system is deliberately simpler than any of them —
a two-channel frontal index and a nine-rung prompt ladder — because a simple controller
makes the latency terms separable and the prompt space finite. A more capable decoder
would add latency to the analysis path, not remove it.

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

`build_prompt` is a pure function. It quantises z onto nine rungs, each 0.5 SD wide and
centred on rung 4, so its range is exactly nine strings and it cannot emit anything else.
§4 depends on this. (Earlier versions used five rungs at 1.0 SD plus one of three trend
suffixes, giving twenty reachable prompts. The suffix was removed because the slope it
gated on is smaller than the noise of the estimator measuring it, and the ladder was
narrowed so that ordinary drift moves the music at all.)

The design applies the **iso-principle** (Altshuler, 1944): the prompt is not set to the
target state but to one rung from the participant's current state in the direction of the
target — match first, then lead. Dropping an aroused listener straight into a sparse
drone is the failure mode the principle exists to avoid. Inside a 0.35 SD deadband around
the target the controller plays the target rung itself, and a minimum dwell (§6) stops a
new prompt change until the previous crossfade has finished.

**A controller can run correctly and still manipulate nothing.** If its rungs are wide
relative to the spread of z, a listener sits on one rung for a whole session, and a yoked
sham is then acoustically identical to the adaptive condition. Studies of adaptive audio
should report prompt changes and distinct prompts per session; ours have to wait for a
session to report.

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

### 3.4 The analysis path dominates, and it is a configuration rather than a floor

With the audio-side bottleneck removed (§4), **85% of the end-to-end budget is the
analysis path**. Work aimed at faster generation therefore optimises the smaller term.

We measured whether that 5.5 s is reducible against labelled ground truth that anyone can
download. The PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al., 2004; Goldberger
et al., 2000), loaded with MNE-Python, gives each of its 109 subjects a one-minute
eyes-open and a one-minute eyes-closed resting run at 160 Hz. It includes AF7 and AF8, the
index channels. It has no TP9/TP10, so T9/T10 stand in. For each subject we spliced
eyes-open 0–30 s, eyes-closed 0–60 s and eyes-open 30–60 s into a 120 s session with one
rising and one falling transition, mean-centring each segment and crossfading the joins
over 50 ms so the splice adds no step. Fifty synthetic sessions at the Muse's 256 Hz,
alternating 60 s blocks whose alpha power steps by 1.34×, repeat everything on the target
sampling rate. Every estimator runs through the deployed feature extractor.

Detection latency is the time from a transition until the estimator crosses the midpoint
between the two state means **and stays across it for 0.5 s**. A crossing undone sooner
counts as noise. Without the hold, the noisiest streaming estimators get credited with
chance crossings at 0.10–0.16 s; for windowed estimators, whose outputs arrive every 0.5
or 1 s, the rule changes nothing. The rule was fixed before the public results were
computed, and first-crossing is reported beside it.

**The budget is confirmed, and the gap to it is explained.** The deployed configuration
detects in a median **4.50 s**. A midpoint crossing needs the smoother to cover half a
step, which takes τ ln 2 = 2.08 s rather than τ, so the expected crossing time is
W/2 + H/2 + τ ln 2 = **4.58 s**. The 5.5 s budget is the full group delay; detection is the
part of it a midpoint rule sees.

Medians across 109 subjects, AF7/AF8:

| estimator | detect | d | rho | ind/min | info/min |
|---|---|---|---|---|---|
| **deployed: 4 s win, 1 s hop, tau=3.0** | **4.50 s** | 2.07 | 0.972 | 0.9 | 1.89 |
| 2 s win, 0.5 s hop, tau=0.5 | 1.50 s | 1.35 | 0.691 | 10.9 | **4.68** |
| streaming, order 4, tau=0.25 | **0.63 s** | 0.41 | 0.118 | 47.3 | 3.11 |

Per-sample discriminability falls as latency falls, as expected. But `d` alone does not
determine how well a session resolves a state difference. The number of *independent*
observations matters equally, and heavy smoothing destroys independence. Scoring on
`d × sqrt(ind/min)`, the t-statistic per root-minute, **the deployed configuration is
dominated by all 9 of the alternatives** (Figure 1). It is not on the efficient frontier.

That holds wherever we looked. It is 9 of 9 under the first-crossing rule, 9 of 9 on
T9/T10 and 9 of 9 on the synthetic sessions. Medians can hide subjects, so we counted
them: the retuned windowed configuration beats the deployed one on both axes in **78.5%**
of subjects individually.

The mechanism connects this section to §7. At tau = 3 s consecutive outputs are nearly
identical (rho = 0.972 over the 120 s session, transitions included), yielding **0.9
independent observations per minute**. That is the
same smoothing that gives a 20-minute session an effective sample size of 51 (§7.1).
**The smoother is simultaneously the largest latency term and the dominant cause of the
autocorrelation that collapses statistical power.** Shortening it pays twice.

Adopting a faster estimator is available, but it is **not** the two-parameter change it
first appears to be. §6 shows that the controller chatters through a less-smoothed index
unless a minimum dwell is added, and the dwell is not free: the library engine (§4) acts on
a prompt change within one crossfade, so a dwell raises worst-case audio latency from one
crossfade to `dwell + crossfade`. On detection latency, the honest comparison is:

| configuration | detection | dwell | crossfade | end-to-end |
|---|---|---|---|---|
| deployed | 4.50 s | — | 1.0 s | **5.50 s** |
| retuned, 1 s dwell | 1.50 s | 1.0 s | 1.0 s | **3.50 s** |

**1.6×, not the 3× available from the estimator in isolation.** A dwell much larger than
one crossfade consumes the gain entirely, and a 4 s dwell would leave the system no faster
than the configuration it replaces. That is the general shape of the result. The
estimator is not the whole loop, and a latency argument that stops at the estimator
overstates what is collectable.

Two caveats remain. The ground truth is an eyes-open/closed splice on a gel cap, not the
Muse (§8). And the retuned controller has not been listened to: zero sub-crossfade
switches establishes that transitions are resolvable, not that the resulting rate of
change is musically acceptable.

### 3.5 The result survives a real Muse and a real transition

Everything above is a gel cap and a splice. HEROIC (Sugden et al., 2024; Zenodo
10.5281/zenodo.11493897, CC BY 4.0) is neither: participants wore a Muse 2 at home or in the
lab, and every session runs about 61 s eyes closed, a spoken cue, and about 61 s eyes open,
recorded continuously at 256 Hz. The transition is real, and the latency we measure includes
the participant's reaction to the cue, whose onset is the transition time.

Two rules were fixed before any estimator touched the data. Four recordings are
byte-identical copies filed under two subject folders (sub005 and sub012), so each is kept
once and the two folders count as one subject. And any recording with a gap over 0.5 s in
the eye blocks is excluded; shorter Bluetooth jitter is interpolated onto a uniform grid.
That leaves **36 of 61 unique recordings, from 13 subjects**. Subjects contribute one to
five sessions, so the figures below are subject-level: each subject's median over their own
sessions, then the median across subjects.

On TP9/TP10, **all 9 alternatives again beat the deployed estimator on both axes**, on
subject medians and on session medians. The deployed estimator takes **8.49 s** to register
eye opening, which is longer than on PhysioNet for the right reason: a real eye opening
adds reaction time and a gradual fade of alpha that a splice does not have. The retuned
windowed setting beats it on both axes in 82.9% of sessions, and the floor's information
rate is **2.68×** the deployed value.

The comparison had to be run on TP9/TP10, and why is the more important finding (§5): on a
Muse, AF7/AF8 barely see the eyes close. Run on those channels, the sweep produced deployed
"detections" at 0.99 s, which is impossible after a real step through a 4 s window and a
3 s smoother. The estimator was crossing a midpoint between two nearly equal levels by
chance. We report that run as uninterpretable rather than as a result.

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

### 4.4 Why not simply use a faster generator?

Non-autoregressive and consistency-model approaches reach faster-than-realtime audio
generation today — AudioLCM reports 333× realtime on a single consumer GPU (Liu et al.,
arXiv:2406.00356), alongside Music Consistency Models at roughly one second per minute of
audio (Fei, Fan & Huang, arXiv:2404.13358), Musika (Pasini & Schlüter, arXiv:2208.08706)
and Music2Latent (Pasini, Lattner & Fazekas, arXiv
2408.06500]`. The decode-loop bound established in §3.2 is a property of MusicGen's model
class, not of generative audio in general.

So the obvious objection is that this architecture solves a solved problem. Two answers,
and only the second is durable.

**It would not help much.** After the audio fix, generation sits inside a 1.0 s crossfade
within a 6.5 s budget of which 85% is analysis (§3.4). Instant generation removes at most
1.0 s.

**Generation is unnecessary, not merely slow.** `build_prompt` is a pure function with a
finite range. Given even an infinitely fast generator, re-synthesising one of nine
possible prompts on demand is strictly worse than selecting a pre-rendered variant: the
same audio, no compute, and no variance in response time. **The finite prompt space, not
the latency, is what justifies the library**, and that argument is unaffected by how fast
generators become.

### 4.5 What the architecture gives up

Streaming conditioned each segment on the tail of the previous one, so the music
developed as a single piece. Library playback blends between independent renders:
coherent within a segment, a crossfade rather than a development across a seam. That is
the cost of the 8× reduction in worst-case response, and it is a musical judgement
rather than a measurable one.

---

## 5. Instrument validation

Claims in §3, §6 and §7 depend on estimators being correct and on the index channels
carrying cortical signal. Each was checked against public data or synthetic ground truth.
None uses a recording made for this project.

**Sensing path.** On the PhysioNet resting runs, through the deployed extractor and the
350 µV rejection threshold, alpha power on AF7/AF8 rose by a median **1.80×** from eyes
open to eyes closed (Figure 3). It rose in 86.8% of the 91 subjects with enough clean
windows (Wilcoxon p = 3.2 × 10⁻¹², d_z = 0.96), with a median 19.8% of windows rejected.
On T9/T10 the rise was smaller, 1.46×, but it appeared in 89.0% of all 109 subjects
(p = 1.2 × 10⁻¹⁷).

A rise in band power is not by itself an alpha rhythm. Broadband power tracking the eye
state would also raise it. So we measured the eyes-closed increase in spectral peak
prominence as well: a median **1.35× on AF7 and 1.41× on AF8**, both above the 1.20
criterion. T9/T10 reach only 1.15 and are not cited as alpha. **On this dataset the
frontal channels show a real alpha peak on eye closure.**

That reverses what the development recordings suggested, where AF7/AF8 failed the same
check, and the reversal should not be over-read. PhysioNet used a gel cap. Dry consumer
electrodes at the same sites are noisier and more prone to ocular artefact, and whether
the Muse reproduces the peak is untested here (§8).

The rejection threshold was moved from 150 µV to 350 µV during development, because a
strict threshold removes eyes-open windows preferentially (blinks happen with the eyes
open) and inflates the contrast. 350 µV is what every number above uses.

**The same check on a Muse fails at the index channels.** On the HEROIC recordings (§3.5),
alpha at AF7/AF8 rose by a median of only **1.06×** on eye closure, in 9 of 13 subjects
(Wilcoxon over subjects p = 0.11), and the alpha peak hardly moved (prominence ratios 1.08 at
AF7, 1.03 at AF8). TP9/TP10 in the same sessions rose **1.82×**, in 10 of 13 subjects
(p = 0.021). This is the pattern the development recordings showed before the paper dropped
them — TP9/TP10 strong, AF7/AF8 weak and without a clear peak — now on 13 people who are not
the author.

One plausible mechanism is the reference. The Muse references to Fpz, a few centimetres from
AF7 and AF8, so alpha common to the reference and the frontal sites cancels in the recorded
difference, while TP9/TP10 sit far enough away to keep it. That is untested here. Whatever
the cause, the arousal index is computed from the two channels where, on its own hardware,
one of the most robust effects in EEG is barely visible. It does not follow that the index
says nothing about arousal, which is a smaller and different signal. It does follow that
nothing in this paper shows those two channels record cortex reliably on a Muse, and that
the obvious next experiment is the DEAP arousal check repeated with the index computed from
TP9/TP10, or after re-referencing.

**That check has been run, and the temporal sites hold up.** DEAP has no TP9/TP10, so the
index was computed from the four sites around them (T7, P7, T8, P8). It correlates with
arousal at rho = +0.454 (p = 0.003), against +0.303 for AF3/AF4. We do not read that as the
temporal sites being better. Four frontal channels averaged (AF3, AF4, Fp1, Fp2) also
improve, to +0.342; a neighbouring temporoparietal set (CP5, CP6, P7, P8) gives +0.295; a
bootstrap interval for the frontal-minus-temporal difference runs from -0.47 to +0.15; and
nine channel sets were tried on 40 trials from one participant. The defensible reading is
narrower: moving the index to TP9/TP10 need not cost it sensitivity to arousal, and on a
Muse those are the channels that demonstrably record cortex.

**Is the temporal index muscle?** That was the worry about moving it. TP9 and TP10 sit over
the temporalis, and muscle activity is broadband, so it raises beta and with it log(beta/
alpha). An index there could correlate with arousal because people clench when aroused.
`scripts/tp_index.py` checks this with the extractor's own 30–45 Hz gamma band as the muscle
proxy, and finds the opposite of the worry:

- Within each eye state, the TP9/TP10 index is essentially unrelated to gamma power on the
  Muse (median Spearman 0.02 across subjects). The AF7/AF8 index is weakly coupled to it
  (0.21). On PhysioNet both are small (0.04 and 0.07).
- Band by band, eye opening on the Muse changes AF7/AF8 alpha by only −5% while beta rises
  12% and gamma 18%. Whatever eye-state contrast the frontal index has is high-frequency
  power, plausibly from the muscles around the eyes. At TP9/TP10 alpha falls 45% and beta
  16%.
- On DEAP, gamma at T7/P7/T8/P8 does not track arousal (rho = −0.095, p = 0.56). Partialling
  it out leaves the temporal index at +0.492 (p = 0.001) and AF3/AF4 at +0.298.

The move costs little elsewhere. Lag-1 autocorrelation of the smoothed index is 0.911 at
TP9/TP10 against 0.905 at AF7/AF8 on the Muse, and 0.906 against 0.918 on PhysioNet T9/T10,
so §7.1's N_eff argument is unchanged. The 350 µV rule rejects more Muse windows at
TP9/TP10 (4.5% against 2.2%), the one measure on which the frontal pair does better. The
deployed system still uses AF7/AF8; the temporal index is what we would change first, and
its arousal evidence is still one DEAP participant.

**Index construct validity.** Two separable claims underlie the measurement: that the
electrodes record cortex, and that log(beta/alpha) measures *arousal*. The second was
inherited from the literature, so we tested it on DEAP, which pairs 32-channel EEG with
self-reported arousal.

Computed with the same `FeatureExtractor` the live system uses, the index correlates with
reported arousal at rho = +0.303 (p = 0.058) on AF3/AF4, against rho = +0.082 (p = 0.615)
for valence as a discriminant control. A single correlation at 40 trials is underpowered,
but the direction across independent montages is not: **7 of 7 montages positive**
(sign test p = 0.016, median rho = +0.162), strongest frontally and weaker posteriorly —
the gradient the construct predicts.

We also asked whether log(beta/alpha) is the *right* index or merely the inherited one,
by comparing seven candidate band ratios on the same data. The deployed index ranks third
of seven by median rho (+0.209 against a best of +0.230), but that gap is far inside the
standard error of a single rho at n = 40 (about 0.16), so the top candidates are
indistinguishable. What the comparison does establish is that the inherited choice sits
in the top group rather than being arbitrary, and that the simpler alternatives are
worse: raw log(beta) achieves only 6/7 positive montages and correlates more strongly
with valence (+0.214) than with arousal, which is the wrong discriminant profile for a
measure named for arousal.

All seven candidates are reported in the repository rather than only the winner, because
selecting among seven on 40 trials is how a spurious result is manufactured. Nothing here
licenses changing the deployed index, which is fixed by the pre-registration.

One participant, self-report, and a 32-channel cap rather than four dry sensors. This is
a construct-validity check on public data that any reader can repeat, not a validation.

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

## 6. Controller stability at low latency

§3.4 makes a faster estimator look nearly free. The controller says otherwise.

We drove the real `PromptGovernor` with z series reconstructed from the PhysioNet
sessions, keeping the 66 that had enough clean eyes-open windows to form a baseline —
132 minutes in all — with target z = −1 and a 1.0 s crossfade. At the deployed settings,
no prompt change arrived before the previous crossfade had finished. Through the
2 s / 0.5 s / tau 0.5 estimator with no dwell, **4176 of 6986 changes did**, or 31.7 per
minute. The output would be a near-continuous blend of two renders rather than music with
transitions in it.

The cause is the ladder, not the estimator. It quantises z with no hysteresis, so a
less-smoothed z crosses rung boundaries constantly. The deployed configuration survives
only because its smoother keeps z away from the boundaries — a property of the smoothing
rather than a designed guarantee, and one that a lower noise level spends.

The fix is a **minimum dwell**, and its value is structural rather than tuned. A dwell of
at least one crossfade is exactly the condition for no switch to start before the
previous crossfade completes, so sub-crossfade switches fall to zero by construction.
Measured with a 1 s dwell: **0**, on the PhysioNet sessions and on the synthetic ones.
The price is the extra second of worst-case latency costed in §3.4.

A general hazard sits under this. The library engine responds within one crossfade where
the streaming engine sampled the prompt once per 8 s segment, and the segment boundary had
been absorbing controller noise. Making the audio path 8× faster did not create the
instability; it removed what had been hiding it. Any latency improvement downstream of a
quantising controller should be checked for this.

---

## 7. Four statistical cautions for closed-loop EEG

### 7.1 Windows are not independent

| quantity | value |
|---|---|
| unspliced PhysioNet resting runs, no rejected windows | 99 |
| lag-1 autocorrelation, deployed index, AF7/AF8 (median) | 0.918 |
| same, DEAP participant 1, AF3/AF4 | 0.918 |
| decorrelation time (1/e) | within 4 s |
| **AR(1) effective sample size, 20 min at 1 s hop** | **51.0 of 1200** |

A 20-minute session at a 1 s hop *looks like* 1200 observations and *behaves like* 51.
Any analysis treating windows as independent — a t-test across windows, a parametric
correlation p-value, a binomial interval on time-in-band — overstates its test statistic
by √(1200/51.0) ≈ **4.85×** (Figure 2).

The time scale is set mostly by the smoother, so this is a property of the design rather
than of any listener. One-minute resting runs also contain none of the slow drift of a
longer session, which makes 0.918 a lower bound on what a closed-loop session would show.
On the Muse recordings, within each clean 61 s eye block, the same index gives rho = 0.905
and a 4.47× overstatement, so the caution does not depend on the cap.
Every inferential quantity in this project either uses the effective sample size or a
permutation null that preserves the autocorrelation.

### 7.2 A continuous coupling index fails when the intervention works

The lagged coupling index cross-correlates the audio envelope with the neural index over
the whole session, so it requires the audio to vary. A listener who reaches the target and
stays there holds the controller on one rung, leaving no controller-driven variation to
correlate against. The estimator is not the problem — it recovers known lags at r ≈ 0.85
(§5).

**The measure is weakest precisely when the intervention is most successful.** This is
not fixable by improving the estimator, and it is a poor property for a primary outcome.

### 7.3 Event-locked measures are confounded by their own trigger

Conditioning on rung changes restores power, and introduces a confound. A rung change
happens *because* z moved, and z keeps moving because it is autocorrelated. Time-locking
to the change therefore produces a positive effect with no therapeutic effect at all —
the music follows the brain, the brain continues, and the follow looks like a lead.

We calibrated this on synthetic sessions with known ground truth. The event-locked
estimator returned +1.503 for a simulated responder, −1.467 for an anti-responder and
+0.011 for an unrelated session. A fourth session had **no audio effect at all**, but its
rung changes were triggered by boundary crossings as in the real controller. It returned
**+0.435**. Its pre-onset slope gives it away — +0.0573 z/s against +0.0012 for a genuine
response — but a within-arm analysis that does not look at the slope would report it as
an effect. Baseline-correcting does not remove it, because the pre-window movement is the
trigger.

**In any closed loop, the stimulus is a function of the signal, so event-locked
contrasts within a single arm are uninterpretable.** A yoked control that reproduces the
same trigger structure with the contingency broken is required, and the difference
between arms is the only interpretable quantity.

### 7.4 Dichotomised outcomes are expensive

Time-in-band is the clinically interpretable quantity. It also discards how far inside or
outside the band a participant sits, and it saturates: someone well outside scores near
zero in both arms and contributes nothing to the contrast. Saturation removes most of a
crossover design's advantage too, because participant offsets cancel exactly in a
difference of means and do not cancel through a nonlinear function.

Simulating 10-participant crossover studies with the PhysioNet autocorrelation and
within-session spread, mean z reached 80% power at an effect of 0.13 SD. Time in band
needed 1.06 SD, **roughly eight times larger**. We do not turn these simulations into a
sample size, because one-minute runs understate the variance of a 20-minute session.

Report time-in-band descriptively; power on the continuous measure.

---

## 8. Limitations

**No efficacy data.** No participants, no EEG recorded for this paper, no adaptive/sham
comparison. Nothing here speaks to whether the intervention benefits anyone. The planned
study (§9) is a feasibility study and is likewise not powered to establish efficacy.

**The end-to-end latency is far outside the neurofeedback regime.** With the library
engine the worst case is 6.5 s — 5.5 s analysis plus one crossfade. The relevant evidence
is not that delay degrades learning gradually, and the most directly comparable study is
about the same rhythm this system measures.

Belinskaia, Smetanin, Lebedev & Ossadtchi (2020) trained **parietal alpha** with visual
neurofeedback delivered immediately, at +250 ms, or at +500 ms. Sustained post-training
change in alpha magnitude was negatively correlated with latency, maximal at the shortest,
and **absent at 500 ms**. Their proposed mechanism is that feedback must arrive while the
alpha spindle is still ongoing. Asai et al. (2022) report a similar shape on EEG
microstates: across 0, 1 and 20 s delays, the trained microstate rose significantly over
baseline only without delay, though not every microstate effect disappeared with delay,
and the authors infer from unchanged controllability ratings that participants were naive
to the delay rather than testing it directly.

Our budget is **13× the delay at which sustained alpha change had already disappeared**,
and the retuned configuration of §3 still sits at 5.5×. The 8× audio improvement does not
change this, because **analysis lag is 85% of the budget**.

**But the architecture has a floor, and it is inside the regime.** The delays in a
streaming estimator are filter group delay and the smoother, and nothing else — no window
centroid, no hop quantisation. At second order with τ = 0.1 s that is **0.189 s** of
analysis path. The end-to-end figure then depends entirely on the crossfade:

| analysis path | crossfade | end-to-end | vs the 500 ms result |
|---|---|---|---|
| deployed, 5.500 s | 1.00 s | 6.500 s | 13.0× |
| retuned, 1.750 s | 1.00 s | 2.750 s | 5.5× |
| streaming floor, 0.189 s | 1.00 s | 1.189 s | 2.4× |
| streaming floor, 0.189 s | 0.25 s | **0.439 s** | **0.9× — inside** |

Two things follow. **The claim that analysis is the dominant term is configuration-
dependent, not architectural**: fix the analysis path and the crossfade becomes the
binding constraint, inverting the decomposition this paper reports. And **reaching the
neurofeedback regime is a measurement away rather than a redesign away** — it requires a
streaming estimator, a crossfade near 0.25 s, and the dwell machinery of §6 to stop a
lightly-smoothed index from chattering through it.

**The floor does not cost information.** On the PhysioNet transitions, a second-order
streaming estimator at τ = 0.1 gives d = 0.39 against the deployed 2.07, a fivefold loss in
per-sample discriminability, while independence rises from 0.9 to 49.7 observations per
minute. Precision goes with the product, and **information per minute at the floor is
1.50× the deployed value**. It is not the *most* informative setting, though. The retuned
windowed configuration is, at 4.68 against the floor's 2.84 — so the choice between them is
a second of latency against about a third of the information rate, and the deployed
setting loses to both.

One caveat the budget hides: measured detection latency does not fall monotonically with
it. At the floor the median is 0.63 s, and below τ ≈ 0.1 it rises again, because the output
becomes noisy enough that crossings no longer hold for 0.5 s. The budget is a floor on
delay, not a promise about detection.

What remains unshown is that any of this survives contact with the real contrast. Every
figure here is measured on an eyes-open/closed manipulation, which is gross compared to
adaptive-versus-sham, and **no session has been run at these settings**. The frontier is
located and priced on a proxy; it is not yet paid for.

We do not think this invalidates the approach, but the argument must be made rather than
assumed. Operant neurofeedback asks the participant to perceive a contingency and learn
self-regulation, and that is where the delay constraint is derived — if the contingency
is the mechanism, breaking it with delay should break the effect, which is what Asai et
al. observed.

This intervention asks nothing of the participant. The proposed mechanism is the
iso-principle, and the evidence for it comes from studies with **no contingency at all**.
Starcke, Mayr & von Georgi (2021) randomised 107 participants to four fixed music
sequences after a sadness induction. Group differences were significant for positive
affect and valence (η² = 0.08 and 0.09) and the iso sequence (sad then happy) scored
highest on both, though not significantly above two happy pieces, and its lower negative
affect was descriptive only (group effect p = 0.77). The sequences were **fixed in
advance**. Starcke & von Georgi (2023) found significantly lower negative affect for iso
than compensatory listening in 59 participants. Whatever carries those effects, it is not
a perceived contingency, because there was none to perceive.

**The largest and only pre-registered test did not replicate this.** Lowe-Brown, Glasser,
Wadley & Koval (2026) randomised 193 participants after an anxiety induction to iso,
compensatory or unguided playlists and found **no significant difference** between iso and
compensatory listening; only an exploratory analysis favoured guided over unguided
listening for calmness. The evidence that the iso principle works at all is therefore
mixed, and weakest in the best-designed study.

**That argument has a sharp edge, and we state it rather than leave it for a reviewer.**
If matching-then-leading works with a fixed playlist and no feedback whatsoever, then
the burden on a closed-loop system is not to beat silence — it is to beat a fixed
playlist. And if, as the pre-registered result suggests, it may not work reliably even
then, an adaptive system cannot borrow its plausibility from the iso literature. The yoked
sham is exactly that comparison: identical music, identical sequence, contingency broken.
So the design already asks the right question, and a null contrast would be informative
rather than merely disappointing.

**Whether a therapeutic effect survives a 6.5 s delay is an open empirical question, and
it is not one this work answers.** It is the most important threat to the approach and
should be tested directly — for example by manipulating delay experimentally, which the
architecture in §4 makes cheap.

**Only part of the data is from the target hardware.** The PhysioNet results come from a
64-channel gel cap sampled at 160 Hz, with spliced, instantaneous transitions, T9/T10
standing in for TP9/TP10, and one-minute runs that limit §7.1 and §7.4. The HEROIC
recordings (§3.5) are a Muse with real transitions and confirm the estimator comparison, but
only at TP9/TP10 and only from 13 subjects in one protocol. On those same recordings the
index channels showed almost no eye-state alpha (§5), so the arousal index is weakly
supported on the hardware it is built for. Moving the index to TP9/TP10 has been checked on
public data (§5) but not against arousal on a Muse, and re-referencing has not been
evaluated.

**Frontal channels only.** The Muse 2 has no occipital electrodes, and frontal alpha is
weaker and more contaminated by ocular and muscle artefact than occipital alpha. The
PhysioNet frontal channels pass the alpha-peak check (§5); on a Muse, in HEROIC, they do
not.

**Single site, single headset, single operator.** The operator cannot be blinded, since
running the sham arm requires supplying the yoked source explicitly.

**Between-participant variance is unmeasured.** It cannot be estimated without
participants, which is why no sample size is derived here. Estimating it is the primary
objective of the planned study.

**Benchmark scope.** One model (`musicgen-small`), two GPUs, two backends. The decode-loop
argument should generalise to other autoregressive audio models, but that is an inference
from mechanism rather than a measurement.

---

## References

**Author lists were verified against the papers on 2026-09-05/06**, except where noted.
One was wrong: reference 6 was attributed to "Sato et al." and is Asai et al. Reference 2
remains unread behind a paywall. URLs accessed 2026-08-28, re-checked 2026-09-06.

1. **MindMelody: A Closed-Loop EEG-Driven System for Personalized Music Intervention.**
   Yimeng Zhang, Yueru Sun, Haoyu Gu & Zhanpeng Jin, arXiv:2605.01235 (May 2026).
   https://arxiv.org/abs/2605.01235
   MusicGen-medium backbone, Transformer-GNN affect encoder, RAG-LLM planner. Describes
   itself as closed-loop real-time; reports no end-to-end latency, inference time, or
   hardware. **The central example for §1.**

2. **Mind to Music: An EEG Signal-Driven Real-Time Emotional Music Generation System.**
   Ran et al., *International Journal of Intelligent Systems* (2024), article 9618884.
   doi:10.1155/int/9618884
   **STILL UNREAD — the last item blocking the §1 claim.** Paywalled; Wiley returns 403
   and no open-access copy exists (checked 2026-09-06, direct fetch and search). Its
   title advertises real-time operation, so it is the most likely remaining
   counterexample to C1. Needs institutional access or an author request. Do not submit
   the latency-reporting claim without it.

3. **A closed-loop, music-based brain-computer interface for emotion mediation.**
   Stefan K. Ehrlich, Kat R. Agres, Cuntai Guan & Gordon Cheng, *PLOS ONE* (2019).
   doi:10.1371/journal.pone.0213516, PMID 30883569.
   **The partial counterexample to C1, and it must be cited as one.** Reports a 4 s
   analysis window at 87.5% overlap giving a 0.5 s update rate, states that zero-phase
   filtering was chosen to avoid delay, and describes tuning a feedback parameter against
   perceived latency — but converts none of it into a signal-to-audio figure. Note that
   Kat R. Agres is also an author of reference 4.

4. **AI-Based Affective Music Generation Systems: A Review of Methods, and Challenges.**
   Adyasha Dash & Kat R. Agres, arXiv:2301.06890 (2023).
   https://arxiv.org/abs/2301.06890
   Field review. **Check directly whether latency is discussed** — the PDF did not
   extract cleanly, so the claim that the field does not report timing rests on the
   individual papers and on search results, not yet on this review.

5. **Gesture2Music: A Low-Latency Real-Time Framework for Continuous Gesture-Driven
   Music Generation.** Rathinaraja Jeyaraj, Barathi Subramanian, Kapilya Gangadharan &
   Anand Paul, arXiv:2511.00793.
   https://arxiv.org/abs/2511.00793
   Reports ~25–30 ms inference, ~60–70 ms full-loop latency against a 100 ms interactive
   threshold. **The contrast case**: the practice exists in adjacent real-time music
   work.

6. **Real-Time Detection and Feedback of Canonical Electroencephalogram Microstates:
   Validating a Neurofeedback System as a Function of Delay.** Tomohisa Asai, Takamasa
   Hamamoto, Shiho Kashihara & Hiroshi Imamizu, *Frontiers in Systems Neuroscience*
   (2022). doi:10.3389/fnsys.2022.786200
   **Previously cited here as "Sato et al." — that attribution was wrong** and was
   inferred from search metadata rather than the paper (corrected 2026-09-06).
   Tests 0 s / 1 s / 20 s feedback delay. The trained microstate (msC) rose significantly
   over baseline only at 0 s; msD rose under the 1 s and 20 s delays as well, so "abolished
   at 1 s" overstates it. Participant naivety to the delay is inferred from unchanged
   controllability ratings, not tested. **Corrected 2026-10-22.** This is the strongest
   single threat to the present approach and §8 states it as such.

7. **Short-delay neurofeedback facilitates training of the parietal alpha rhythm.**
   Anastasiia Belinskaia, Nikolai Smetanin, Mikhail Lebedev & Alexei Ossadtchi,
   *Journal of Neural Engineering* (2020). doi:10.1088/1741-2552/abc8d7, PMID 33166941.
   Immediate versus +250 ms versus +500 ms visual feedback on parietal alpha. Sustained
   change was negatively correlated with latency and absent at 500 ms.
   **The sharpest and most directly relevant version of the delay threat**, because it
   is the same rhythm this system measures rather than a different EEG feature. §8.

8. **Four years' experience with music as a therapeutic agent at Eloise Hospital.**
   I. M. Altshuler, *American Journal of Psychiatry* 100(7):792–794 (1944).
   doi:10.1176/ajp.100.7.792. The source Starcke et al. (2021) cite for the iso-principle:
   match the patient's present state first, then lead it. Cited in §2.2.
   **Corrected 2026-10-22:** this entry previously gave a 1948 paper titled "The past,
   present and future of psychiatric music therapy", which does not exist.

9. **Emotion Modulation through Music after Sadness Induction — The Iso Principle in a
   Controlled Experimental Study.** Katrin Starcke, Johanna Mayr & Richard von Georgi,
   *International Journal of Environmental Research and Public Health* (2021).
   PMC8656869 (open access).
   n = 107, randomised to four fixed sequences after a film-clip sadness induction.
   Significant group effects on positive affect and valence (η² = 0.08 and 0.09), iso
   highest on both but not significantly above happy-happy; the negative-affect group
   effect was not significant (p = 0.77). **Fixed sequences, no feedback and no
   contingency** — which is what makes it the right evidence for §8. The paper does not
   describe the listening as passive. **Corrected 2026-10-22.** Note the effect appeared in female
   participants and not male, a moderator this study is not powered to address.

10. **Music listening according to the iso principle modulates affective state.** Katrin
   Starcke & Richard von Georgi, *Musicae Scientiae* 28(1):131–142 (online 2023, issue
   2024). doi:10.1177/10298649231175029
   n = 59, iso versus compensatory listening after a sadness induction; the iso group
   reported significantly lower negative affect. A "d = 0.52" previously given here could
   not be confirmed from the abstract and has been removed until checked against the full
   text. **Corrected 2026-10-22.**

11. **Personalised affect-regulation playlists: a pre-registered experimental test of the
    iso principle in the general population.** Xanthe Lowe-Brown, Solange Glasser, Greg
    Wadley & Peter Koval, *Musicae Scientiae* (2026). doi:10.1177/10298649261421187
    Musicae Scientiae 30(3):428–446. N = 193, anxiety induction, iso vs compensatory vs
    unguided Spotify playlists built from participants' own songs. **Pre-registered
    analyses found no significant difference between iso and compensatory listening**;
    exploratory analyses favoured guided over unguided listening for calmness. This is the
    pre-registered null §8 anticipated, and §8 now states it. Read from the abstract;
    obtain the full text before submission. **Updated 2026-10-22.**

12. **BCI2000: A general-purpose brain-computer interface (BCI) system.** G. Schalk,
    D. J. McFarland, T. Hinterberger, N. Birbaumer & J. R. Wolpaw, *IEEE Transactions on
    Biomedical Engineering* 51(6):1034–1043 (2004). doi:10.1109/TBME.2004.827072.
    The source of the PhysioNet EEG Motor Movement/Imagery Dataset used in §3.4, §5–§7.
    **Added and verified against Crossref and PubMed 2026-11-22.**

13. **PhysioBank, PhysioToolkit, and PhysioNet.** A. L. Goldberger et al., *Circulation*
    101(23):e215–e220 (2000). doi:10.1161/01.CIR.101.23.e215. The required citation for
    PhysioNet data. **Added and verified against Crossref and PubMed 2026-11-22.**

14. **MEG and EEG data analysis with MNE-Python.** A. Gramfort et al., *Frontiers in
    Neuroscience* 7:267 (2013). doi:10.3389/fnins.2013.00267. Loader for the PhysioNet
    data. **Added and verified against Crossref and PubMed 2026-11-22.**

15. **HEROIC: a platform for remote collection of electroencephalographic data using
    consumer-grade brain wearables.** R. J. Sugden, I. Campbell, V.-L. L.
    Pham-Kim-Nghiem-Phu, R. Higazy, E. Dent, K. Edelstein, A. Leon & P. Diamandis, *BMC
    Bioinformatics* 25:243 (2024). doi:10.1186/s12859-024-05865-9. Data: Zenodo
    10.5281/zenodo.11493897 (CC BY 4.0). The Muse 2 dataset of §3.5 and §5.
    **Added 2026-11-25; metadata checked against Crossref.**

### Outstanding literature work

**Done 2026-09-05; the claim narrowed as a result.** Every reference in §1 was resolved
individually — see the verification record at the end of
[related_work.md](related_work.md). The claim is no longer that this literature does not
report latency. It is:

> No system in this group reports a **measured, decomposed end-to-end latency budget.**
> Ehrlich et al. (2019) report a 4 s analysis window at a 0.5 s update rate and reason
> explicitly about filter delay, without converting either into a signal-to-audio figure;
> the rest report neither.

Ehrlich et al. must be cited in §1 as the partial counterexample rather than left for a
reviewer to find — particularly since their 4 s window at a 0.5 s update sits close to the
configurations §3 measures.

Two further changes fall out of the same check. A 2026 near-neighbour
(Monroy-D'Croz et al., arXiv:2606.01473 — prefrontal EEG over Lab Streaming Layer into
Ableton) was missing from §1 and belongs there; it reports no latency and ran no control
condition, and its emotion-induction manipulation accounted for only 0.40% of total
signal variance, which is independent published support for the frontal-SNR limitation
this paper reports in §8.
And Neurophone runs on a Muse S, the same consumer headband family used here, so it is the
nearest hardware neighbour in the table.

**Still outstanding:** *Mind to Music* (Ran et al.) is paywalled and unread; its title
advertises real-time operation, making it the most likely remaining counterexample. Read
it before submission.

Additional sources supporting the pre-registered effect size are in §8 of
[analysis_plan.md](analysis_plan.md).
