# Related work: where this project sits, and where it can actually contribute

Compiled 2026-08-28 from a targeted search of arXiv and the neurofeedback literature.
Written to answer one question honestly: **after accounting for what already exists, what
is left that is worth publishing?**

Author lists marked `[VERIFY]` came from search metadata rather than the papers and must be
checked before citation.

---

## 1. The crowded part: EEG-driven music systems

Building a closed-loop EEG-to-music system is **not a contribution**. The space is active
and recent.

| system | year | approach | what it reports |
|---|---|---|---|
| **MindMelody** — Zhang, Sun, Gu & Jin, arXiv:2605.01235 | 2026 | Transformer-GNN affect encoder → RAG-LLM planner → hierarchical controller on **MusicGen-medium** | FAD 3.18, Emo-MSE 0.082, Emo-MOS 4.21, valence Δ0.22; within-subject pilot with 4 conditions |
| **Mind to Music** — Ran et al., *Int. J. Intelligent Systems* 9618884, doi:10.1155/int/9618884 | 2024 | EEG-driven real-time emotional music generation, calibration for new users | system design, emotional alignment |
| **Closed-loop music BCI for emotion mediation** — Ehrlich, Agres, Guan & Cheng, *PLOS ONE*, doi:10.1371/journal.pone.0213516 | 2019 | synthesized affective music as a feedback channel; two studies (n=11 listening, n=5 closed-loop) | **4 s window, 87.5% overlap, 0.5 s update rate**; explicit reasoning about filter delay |
| **Neurophone** — Velasquez-Peña et al., *IEEE Int. Symposium on the Internet of Sounds (IS2)* | 2025 | **Muse S** → mobile app → OSC/WiFi → Ableton Live + Max for Live; consumer-grade by design | accessibility and portability of the stack |
| **A Minimalist BCMI for Real-Time Emotion-Driven Sonification** — Monroy-D'Croz, Ramirez-Melendez & Cespedes-Guevara, arXiv:2606.01473 | 2026 | prefrontal EEG → Python DSP → Ableton Live, **synchronised over Lab Streaming Layer** | **negative result**: frontal alpha asymmetry did not separate instructed emotional states (0.40% of variance) |
| **AI-Based Affective Music Generation Systems: A Review of Methods, and Challenges** — Dash & Agres, arXiv:2301.06890 | 2023 | field review | methods and challenges |

### What they do well, and better than this project

- **Affect decoding.** MindMelody reports 76.8% valence / 72.4% arousal cross-subject on
  DEAP. This project uses a single hand-built index, log(β/α) on two channels. Their
  decoder is unambiguously more capable.
- **Model capacity.** MusicGen-medium (1.5B) against musicgen-small (300M).
- **Semantic control.** An LLM planner producing structured intervention plans is
  considerably more expressive than a five-rung prompt ladder.

**Any claim of novelty in system capability is unavailable.** This project's controller is
deliberately the simplest thing that works.

### What none of them report

**A measured end-to-end latency.** MindMelody describes itself as closed-loop and
real-time, and its full text reports no latency figure, no per-generation inference time,
and no hardware specification at all. The minimalist BCMI of Monroy-D'Croz et al. — the
nearest architectural neighbour to this project, prefrontal EEG over Lab Streaming Layer
into Ableton — reports none either. Across the group the pattern is architecture, decoding
accuracy, and audio-quality or subjective metrics, not timing.

That omission is the opening, but it is narrower than it first looked. **See the
verification below: Ehrlich et al. is a partial counterexample and must be cited as one.**

---

## 2. The adjacent fields that *do* report timing

This is what makes the omission a gap rather than a convention.

| work | domain | reports |
|---|---|---|
| **Gesture2Music** `[VERIFY]`, arXiv:2511.00793 | gesture → music | ~25–30 ms inference, **60–70 ms full loop**, explicitly against a 100 ms interaction threshold |
| **Designing Neural Synthesizers for Low-Latency Interaction** `[VERIFY]`, arXiv:2503.11562 | neural synthesis | latency as a first-class design constraint |
| **Sato et al.** `[VERIFY]`, Front. Syst. Neurosci. (2022) | EEG neurofeedback | tests 0 s / 1 s / 20 s feedback delay; real-time improved performance, delayed did not |
| **Smetanin et al.** `[VERIFY]`, *Towards Zero-Latency Neurofeedback*, bioRxiv 424846 | EEG neurofeedback | least-squares FIR envelope estimation; explicit latency/accuracy trade-off |
| **Real-time low-latency estimation of brain rhythms with DNNs** `[VERIFY]`, PMID 37683653 | EEG | TCN achieving >90% envelope correlation at **<10 ms** effective delay |

**Reported neurofeedback operating range: 300–1000 ms.** This project measured **5500 ms**
for its analysis path alone — an order of magnitude outside it, and it did not know that
until the number was measured.

So the practice of reporting and optimising closed-loop latency is established in
neighbouring fields. It has not crossed into EEG-driven music, which is the harder case
because band-power estimation requires integrating over time before any model runs.

---

## 3. The bottleneck has a known solution, and the draft must say so

This is the most important thing the sweep turned up, because it **weakens one of this
project's claims**.

The measurement showed MusicGen cannot reach realtime and that the binding constraint is
the autoregressive decode loop rather than arithmetic. That conclusion stands. But the
draft implies the fix is speculative ("batching, speculative decoding, or a
non-autoregressive model"). It is not speculative — it is published:

| work | claim |
|---|---|
| **AudioLCM** `[VERIFY]`, arXiv:2406.00356 | latent consistency model, **333× faster than realtime** on a single 4090Ti |
| **Music Consistency Models** `[VERIFY]`, arXiv:2404.13358 | consistency models for music generation |
| **Musika!** `[VERIFY]`, arXiv:2208.08706 | fast infinite waveform music generation |
| **Music2Latent** `[VERIFY]`, arXiv:2408.06500 | consistency autoencoders for latent audio |

Non-autoregressive and consistency-model approaches reach faster-than-realtime generation
today. A reviewer will ask why this system does not simply use one.

### The honest answer, which is stronger than the draft's

Two reasons, and only the second is durable:

1. **It would not help.** After the audio-side fix, the analysis path is 85% of the
   budget. Replacing a 1.0 s crossfade with instant generation removes at most 1.0 s from
   6.5 s. **Generation speed is optimising the smaller term** — which is precisely the
   point the latency decomposition establishes.

2. **Generation is unnecessary, not merely slow.** `build_prompt` is a pure function with
   a finite range. Even given an infinitely fast generator, re-synthesising one of twenty
   possible prompts on demand is strictly worse than selecting a pre-rendered variant: same
   audio, no compute, no variance in response time. **The finite prompt space, not the
   latency, is what justifies the library.**

Argument 2 survives arbitrarily fast generators; argument 1 does not survive a
sufficiently fast analysis path. The draft currently leans on 1 and must lean on 2.

**Action:** §4 of the draft must cite the consistency-model literature and make argument 2
explicitly. Not doing so leaves the architecture looking like a workaround for a solved
problem.

---

## 4. What is actually left

After the above, three contributions survive scrutiny.

### C1. A measured latency budget, decomposed — *unclaimed by this literature*

Nine runs, two GPU tiers, two backends, three precision modes. Nothing reaches realtime;
best 1.05×. The binding constraint is the decode loop, established by giving the workload
8× the arithmetic (fp16 on T4 tensor cores) and observing it get **slower**.

The decomposition is the more useful half: **85% of the remaining budget is the analysis
path**, so work aimed at faster generation optimises the smaller term.

Nobody in §1 reports this. Everybody in §2 would consider it mandatory.

**Stated precisely, after the 9/5 verification:** no system in §1 reports a measured,
decomposed end-to-end latency budget. Ehrlich et al. report an analysis window and update
rate and reason about latency qualitatively, which is the closest anyone comes and must be
cited as such. The contribution is the *measurement and decomposition*, not the observation
that latency exists.

### C2. The analysis latency is a dominated configuration — *new, and the strongest result*

See `finding_analysis_latency.md`. Measured against labelled ground truth, the deployed
configuration takes 5.67 s to register a state change and yields **1.2 independent
observations per minute** (ρ = 0.962).

Eight of ten alternatives beat it **on both axes simultaneously**. A two-parameter change
gives 3.4× faster response *and* 2.0× more statistical information per minute.

The mechanism is the finding: the smoothing constant is simultaneously the largest latency
term and the dominant cause of the autocorrelation that collapses effective sample size.
**Shortening it pays twice.** This connects the systems result to the statistics result
through a single parameter, and it is not a trade-off — the current setting is simply off
the efficient frontier.

This appears genuinely unreported. The neurofeedback latency literature (§2) optimises
delay; the statistics literature treats autocorrelation as a nuisance to correct for.
Treating them as the same parameter, with a metric that scores both, is the novel move.

### C3. Four statistical cautions for closed-loop EEG — *unaddressed in §1*

Window autocorrelation (effective n of 25 from 1043); the structural failure of continuous
coupling indices exactly when the intervention succeeds; the trigger confound that makes
event-locked measures uninterpretable within a single arm; and the ~3× cost of dichotomised
outcomes.

The §1 systems report subjective MOS and audio-quality metrics, so none of these arise for
them. They arise for anyone attempting a controlled trial, which is where this field is
heading.

---

## 5. What this means for the paper

**The framing already adopted is correct and this sweep reinforces it**: the contribution
is measurement and methodology, not a system.

Three changes follow.

1. **§4 must cite the consistency-model work** and shift the library's justification from
   "generation is too slow" to "generation is unnecessary given a finite prompt space".
2. **§3 must incorporate C2.** The current text calls 5.5 s structural. It is not, and the
   corrected claim is stronger — a measured dominated configuration is a better result than
   an asserted floor.
3. **The title should foreground the decomposition**, since that is what neither the
   systems papers nor the fast-generation papers provide. Something closer to:
   *"Where the latency actually is in closed-loop EEG-driven music"*.

**Risk checked 2026-09-05. It was a real risk, and it partly materialised.** Every
reference in §1 was resolved individually. Findings, in order of consequence:

**1. Ehrlich et al. (2019) is a partial counterexample, and C1 must be narrowed.** It
reports a 4 s analysis window with 87.5% overlap giving a 0.5 s update rate, states that
zero-phase filtering was chosen to avoid delay, and describes tuning a feedback parameter
against perceived latency. That is not an end-to-end measurement — no figure anywhere
converts those components into a signal-to-audio delay — but it is considerably more than
"reports no timing".

So the claim cannot be *"this literature does not report latency"*. The defensible version:

> No system in this group reports a **measured, decomposed end-to-end latency budget**.
> One (Ehrlich et al.) reports its analysis window and update rate and reasons about
> latency qualitatively; the rest report neither.

That is a weaker claim and still an accurate one, and it is the honest framing given that
Ehrlich's design — a 4 s window at a 0.5 s update — is close to the configuration this
project measures. **Cite it explicitly rather than let a reviewer find it.**

**2. A 2026 near-neighbour was missing entirely.** Monroy-D'Croz, Ramirez-Melendez &
Cespedes-Guevara (arXiv:2606.01473, May 2026) build a minimalist BCMI on prefrontal EEG
over Lab Streaming Layer into Ableton Live — architecturally the closest published system
to this one. The original sweep did not surface it. It strengthens the position rather
than weakening it, on two counts: it reports no latency, and it ran **no sham or control
condition**.

It also reports a **negative result that this project should cite in its own defence**:
their frontal alpha asymmetry did not reliably separate instructed emotional states,
accounting for 0.40% of signal variance, with individual differences explaining more than
the manipulation. That is independent evidence that prefrontal affective indices are hard
— directly relevant to `finding_channel_validation.md`, where this project's own AF7/AF8
validation failed. The right framing is not that this project has a problem others have
solved; it is that the problem is real, published, and this project is one of the few
measuring it rather than assuming it away.

**3. Neurophone is the nearest hardware neighbour and had no citation at all.** It was
listed with no author, no year and no venue. Resolved: Velasquez-Peña et al., IEEE
International Symposium on the Internet of Sounds, 2025 — and it runs on a **Muse S**,
the same consumer headband family as this project, streaming over OSC into Ableton Live.
A reviewer in this area will know it.

**4. Two citations were wrong in detail.** MindMelody has a fourth author (Zhanpeng Jin)
who was omitted. The Dash & Agres title was truncated. Both corrected in the table.

**Still unverified.** *Mind to Music* (Ran et al.) is paywalled and its full text could not
be checked for timing figures; its title advertises real-time operation, so it is the most
likely remaining counterexample. **Do not submit C1 without reading it.** Neurophone's full
text is likewise unread — an OSC-into-Ableton pipeline is exactly the kind of system whose
authors may quote a latency.

C2 and C3 do not depend on any of this, which is a reason to lead with them.
