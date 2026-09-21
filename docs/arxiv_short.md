# Where the latency actually is in closed-loop EEG-driven music

**A measured architecture, and four statistical properties of closed-loop EEG data**

Kirthan Kulkarni

---

## Abstract

Closed-loop systems that drive music generation from EEG are an active area, and a working
one: several have been built and described. What is not reported is how long they take.
Across the systems we could resolve individually, none publishes a measured, decomposed
end-to-end latency budget; one reports its analysis window and update rate and reasons about
delay qualitatively.

We build such a system on consumer hardware — a four-channel Muse headband, a frontal
log(β/α) index z-scored against the participant's own baseline, a nine-rung prompt ladder,
and a resident library of pre-rendered audio — and measure it. Generation with MusicGen runs
**1.05–6.27× slower than realtime** across every machine, backend and precision we tested,
including a datacenter T4, so a precomputed library is not a compromise but the correct
architecture for a finite prompt space. End-to-end worst case is **6.5 s**, of which **84.6%
is the analysis path**, not the model.

That decomposition turns out to be configuration-dependent rather than architectural.
Against labelled eyes-open/closed ground truth, the deployed estimator is **dominated by 8
of its 9 alternatives** on both detection latency and information per minute simultaneously.
The architecture's floor is **0.189 s** of analysis path, and at that floor information rate
is **1.83× the deployed value** — the fastest configuration is also the more informative one.
With a 0.25 s crossfade the end-to-end figure is **0.439 s**, inside the regime where
published alpha-neurofeedback studies still find an effect and where our deployed
configuration is 13× too slow.

We also report four statistical properties of closed-loop EEG data that affect any study of
this kind, three of which cost us designs before we measured them: autocorrelation inflates
apparent evidence **6.4×**; continuous audio-neural coupling loses power precisely when the
intervention succeeds; event-locked responses are confounded by their own trigger to a degree
we calibrate against synthetic ground truth; and dichotomised outcomes cost roughly 3× in
detectable effect.

**This is a systems and methods paper. It makes no claim that the intervention helps anyone.**
Two self-administered unblinded sessions exist; zero adaptive/sham pairs; zero participants.

---

## 1. What is missing from a crowded field

Building an EEG-to-music loop is not a contribution. MindMelody (Zhang, Sun, Gu & Jin, 2026)
pairs a transformer-GNN affect encoder with a retrieval-augmented planner over
MusicGen-medium; Ran et al. (2024) drive real-time emotional music generation from EEG;
Ehrlich, Agres, Guan & Cheng (2019) close the loop with synthesised affective music;
Velasquez-Peña et al. (2025) do it on a consumer Muse S into Ableton Live; Monroy-D'Croz,
Ramirez-Melendez & Cespedes-Guevara (2026) build a minimalist BCMI on prefrontal EEG over
Lab Streaming Layer. Dash & Agres (2023) survey the field.

These systems decode affect better than ours. We use a single hand-built index on two
electrodes; MindMelody reports 76.8% valence and 72.4% arousal cross-subject on DEAP. Any
claim of novelty in system capability is unavailable to us, and we make none.

**What they do not report is timing.** MindMelody describes itself as closed-loop and
real-time and gives no latency figure, no inference time and no hardware specification.
Monroy-D'Croz et al. — architecturally the closest published system to ours — reports none
either, and ran no control condition.

One paper is a partial counterexample and we state it rather than leave it for a reviewer.
Ehrlich et al. report a 4 s analysis window at 87.5% overlap giving a 0.5 s update rate, say
that zero-phase filtering was chosen to avoid delay, and describe tuning a feedback parameter
against perceived latency. They convert none of this into a signal-to-audio figure. So the
defensible claim is not that this literature ignores latency; it is:

> No system in this group reports a **measured, decomposed end-to-end latency budget.**

The adjacent fields treat it as mandatory. Gesture2Music (Jeyaraj et al.) reports 30 ms
inference against a stated 100 ms interaction threshold. Neurofeedback work treats delay as
the independent variable: Belinskaia, Smetanin, Lebedev & Ossadtchi (2020) trained parietal
alpha with feedback at 0 ms, +250 ms and +500 ms and found sustained post-training change
negatively correlated with latency and **absent at 500 ms**; Asai, Hamamoto, Kashihara &
Imamizu (2022) found an effect on EEG microstates **only** at zero delay, with one second
enough to abolish it and participants unable to detect the delay that did so.

Our question is therefore not whether such a system can be built. It is: **when one is
measured rather than described, what is the number, and what follows from it?**

---

## 2. The system

**Sensing.** A Muse 2 streams four channels at 256 Hz over LSL. Frontal AF7/AF8 are averaged;
each window is detrended, band-passed 1–45 Hz, notched at 60 Hz, and Welch-transformed. The
index is log(β/α), smoothed with a one-pole filter (τ = 3 s) and z-scored against the
participant's own 120 s eyes-open baseline. Windows exceeding a 350 µV peak-to-peak threshold
are rejected and do not update the smoother.

**Control.** A nine-rung energy ladder spans arousal at 0.5 SD per rung. The controller
applies the iso-principle (Altshuler, 1948): it does not jump to the target state but moves
one rung from the participant's current rung toward it. Output is therefore a pure function
of (z, target), with a finite range of exactly **9 strings**.

**Audio.** Each rung has 32 pre-rendered 8 s segments (288 total, 38.4 minutes). A prompt
change abandons the current segment mid-playback and equal-power crossfades into one drawn
from the new rung.

Two properties of this design carry the paper.

**The prompt space is finite and small, so generation is unnecessary rather than merely slow.**
A library covering nine strings covers the controller's entire output space exactly. Even
given an infinitely fast generator, re-synthesising one of nine prompts is strictly worse
than selecting a pre-rendered variant. This argument survives arbitrarily fast models; the
latency argument below does not.

**The library responds in one crossfade, not one segment.** Streaming commits: once a segment
is queued it plays to completion, so a prompt change waits up to queue_depth × segment_seconds
regardless of GPU speed. Worst-case audio latency drops from 8.0 s to 1.0 s — 8×, and it is
the architecture, not the hardware, that buys it.

---

## 3. The latency budget, measured

### 3.1 Generation is slower than realtime everywhere we looked

Nine benchmark runs across two GPU tiers, two backends and three precision modes. Every one
of 44 configuration cells lands between **1.05× and 6.27× slower than realtime**. The best
case is a datacenter T4.

The bound is the sequential decode loop, not arithmetic, established by giving the workload
8× the arithmetic and watching it get slower: on the T4, fp32 beats fp16-half by a factor of
**0.952**, and fp32 < fp16-half < fp16 held in **6 of 6** cells. Autocast inserts a cast at
every operation, which amortises over large batched matmuls in training and is pure overhead
for batch-1 autoregressive decoding.

Between-run variance is a property of the machine, not the measurement: up to **1.96×** on a
thermally limited laptop and **1.18×** on the T4 for the same configuration. Report ranges,
not point estimates — a single run of this probe is not a measurement on a laptop GPU.

### 3.2 The dominant term is the analysis path

End-to-end worst case with the library engine is **6.5 s**: 5.5 s of analysis plus one 1.0 s
crossfade. **84.6% is analysis.** Work aimed at faster generation optimises the smaller term.

The analysis budget is structural, not implementation slop: a 4 s window contributes 2.0 s of
centroid delay, a 1 s hop contributes 0.5 s of quantisation, and a τ = 3 s smoother
contributes 3.0 s of group delay. Predicted 5.5 s; measured **5.665 s** against labelled
ground truth.

### 3.3 The deployed configuration is dominated

Latency alone would favour a wire and discriminability alone would favour smoothing
everything to a constant, so we score estimators on **d × √(independent observations per
minute)** — the t-statistic per root-minute, which is what determines how precisely a
fixed-duration session resolves a state difference.

| estimator | detect | d | ind/min | info/min |
|---|---|---|---|---|
| **deployed** (4 s window, 1 s hop, τ = 3) | 5.67 s | 1.99 | 1.2 | **2.14** |
| 2 s window, 0.5 s hop, τ = 0.5 | 1.68 s | 1.14 | 13.6 | **4.20** |
| streaming, order 4, τ = 0.25 | 0.17 s | 0.70 | 30.9 | 3.91 |

**The deployed setting is dominated by 8 of its 9 alternatives on both axes at once.** It is
not a latency/accuracy trade-off; it is off the efficient frontier.

The mechanism links this section to §4. At τ = 3 s consecutive outputs are nearly identical
(ρ = 0.962), yielding 1.2 independent observations per minute — which is precisely the
coefficient that produces an effective sample size of 25 from a 20-minute session. **The
smoother is simultaneously the largest latency term and the dominant cause of the
autocorrelation that collapses statistical power.** Shortening it pays twice.

### 3.4 The floor, and what binds it

A streaming estimator has no window centroid and no hop quantisation; its only delays are
filter group delay and the smoother. At second order with τ = 0.1 s that is **0.189 s**.

| analysis path | crossfade | end-to-end | vs the 500 ms result |
|---|---|---|---|
| deployed, 5.500 s | 1.00 s | 6.500 s | 13.0× |
| retuned, 1.750 s | 1.00 s | 2.750 s | 5.5× |
| floor, 0.189 s | 1.00 s | 1.189 s | 2.4× |
| floor, 0.189 s | 0.25 s | **0.439 s** | **0.9× — inside** |

Two consequences. **"Analysis is 85% of the budget" is configuration-dependent, not
architectural**: fix the analysis path and the crossfade becomes the binding term, inverting
the decomposition we report in §3.2. And **the floor is not a trade-off**: per-sample
discriminability collapses there (d = 1.99 → 0.61) but independence rises from 1.2 to 41.4
observations per minute, and information per minute at the floor is **1.83×** the deployed
value. The fastest configuration this architecture can reach is also the more informative one.

Three caveats. Measured *detection* does not fall monotonically with the budget — below
τ ≈ 0.25 the estimate is noisy enough that midpoint crossing becomes harder to time, so the
budget is a floor on delay rather than a promise about detection. Every figure here is
measured on an eyes-open/closed manipulation, which is gross compared to adaptive-versus-sham.
And no session has been run at these settings: the frontier is located and priced on a proxy,
not paid for.

### 3.5 A finer estimator needs controller work, not a flag

Adopting the faster estimator is not the two-parameter change it appears to be. Replayed
against a pilot recording, `2 s / 0.5 s / τ = 0.5` produces **591 switches inside a
crossfade** — a continuous blend of two independent renders rather than music with
transitions in it, and a larger fraction than the defect that made our first pilot's audio
unusable. The cause is not the estimator: it is that the ladder quantises z with no
hysteresis, so a less-smoothed index crosses rung boundaries constantly.

The fix is a **minimum dwell**, and its value is structural rather than tuned: a dwell of at
least one crossfade is exactly the condition for no switch arriving before the previous
crossfade completes, so sub-crossfade switches fall to **0** by construction. The dwell is
not free — the library engine acts within one crossfade, so worst case becomes dwell +
crossfade — which is why the honest figure for the retuned configuration is 1.8× rather than
the 2.4× the estimator alone suggests.

---

## 4. Four statistical properties of closed-loop EEG data

### 4.1 Autocorrelation inflates evidence by a factor of six

A 20-minute session yields 1043 valid windows at a 1 s hop, and it is tempting to treat them
as 1043 observations. Lag-1 autocorrelation is **0.953**, giving an AR(1) effective sample
size of **25.3**. Any test treating windows as independent overstates its evidence by
**√(1043/25.3) = 6.4×** — the difference between p = 0.05 and p = 0.4.

### 4.2 Continuous coupling loses power exactly when the intervention works

Cross-correlating the audio envelope against the neural index over a whole session has a
perverse property: a participant who reaches target and stays there keeps the controller on
one rung, so the audio stops varying in the way the controller drives it and there is almost
nothing left to correlate. Our first pilot returned r = −0.054, p = 0.795 while sitting on
one rung for **95.9%** of the session. The estimator was not failing — it recovers a known
+6 s lag from synthetic data. There was simply no controller-driven variation in the window
it was given. **This is a bad property for a primary outcome and it is not fixable by
improving the estimator.**

### 4.3 Event-locked responses are confounded by their own trigger

Conditioning on rung changes restores power, because it looks only at moments that carry
information. But a rung change happens *because* z moved, and z keeps moving — so a positive
effect is exactly what a closed loop with no therapeutic effect whatsoever would produce.

We calibrate this rather than warn about it. Against synthetic sessions with known ground
truth, the estimator recovers +1.503 on a responder, −1.467 on an anti-responder, and +0.011
(p = 0.573) on an unrelated session. A fourth session is constructed with **no audio effect
at all** — z wanders and rung changes fire on boundary crossings, as the real controller does
— and it scores **+0.435 at p = 0.010**.

Our pilot measures **+0.412**. The two are indistinguishable. The diagnostics separate them
from a real response and the pilot sits on the wrong side: its pre-window slope is +0.1292
z/s against +0.0573 for the pure confound and +0.0012 for a genuine response, and it is the
only one of the three whose curve peaks at onset and decays — the shape a trailing excursion
makes, and the opposite of a real response, which rises *after* onset. **A positive
within-arm event-locked effect is not evidence of an effect.** Only the yoked contrast is
interpretable.

### 4.4 Dichotomising costs about three times the effect

Time-in-band is the interpretable clinical quantity and a costly primary outcome: it discards
how far inside or outside the band a participant is and saturates, since someone well outside
the band scores near zero in both arms. Simulated against the measured autocorrelation, it
costs roughly 3× in detectable effect. We register mean z as primary and report time-in-band
descriptively.

**Consequence for design.** At n = 10 with a crossover design, power to detect the
literature-matched 0.15 z effect is **38.8%**; reaching 80% needs **25 participants per arm**.
A study reporting p > 0.05 at 38% power has learned nothing, so we register a feasibility
study reporting an interval and two variance components, and no p-value for the primary
contrast.

---

## 5. Does the index measure anything?

Two separable claims: that the electrodes record cortex, and that log(β/α) tracks arousal.

**Cortex.** On an eyes-open/eyes-closed manipulation, alpha rose **2.13×** with eye closure
on the temporal pair (d = 1.55, p = 2.8 × 10⁻²³). That figure is threshold-dependent and we
report the sensitivity rather than the single number: the recording used a 150 µV rejection
threshold, which removes eyes-open windows preferentially (blinks occur with the eyes open)
and leaves a 1.97 imbalance; at the pipeline's deployed 350 µV threshold the conditions are
near-balanced and the ratio is **1.90×**. The effect is significant at **every** threshold
tested including no rejection at all, so it is not manufactured by the rejection rule — only
its magnitude is.

**The channels the study actually uses are weaker, and we report both recordings rather than
substituting one.** The validation above is on TP9/TP10; the index runs on AF7/AF8. A second
recording on AF7/AF8 with good contact (6.8% rejection) shows a significant alpha rise —
**1.34×** geometric, d = 0.61, p = 5 × 10⁻⁷ — but fails a stricter diagnostic: the
eyes-closed rise in *spectral peak prominence*, which the temporal channels pass clearly
(2.46×, 1.91×), is **1.09× on AF7 and 1.13× on AF8**, below a 1.2× threshold. Alpha power
rises on the forehead; a distinct alpha peak barely does. We report frontal signal quality
as a measured limitation. This is consistent with Monroy-D'Croz et al., whose frontal alpha
asymmetry explained 0.40% of signal variance.

**Arousal.** Tested on DEAP, which pairs 32-channel EEG with self-reported affect. Computed
with the same feature extractor the live system uses, the index correlates with reported
arousal at **ρ = +0.303** (p = 0.058) on AF3/AF4, against +0.082 with valence — the right
discriminant profile, at a sample size where no single montage reaches significance. The
direction holds across **7 of 7** independent montages (sign test p = 0.016), which is the
result; no single correlation is.

---

## 6. Reproducibility as an engineering problem

A preprint's numbers are copied from a terminal into prose once, and then the code moves.
Every headline number in this paper is regenerated from artefacts on disk by a single
command, which checks it against the value asserted here: **44 claims**, all reproducing. The
practice caught real drift, including one instance where our own motivating example (a
manuscript saying 1.14× while the repository said 1.05×) was live in the codebase.

Three failure modes were worth the engineering, and we report them because they generalise to
any pre-registered study with a live codebase:

**Recordings must be pinned by name, not selected as "the newest".** A second pilot silently
re-based a dozen claims, five of seven figures and eight tests onto itself — including a
claim that documented a *historical* defect, which would have been overwritten with the
post-fix number and the finding quietly deleted.

**Synthetic data must be unable to reach the manuscript.** A thirty-second demo run wrote a
session that recomputed six claims from a signal generator. The marker distinguishing
synthetic from real already existed; nothing read it.

**The freeze mechanism must not force its own violation.** Our pre-registration is
hash-checked on every test run, but its deviations section is *inside* the hashed file, so
logging a deviation fails the check. The obvious repair — updating the stored hash — destroys
the guarantee permanently. A guard that makes disabling itself the reasonable next step is
worse than no guard; the log lives outside the hash.

Two data-acquisition defects are also worth reporting, because both were invisible to every
check we had. The streaming layer delivered **each packet two to four times** with timestamps
stepping backwards between copies, which downstream code — counting samples rather than time
— read as the signal jumping backwards every 12 samples and reported as electrode failure on
a headset that was fine. And raw timestamps were stored at a precision that depended on
**how long the laptop had been running**: 8 ms after a fresh boot, 0.25 s after 27 days up.

---

## 7. What this work does not establish

**No efficacy claim of any kind is made or supported.** The evidence base is two
self-administered unblinded sessions, zero adaptive/sham pairs and zero participants.

**The measured latency is far outside the regime where the delay literature finds effects.**
Our deployed budget is 13× the delay at which sustained alpha change had already disappeared
in Belinskaia et al., and the retuned configuration is still 5.5×. We do not think this
invalidates the approach, but the argument must be made rather than assumed: operant
neurofeedback asks the participant to perceive a contingency, which is where the delay
constraint is derived, whereas the proposed mechanism here is the iso-principle, whose
evidence comes from **fixed sequences with no contingency at all** — Starcke, Mayr & von
Georgi (2021) randomised 107 participants to four pre-set sequences after a sadness induction
and found the iso sequence produced higher positive affect and valence (η² = 0.08, 0.09) with
purely passive listening. That argument has a sharp edge we state rather than hide: if
matching-then-leading works with a fixed playlist, the burden on a closed-loop system is not
to beat silence but **to beat a fixed playlist** — which is exactly what a yoked sham tests.

**A controller can be well-behaved and still produce no manipulation.** Our second pilot ran
for twenty minutes with zero prompt changes: at 1.0 SD per rung, a participant who sits at or
below target maps to a single rung for the whole session. A yoked sham of such a session is
acoustically identical to the adaptive arm, so the contrast has nothing to test. We widened
the ladder to nine rungs at 0.5 SD, which restores variation (33 changes across 5 prompts on
one pilot, 26 across 3 on the other) — but this was discovered only by running the system on
a person and counting what it played. **We suggest reporting prompt-change counts and
distinct-prompt counts as standard for any adaptive audio intervention**; they are cheap, and
their absence hides a degenerate manipulation behind a functioning system.

---

## 8. Conclusion

We measured what a closed-loop EEG music system actually costs in time, and found the
dominant term in the analysis path rather than the model — then found that this decomposition
is a property of the configuration, not the architecture. The deployed estimator is dominated
by 8 of 9 alternatives on both latency and information rate; the architecture's floor is
0.189 s of analysis and 0.439 s end-to-end, inside the regime where alpha neurofeedback still
works, and more informative than the configuration we shipped.

Alongside, four statistical properties of closed-loop EEG data that shaped every design
decision we made: autocorrelation inflating evidence 6.4×, continuous coupling failing when
the intervention succeeds, event-locked effects indistinguishable from their own trigger
confound, and dichotomisation costing 3× in detectable effect.

None of it says the intervention works. All of it is what we would have wanted to read before
building the system.

---

## References

1. Yimeng Zhang, Yueru Sun, Haoyu Gu & Zhanpeng Jin. *MindMelody: A Closed-Loop EEG-Driven
   System for Personalized Music Intervention.* arXiv:2605.01235 (2026).
2. Ran et al. *Mind to Music: An EEG Signal-Driven Real-Time Emotional Music Generation
   System.* International Journal of Intelligent Systems 9618884 (2024).
   doi:10.1155/int/9618884
3. Stefan K. Ehrlich, Kat R. Agres, Cuntai Guan & Gordon Cheng. *A closed-loop, music-based
   brain-computer interface for emotion mediation.* PLOS ONE (2019).
   doi:10.1371/journal.pone.0213516
4. Adyasha Dash & Kat R. Agres. *AI-Based Affective Music Generation Systems: A Review of
   Methods, and Challenges.* arXiv:2301.06890 (2023).
5. Velasquez-Peña et al. *Neurophone: A Brain-Computer Music Interface for Emotional
   Neurofeedback.* IEEE Int. Symposium on the Internet of Sounds (2025).
6. Pablo A. Monroy-D'Croz, Rafael Ramirez-Melendez & Julian Cespedes-Guevara. *A Minimalist
   Brain-Computer Musical Interface for Real-Time Emotion-Driven Sonification.*
   arXiv:2606.01473 (2026).
7. Anastasiia Belinskaia, Nikolai Smetanin, Mikhail Lebedev & Alexei Ossadtchi. *Short-delay
   neurofeedback facilitates training of the parietal alpha rhythm.* Journal of Neural
   Engineering (2020). doi:10.1088/1741-2552/abc8d7
8. Tomohisa Asai, Takamasa Hamamoto, Shiho Kashihara & Hiroshi Imamizu. *Real-Time Detection
   and Feedback of Canonical Electroencephalogram Microstates.* Frontiers in Systems
   Neuroscience (2022). doi:10.3389/fnsys.2022.786200
9. Rathinaraja Jeyaraj, Barathi Subramanian, Kapilya Gangadharan & Anand Paul.
   *Gesture2Music: A Low-Latency Real-Time Framework for Continuous Gesture-Driven Music
   Generation.* arXiv:2511.00793.
10. I. M. Altshuler. *The past, present and future of psychiatric music therapy.* American
    Journal of Psychiatry (1948).
11. Katrin Starcke, Johanna Mayr & Richard von Georgi. *Emotion Modulation through Music
    after Sadness Induction — The Iso Principle in a Controlled Experimental Study.*
    International Journal of Environmental Research and Public Health (2021). PMC8656869.
12. Huadai Liu et al. *AudioLCM: Text-to-Audio Generation with Latent Consistency Models.*
    arXiv:2406.00356 (2024).

---

*Reproducibility: 44 of the numbers in this paper are regenerated from artefacts on disk by
`python scripts/verify_claims.py`, which checks each against the value asserted here and
fails if any has moved. The synthetic-ground-truth figures in §4.3 come from
`validate_event_locked.py`, the coupling recovery in §4.2 from `validate_coupling.py`, and
the threshold sweep in §5 from `alpha_sensitivity.py`. Effect sizes quoted from the cited
literature are not ours to regenerate.*
