# The sensing path was validated on channels the study does not use

**Found 2026-08-28 while building the estimator sweep. This is the most serious
substantive problem found in the project so far, and it is not a code defect.**

## What happened

The eyes-closed alpha validation — Figure 0, and the evidence that the rig records
cortex rather than amplifier noise — was recorded with
`frontal_channels: ["TP9", "TP10"]`. That is in the session manifest and was not
noticed until the raw recording was reanalysed.

The live system computes its arousal index on **AF7/AF8**. `live_music.py` defaults to
`--channels AF7,AF8`, and PILOT01 ran that way.

**The validation and the instrument use different montages.**

## The effect does not transfer

Recomputed from the raw recording of the same session, both pairs, identical pipeline:

| pair | alpha ratio, closed/open | Cohen's d | p | windows rejected |
|---|---|---|---|---|
| TP9/TP10 (validated) | **1.85×** | 1.23 | 1.8 × 10⁻²⁵ | 3% |
| **AF7/AF8 (the study's index)** | **0.91×** | −0.17 | **0.26** | **48%** |

On the channels the study actually uses, in the one session that could test it, the
canonical eyes-closed alpha response **was not detectable**, and nearly half the windows
were rejected as artefact.

Reproduce with `scripts/estimator_sweep.py --report-channels`.

## What this does and does not mean

**It is not evidence that AF7/AF8 cannot work.** The 48% rejection rate is the key
context: frontal electrodes sit on skin that moves, and that session evidently had poor
frontal contact. PILOT01, which ran on AF7/AF8, rejected only 13.1% — so frontal contact
can be good. It was not good during the alpha test.

So the honest statement is narrower and worse than "frontal channels fail":

> The sensing path has never been validated on the channels the study uses. The one
> session capable of testing it had frontal contact too poor to answer the question.

That is a gap, not a refutation. But it is a gap directly under the study's main
measurement.

## Follow-up: it is contamination, not merely noise, and PILOT01 looks better

Per-channel analysis (`scripts/signal_quality.py`) sharpened the picture in two ways.

**AF7 was not just noisy — it was systematically wrong.** Analysed alone rather than
averaged, AF7 rejects 21% of windows and shows a **significant reversal** of the
eyes-closed effect: 0.83x, d = -0.32, p = 0.009. A pure noise channel gives d near zero.
A significant reversal means the channel tracked something anti-correlated with alpha,
which is the signature of ocular contamination: eyes open means more blinking, and blink
energy leaks into the analysis band. AF8 alone rejected 97% and is simply dead.

So per-channel quality gating would not rescue that session. There was no good frontal
channel to fall back to.

**PILOT01's frontal contact was much better.** Amplitude beyond 100 uV, by session:

| channel | alphatest | PILOT01 |
|---|---|---|
| AF7 | 17.8% | **7.1%** |
| AF8 | **49.0%** | **3.3%** |
| TP9 | 1.8% | 2.2% |
| TP10 | 1.3% | 2.8% |

AF8 went from unusable to acceptable. AF7 remains the noisiest channel (sd 101 uV against
34 for TP9), which is consistent with residual ocular contamination but is not disqualifying.

**Alpha peak prominence in PILOT01 is consistent with a normal eyes-open recording**, not
with contamination: 1.04-1.09 across all four channels, against an eyes-open reference of
1.10 measured on TP9 in the alpha-validation session (which rises to 2.08 eyes-closed).

That is reassuring but **not** a validation. Prominence cannot separate "eyes-open cortex"
from "no cortex" without an eye-closure manipulation, which is exactly the missing
experiment. It does mean there is no positive evidence of contamination in PILOT01, and
that a repeat on AF7/AF8 has a reasonable chance of succeeding.

## Why it was not caught earlier

The alpha test was run first, as a rig check, before the channel decision was settled.
`eeg_features.py` records that AF7/AF8 became the pre-registered choice after an earlier
version read TP9 — so the change was deliberate and documented. What was missed is that
the validation predates the change and was never repeated.

Figure 0 now labels its channel pair from the session manifest rather than from an
assumption, and carries the mismatch in its caption. Reading a label off the data rather
than off memory is what surfaced this.

## What has to happen before participant data

**Repeat the alpha validation on AF7/AF8, with contact verified first.** It is a
20-minute protocol that already exists (`scripts/alpha_test.py --channels AF7,AF8`), and
it gates everything: if the eyes-closed response is not recoverable on the frontal pair
with good contact, the arousal index is not measuring what the study claims.

Three outcomes and what each means:

| result on AF7/AF8 with good contact | consequence |
|---|---|
| ratio ≥ 1.5×, low rejection | the gap closes; Figure 0 is replaced and the study proceeds |
| effect present but weak | proceed, and report frontal SNR as a limitation with the measured number |
| no effect, contact good | **the index is not validated on its own channels.** Switch to TP9/TP10 or justify β/α without an alpha manipulation |

The third outcome would be a serious finding and is better discovered now than after ten
participants.

## Consequences for the manuscript

- Figure 0's caption must state the channel pair and the mismatch. Done.
- §5 (Instrument validation) currently implies the sensing path is validated for the
  system as run. It is not, and must say so until the repeat is done.
- §8 (Limitations) gains this explicitly.
- The pre-registration is unaffected: it specifies AF7/AF8 and does not claim the
  validation covers them. No deviation is required, and none has been logged.
