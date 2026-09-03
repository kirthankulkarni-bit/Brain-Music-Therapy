# The 5.5 s analysis latency is not a floor. It is a dominated configuration.

**Found 2026-08-28.** Reproduce with `python scripts/estimator_sweep.py`.

## The claim being tested

After the precomputed library removed the audio-side bottleneck, the analysis path became
85% of the closed-loop budget — 5.5 s of a 6.5 s worst case. The manuscript called that
term *structural*: the price of estimating band power from a finite window and smoothing a
noisy result.

That framing was wrong, and the error was not arithmetic. The budget is correct. What was
wrong is the implication that the number could not be much lower.

## Method

Measured on the alpha-validation session: 359 s of real EEG with five labelled
eyes-open/eyes-closed transitions, which gives a ground-truth state change to time against.

Two metrics per estimator:

- **detection latency** — median seconds from a real state change until the estimator
  crosses the midpoint between the two block levels. End-to-end, and includes window
  centroid delay and hop quantisation, which a group-delay figure misses entirely.
- **discriminability (d)** — Cohen's d between the two states, computed only well inside
  blocks so the transition cannot inflate it.

The windowed baseline uses the **real `FeatureExtractor`**, verified at **r = 0.989**
against the session log before anything was compared. An earlier hand-rolled version
correlated r = 0.05, because it skipped the detrend, zero-phase bandpass and 60 Hz notch
the pipeline applies inside each window. The script now refuses to report if that
correlation falls below 0.9.

Streaming estimators are strictly causal — forward filtering only. The windowed baseline
filters zero-phase *within* its completed window, which is legitimate and is precisely the
trade being measured.

## Result

| estimator | detect | d | rho | ind/min | info/min |
|---|---|---|---|---|---|
| **deployed: 4 s win, 1 s hop, tau=3.0** | **5.67 s** | 1.99 | 0.962 | **1.2** | **2.14** |
| 4 s win, 1 s hop, tau=1.0 | 3.67 s | 1.49 | 0.907 | 2.9 | 2.55 |
| 2 s win, 0.5 s hop, tau=1.0 | 3.17 s | 1.35 | 0.781 | 7.4 | 3.67 |
| **2 s win, 0.5 s hop, tau=0.5** | **1.68 s** | 1.14 | 0.629 | 13.6 | **4.20** |
| streaming o4, tau=1.0 | 0.42 s | 0.83 | 0.701 | 10.6 | 2.69 |
| streaming o4, tau=0.5 | 0.30 s | 0.76 | 0.499 | 20.1 | 3.41 |
| **streaming o4, tau=0.25** | **0.17 s** | 0.70 | 0.320 | 30.9 | 3.91 |
| streaming o2, tau=0.25 | 0.11 s | 0.63 | 0.300 | 32.3 | 3.61 |

**First, a validation.** The deployed configuration measures 5.67 s against a theoretical
budget of 5.5 s. The budget arithmetic in the manuscript is confirmed empirically on real
data, which is worth stating in its own right.

## Why this is not a latency/accuracy trade-off

Per-sample `d` does fall as latency falls, which is what one expects. But `d` is not what
determines how well a session pins down a state difference — the number of *independent*
observations matters equally, and heavy smoothing destroys independence.

`info/min = d × sqrt(ind/min)`. Since a t-statistic goes as `d × sqrt(n)` and
`n = (ind/min) × minutes`, this is the t-statistic per root-minute of recording: a
duration-independent measure of how much a configuration extracts per unit time.

**By that measure the deployed configuration is dominated by 8 of the 10 alternatives.**
It is not on the efficient frontier at all. Two examples:

- **2 s window, 0.5 s hop, tau=0.5** — a two-parameter change, no new code:
  **3.4× faster response and 2.0× more information per minute.**
- **streaming, order 4, tau=0.25** — a new estimator:
  **33× faster and 1.8× more information per minute.**

The reason the deployed setting loses on both axes is that tau = 3 s smooths so hard that
consecutive outputs are nearly the same number: rho = 0.962, yielding **1.2 independent
observations per minute**. That is the same coefficient that produced an effective sample
size of 25 from a 20-minute session (20 × 1.2 ≈ 24, against 25.3 measured) — so the
smoother is simultaneously:

- the largest single term in the closed-loop latency budget, **and**
- the dominant cause of the autocorrelation that collapses statistical power.

Shortening it pays twice. That connection is the finding.

## What this changes

**For the system.** The conservative move is `--window 2 --hop 0.5 --tau 0.5`, which uses
existing code paths and existing flags. End-to-end worst case would fall from 6.5 s to
about **2.7 s** (1.68 s analysis + 1.0 s crossfade).

**For the statistics.** More independent observations per minute directly attacks the
power problem in `analysis_plan.md` §4, where achievable precision is limited by n_eff.
A 2× gain in information per minute is equivalent to a substantially longer session at no
cost in participant time.

**For the manuscript.** §3 must stop describing 5.5 s as structural. The honest statement
is stronger: *we measured the budget, found the dominant term, showed it is a
configuration rather than a floor, and quantified a setting that improves latency and
statistical efficiency simultaneously.*

## The consequence for sample size, which is the reason this matters most

`analysis_plan.md` §4 concluded that detecting the literature-matched 0.15 z effect needs
about 60 sessions however distributed, and that the project can afford roughly 20. That is
why it became a feasibility study.

That conclusion was reached with the deployed estimator's information rate. Recomputing
with the measured rates — and accounting for BOTH the independence gained and the
per-sample discriminability lost, since a faster estimator trades one for the other:

> required n scales as 1 / (d × sqrt(ind/min))²

| configuration | d | ind/min | info | gain | n per arm |
|---|---|---|---|---|---|
| deployed | 1.99 | 1.2 | 2.18 | 1.00× | **25** |
| 2 s win, 0.5 s hop, tau=0.5 | 1.14 | 13.6 | 4.20 | 1.93× | **7** |
| streaming o4, tau=0.25 | 0.70 | 30.9 | 3.89 | 1.78× | 8 |

The discriminability loss offsets part of the independence gain but does not cancel it. A
simulation adjusting only the autocorrelation gives 6 for the retuned configuration; this
fuller calculation gives 7, so the conclusion does not depend on which is used.

**If this holds, a properly powered study moves from infeasible to feasible** — roughly
7 participants per arm rather than 25, which is 14 sessions rather than 50. That would
change the study from a feasibility exercise back into a test of H1.

**It is a projection, not a result, and rests on one assumption that has not been
measured.** `d` was measured on an eyes-open/closed contrast, which is gross compared to
adaptive-versus-sham. If the subtler contrast degrades faster under a noisier estimator,
the gain shrinks — possibly a lot. Nothing here establishes that it does not.

That assumption is directly measurable, and measuring it is the strongest argument for
running PILOT02 at the retuned settings: it would yield the information rate for the real
contrast rather than for a proxy, and either confirm the projection or kill it before the
protocol is committed.

## What this does not establish

- **Measured on one session, on TP9/TP10.** That is the only session with labelled ground
  truth, and its frontal channels had 48% rejection (see `finding_channel_validation.md`).
  The ranking should be re-derived on AF7/AF8 once a clean frontal alpha test exists.
- **The eyes-closed contrast is gross.** The study's contrast — adaptive versus sham — is
  far subtler, and a configuration with lower per-sample `d` may behave differently there.
  `info/min` is the right currency for that comparison, but the argument is an inference
  from this benchmark rather than a measurement of the real contrast.
- **Nothing here is retrofitted to PILOT01.** That session was recorded at tau = 3 and
  cannot be re-derived at another tau, since only smoothed values were logged.
- **The controller was tuned against tau = 3.** The hysteresis thresholds in
  `music_engine.py` were calibrated against the noise of the current estimator. Changing
  tau changes that noise, and the thresholds must be re-derived.

That last point matters most operationally: **this change cannot be made without re-running
the hysteresis calibration.** It is not a one-line config edit despite looking like one.
