# Brain-Music-Therapy

Closed-loop AI music therapy driven by real-time EEG. A Muse 2 headband streams
four-channel EEG over LSL; a frontal arousal index is extracted every second,
normalized against the participant's own resting baseline, and used to steer
continuous MusicGen audio toward a target state.

The system's contribution is not a faster model. It is a **characterization of the
irreducible latency floor for consumer-EEG closed-loop audio**, plus the scheduling
architecture that reaches it.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify the hardware before anything else:

```bash
python scripts/verify_sample_rate.py
```

Run a session with no headset and no GPU, to see the whole loop work:

```bash
python src/live_music.py --mock --baseline-seconds 20 --duration 2
```

A real session:

```bash
python src/live_music.py --participant P01 --duration 10 --target -1.0
```

Then analyze it:

```bash
python src/analyze_session.py
```

---

## The sampling-rate defect

The live pipeline previously declared `sfreq = 128`, with a comment calling it a
Muse 2 hardware spec. **The Muse 2 streams EEG at 256 Hz.** The 128 Hz figure is
the DEAP dataset's rate, carried over when the project moved from offline DEAP
analysis to the live headset.

Every frequency on the axis was consequently halved:

| mask in the code | frequencies actually measured |
|---|---|
| alpha 8–13 Hz | true 16–26 Hz |
| beta 13–30 Hz | true 26–60 Hz |

The beta mask was inclusive at 30 Hz, and 60 Hz mains hum maps to exactly 30 Hz
under a halved axis. The dominant contributor to "beta power" was therefore power
line noise on dry electrodes. That is why the resting baseline sat near 0.4 and why
the hysteresis thresholds had to be retuned to 0.35 / 0.55 to stop the state
machine chattering.

Everything in `logs_precorrection/` predates the fix and **cannot be cited as pilot
data**. See `logs_precorrection/README.txt`.

Three things prevent a recurrence:

1. `stream_utils.get_inlet()` returns `(inlet, sampling_rate)`, read from the
   stream's own metadata. The live path contains no sampling-rate literal.
2. `FeatureConfig.sampling_rate` has no default and rejects implausible values.
3. Every session manifest records the rate that was actually used.

`scripts/validate_deap.py` is the control: it confirms the new feature path
reproduces the old offline DEAP results (Spearman ρ = 1.00 across trials), which
localizes the defect to the live path rather than to the feature definition.

---

## Latency

Measure it on your own machine:

```bash
python benchmarks/latency_probe.py --skip-musicgen
python benchmarks/latency_probe.py
```

The two real bottlenecks are neither of them the GPU.

**Bottleneck 1 — the analysis path (structural).** Window centroid delay + hop
quantization + smoother group delay. Under the old 5 s window / 2 s hop / 5-sample
boxcar this was 7.5 s. At 4 s / 1 s / τ=3 s it is 5.5 s. DSP compute is ~1.4 ms per
window, roughly 700× headroom against the hop — the compute is irrelevant, the
structure is everything.

**Bottleneck 2 — the audio queue.** Once a segment is queued it must play to
completion, so worst-case audio latency is `queue_depth × segment_seconds`. Depth
is 1 by default, and that is a latency decision rather than an oversight. Raise it
only if `analyze_session.py` reports underruns.

MusicGen-small at fp16 on an RTX 3050 Ti generates 8 s of audio in roughly 4–7 s —
faster than realtime, so generation hides behind playback and never binds.

---

## Layout

```
scripts/
  verify_sample_rate.py   run first: nominal vs empirical rate vs mains ground truth
  validate_deap.py        control experiment; isolates the defect to the live path
benchmarks/
  latency_probe.py        the full latency budget table
src/
  live_music.py           session orchestrator: baseline phase, control loop, dashboard
  eeg_features.py         filtering, PSD, band integration, smoothing, z-normalization
  music_engine.py         continuation-based MusicGen generation + gapless playback
  session_logger.py       JSONL events + raw float32 EEG + manifest
  analyze_session.py      session metrics and the lagged audio-neural coupling index
  stream_utils.py         LSL connection; the only source of the sampling rate
  lsl_receiver.py         standalone connection diagnostic
  data_smoothing.py       offline DEAP script (128 Hz is correct there)
data_to_music.py          offline DEAP demo (128 Hz is correct there)
load_deap.py              DEAP loader
sessions/                 session output, one directory per run
logs_precorrection/       pre-fix CSVs, retained for provenance only
testing/                  retired scripts and hardware experiments, kept as history
```

## Session protocol

1. **Baseline, 120 s, mandatory.** Eyes open, at rest, no audio. Produces the
   per-participant mean and SD used for z-scoring. Without it, absolute beta/alpha
   values are not comparable across people and participants cannot be pooled. A
   baseline with too few valid windows aborts the session — that means bad
   electrode contact, and catching it here is the point.
2. **Intervention.** Every hop, a window is extracted. Invalid windows are logged
   and skipped without updating the smoother. Valid ones are smoothed, z-scored,
   and mapped to one of five graded prompts by their error against the target.
3. **Analysis.** `analyze_session.py` reports time in band, time to target,
   rejection rate, generation latency, and the coupling index.

Hysteresis is gone. It existed to debounce a binary switch; graded prompts need no
debouncing, and any deadband that remains is defined in z units so it transfers
across participants.

## Yoked sham arm

```bash
python src/live_music.py --participant P02 --condition sham --yoke-from sessions/P01_20260814_...
```

Replays a prior adaptive session's prompt schedule on its original timeline,
ignoring the current participant's EEG. Every acoustic property is matched; only
the contingency between brain and music is broken. The adaptive-vs-sham contrast in
the coupling index is what rules out regression to the mean, which a single-arm
design cannot.

## Lagged audio-neural coupling index

Cross-correlates the generated audio's amplitude envelope against the alpha power
envelope across lags. Reports peak correlation, the lag where it occurs, and a
p-value from a circular-shift null that respects the autocorrelation in both
series. In the adaptive arm coupling should peak at positive lag, audio leading
brain; in the yoked sham it should collapse. The lag is a latency measurement taken
on the brain rather than on the clock.

## Human subjects

ISEF human participant research requires SRC approval and Form 4 **before** any
data collection, and an anxiety framing may trigger a Qualified Scientist
requirement. Turnaround is multi-week. Start that paperwork in parallel with the
code, not after it.
