# Hardware validation session — runbook

**Goal for this session: prove the sensing path is trustworthy.** Not the audio path.
Audio stays in mock mode throughout (`--mock-audio` synthesizes pads instead of
calling MusicGen). Live MusicGen is known-infeasible on this GPU — measured at ~5×
slower than realtime on 2026-08-14, see `benchmarks/latency_results.json` — and the
precomputed segment library that replaces it is not built yet. Do not spend any of
today's time fighting that.

Budget roughly 3 hours. Steps 1–3 are gates: if one fails, stop and fix it rather
than pushing on, because everything downstream inherits the problem.

Record results as you go. Sections marked **JOURNAL** are worth writing up.

---

## 0. Pre-flight (5 min)

```bash
cd C:\Users\KingKirthan\OneDrive\Desktop\BrainMusicProject
.venv\Scripts\activate
python -c "import pylsl, scipy, sounddevice; print('imports OK')"
```

Then, in order:

1. Charge the Muse, or plug it in. A low battery raises the noise floor.
2. Launch BlueMuse, connect the headset, press **Start Streaming**.
3. Confirm the stream is alive: `python src/lsl_receiver.py` — you should see
   timestamps and four voltages scrolling. Ctrl+C once you do.

Practical notes on wearing it: the two frontal sensors (AF7, AF8) sit on your
forehead, and hair or dry skin under them is the usual cause of bad contact. Wipe
your forehead first. A very slightly damp sensor reads much better than a dry one.
The behind-the-ear sensors (TP9, TP10) need to sit on skin, not hair.

---

## 1. GATE — verify the sampling rate (10 min)

```bash
python scripts/verify_sample_rate.py --seconds 30
```

**Expect:** empirical rate ≈ 256 Hz, and the mains check identifying 256 Hz as the
assumption that puts the 60 Hz peak at 60 Hz.

**If it reports ~128 Hz:** stop. Either BlueMuse is decimating, or the assumption
behind the entire correction is wrong. Do not collect data until this is resolved.

**If the mains peak is unclear:** not a failure. It means either good contact or a
well-filtered supply. The timestamp-derived rate is the authoritative number here.

**JOURNAL:** the empirical rate, and whether the mains check agreed. This is the
empirical confirmation of the defect you corrected — it belongs in the paper.

---

## 2. GATE — electrode contact (15–30 min, expect to iterate)

```bash
python scripts/contact_check.py --seconds 30
```

**Expect:** AF7 and AF8 both `GOOD`. RMS in the 5–30 µV range, 60 Hz ratio below
0.10.

This is the step to be stubborn about. Adjust, rerun, adjust again. Things that
help, roughly in order of effect:

- Reseat so the frontal sensors sit flat, and push hair out from under them
- Dampen the sensors very slightly with water
- Move away from laptop chargers, monitors, and power strips (mains pickup)
- Unplug the laptop and run on battery for the test — often the single biggest win
- Sit still with a relaxed jaw; clenching swamps everything

**If AF7 or AF8 will not go above `FAIR`:** you can proceed, but log it. Expect a
higher rejection rate, and note that the frontal pair is averaged, so one bad
channel corrupts the index.

**If a channel reads `DEAD` repeatedly after reseating:** likely a hardware fault.
Note it — it constrains everything afterward.

**JOURNAL:** final per-channel verdicts and 60 Hz ratios, plus what it took to get
there. "Time to acceptable contact" is a real usability finding for consumer EEG.

---

## 3. GATE — the eyes-closed alpha test (20 min)

**This is the single most important step of the day.** It is the classic proof that
you are recording real brain activity and not amplifier noise: occipital and frontal
alpha rises sharply when the eyes close, typically by 1.5–3×. If you cannot
reproduce this, nothing built on top of it means anything.

```bash
python scripts/alpha_test.py
```

The script runs the block sequence itself — 6 blocks of 60 s, alternating — so every
window is labelled with the condition it belongs to and you get a statistic instead
of an impression. **Cues are audible**, because you cannot read a terminal with your
eyes closed:

- **low tone** → close your eyes
- **high tone** → open your eyes

Sit still, jaw relaxed, breathe normally. Windows that straddle a transition are
discarded automatically, as is the first 3 s after each cue.

**Expect:** ratio ≥ 1.5×, p < 0.01, and a clean square wave in the saved figure.
The script prints PASS, WEAK, or FAIL and exits non-zero on failure.

Typical frontal values are in the 1.5–3× range. Occipital alpha is much stronger,
but the Muse has no occipital electrodes, so do not expect the 5–10× ratios quoted
in textbooks that use Oz.

**If there is no visible difference:** the most likely causes, in order — contact
degraded during the run (rerun step 2), the frontal sensors are picking up mostly
muscle rather than cortex, or you were not relaxed enough for alpha to appear. Some
people show weak frontal alpha; if so, note it, because it affects whether
beta/alpha is the right index for you.

**JOURNAL:** the ratio, Cohen's d, p-value, and the saved `alpha_validation.png`.
This is a figure in your paper — it is your evidence that the sensing path measures
cortex rather than noise.

---

## 4. Baseline stability (20 min)

Two consecutive baselines, several minutes apart, no intervention needed:

```bash
python src/live_music.py --participant BASELINE1 --mock-audio --baseline-seconds 120 --duration 0.5
python src/live_music.py --participant BASELINE2 --mock-audio --baseline-seconds 120 --duration 0.5
```

Compare the reported baseline mean and SD from each.

**Why this matters:** every z-score in the study is computed against these two
numbers. If the baseline mean drifts substantially between runs 10 minutes apart on
the same person, then z-scores are not comparable across sessions, and any
within-subject design needs a baseline at the start of every session (which is what
the code already enforces — this tells you whether that is sufficient).

**Expect:** means within roughly 0.1–0.2 of each other, similar SDs.

**JOURNAL:** both means and SDs. Test–retest stability of the baseline is a
methods-section number that reviewers look for.

---

## 5. Artifact characterization (15 min)

Run one more short session and *deliberately* produce artifacts, noting the wall
time of each:

```bash
python src/live_music.py --participant ARTIFACT --mock-audio --baseline-seconds 60 --duration 3
```

At roughly 30-second intervals: blink hard 5×, clench your jaw, look sharply left
and right, raise your eyebrows, then sit still for a full minute.

```bash
python src/analyze_session.py
```

**Expect:** the rejection reasons breakdown to show amplitude artifacts clustering
around the times you produced them, and a low rejection rate during the still
minute.

**Why:** this validates that the artifact rejection is catching what it should, and
gives you a defensible rejection-rate figure for clean data. If sitting still still
yields >20% rejection, the thresholds in `FeatureConfig` need adjusting for your
setup rather than being taken as given.

**JOURNAL:** rejection rate while still, and which artifact types the detector
caught versus missed.

---

## 6. Wrap-up (10 min)

```bash
git add sessions/ docs/
git status --short
```

Commit the sessions with a journal-style message describing what you observed. The
raw `.f32` files are gitignored; the event logs and metrics are small and are the
data.

---

## Failure modes, quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| "No Muse stream found" | BlueMuse not streaming | Press Start Streaming; check the headset is paired |
| Stream found, no samples | Headset asleep | Power cycle the Muse |
| All channels `DEAD` | Headset off head, or not seated | Reseat; confirm the LED |
| High 60 Hz on every channel | Mains environment | Unplug the laptop, move away from the monitor |
| Baseline aborts | <25% valid windows | Rerun step 2; do not force past this |
| Rejection rate >30% while still | Thresholds wrong for your setup | Note it; adjust `reject_peak_to_peak_uv` deliberately and record the change |
| Qt window blank or frozen | Dashboard/thread issue | Rerun with `--headless` and analyze afterward |
| Audio silent | Expected | `--mock-audio` is on; audio is out of scope today |

---

## Explicitly out of scope today

- **Live MusicGen.** Measured at ~5× slower than realtime on this GPU. It will
  starve the queue and waste your evening.
- **Tuning the prompt ladder.** It needs the segment library first.
- **Anything involving other people as participants.** SRC approval and Form 4 come
  first. Today is engineering validation on yourself, which is a different thing.

## Next session (not today)

Build the precomputed segment library: generate ~40 clips offline (5 ladder rungs ×
8 variants × 8 s, roughly 50 minutes of one-time GPU time), and replace the
streaming generator with runtime selection and crossfade.
