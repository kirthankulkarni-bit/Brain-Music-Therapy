# Session runbook — running one participant

Operational procedure for a single closed-loop session. Distinct from
[runbook_hardware_validation.md](runbook_hardware_validation.md), which is a record of
the 2026-08-14 rig-validation day and not a per-session procedure.

**This document covers the technical procedure only.** Consent, screening, debrief,
adverse-event handling and data-protection steps are set by your SRC approval and
belong in the protocol, not here. Do not run a participant before that approval is in
place; the technical readiness described below is necessary and not sufficient.

Everything here is derived from PILOT01 (2026-08-22), the first full closed-loop
session. Where a step exists because something went wrong, it says so.

---

## 0. The day before

```bash
python scripts/run_tests.py
```

24 checks, no hardware. If anything fails, fix it before a participant is in the
building — a failed session cannot be repeated on the same person without
contaminating it.

Confirm the library exists and matches the controller:

```bash
python scripts/verify_library.py
```

Charge the Muse fully. A low battery raises the noise floor, and you will not be able
to tell that apart from bad contact once you are in the session.

---

## 1. Pre-flight, before the participant arrives (10 min)

```bash
python -c "import pylsl, scipy, sounddevice, soundfile; print('imports OK')"
```

1. Launch BlueMuse, connect the headset, press **Start Streaming**.
2. Confirm the stream is alive — four voltages scrolling, then `Ctrl+C`:

```bash
python src/lsl_receiver.py
```

3. **Unplug the laptop and run on battery.** Mains pickup through the charger was the
   single largest contributor to 60 Hz contamination during rig validation.
4. Set the playback volume now, on a segment from the library, and do not change it
   mid-session. Volume is an uncontrolled acoustic variable across arms.

```bash
python src/library_engine.py --seconds 20
```

---

## 2. Fitting and the contact gate (15–30 min, expect to iterate)

Wipe the participant's forehead. Frontal sensors sit on skin, and hair or dry skin
under AF7/AF8 is the usual cause of failure. A very slightly damp sensor reads far
better than a dry one. Behind-the-ear sensors need skin, not hair.

```bash
python scripts/contact_check.py --seconds 30
```

**Gate: AF7 and AF8 must both read `GOOD`.** They are averaged into the arousal index,
so one bad channel corrupts every downstream number. Be stubborn here; reseat and
rerun rather than proceeding.

**Run this with the headphones already on.** Over-ear cups press down exactly where
TP9 and TP10 sit, so a contact check done bare-headed does not describe the
configuration you will record in. Earbuds or speakers avoid the problem entirely and
are preferred.

If a channel will not exceed `FAIR`, you may proceed but must log it — expect a higher
rejection rate, and note it as a per-session covariate.

---

## 3. Running the session

### Adaptive arm

```bash
python src/live_music.py --participant P01 --condition adaptive --duration 20
```

### Yoked sham arm

```bash
python src/live_music.py --participant P02 --condition sham --yoke-from sessions/P01_20260822_153652 --duration 20
```

The sham replays a prior session's prompt schedule on its original timeline, ignoring
the current participant's EEG. Every acoustic property is matched; only the
contingency is broken.

Known fidelity limit, measured rather than assumed: the replay reproduces the source's
prompt sequence exactly but runs **0.4–1.0 s early throughout**, and the source's first
prompt is skipped. Offsets normalize against the first *audio event*, which is logged
when a segment is produced rather than when the prompt was set. Describe the arms as
"matched in sequence and duration, with a known sub-hop lead", never as identical.

### What the participant does

- **Baseline, 120 s:** eyes open, sit still, no audio. Do not skip or shorten this.
  Without a usable baseline there are no z-scores and the intervention is steering on
  an uncalibrated index. A baseline with fewer than 10 valid windows aborts the
  session by design.
- **Intervention:** sit still, jaw relaxed, breathe normally. Jaw clenching swamps the
  frontal channels and every clenched window is rejected.

### What to watch on the console

| signal | expected | action if not |
|---|---|---|
| rejection rate | ~13% (PILOT01) | above 40%, stop and re-seat — the coupling analysis needs the windows |
| `SESSION FAILED` | never appears | stop. The run is unusable; the message names the phase and error |
| audio | continuous, audible | silence means the engine starved; check `underruns` at the end |

---

## 4. Immediately after, before the participant leaves

```bash
python src/analyze_session.py
```

Check these four before letting them go, because only some are recoverable by an
immediate repeat:

| check | expected |
|---|---|
| `session complete` in the log | present — its absence means the worker crashed |
| buffer underruns | 0 |
| `missing_prompts` | empty — non-empty means the library is stale |
| intervention rejection rate | under ~40% |

A crashed worker writes `session FAILED` with its phase and error, and does **not**
write a completion note. This distinction exists because a `NameError` on 2026-08-16
killed the control loop on its first intervention window while still recording the
session as complete — baseline logged normally, manifest written, zero intervention
data, and a directory that looked successful.

---

## 5. Data handling

Each session writes to `sessions/<PARTICIPANT>_<TIMESTAMP>/`:

| file | contents | in git |
|---|---|---|
| `events.jsonl` | every window, audio event and note | yes |
| `manifest.json` | full config, engine, latency budget, code version | yes |
| `metrics.json` | computed outcomes | yes |
| `raw_eeg.f32` | raw samples, ~18 MB/hour | **no** (gitignored) |

Back up `raw_eeg.f32` separately if your protocol requires raw retention — it is the
only artefact that cannot be regenerated.

Participant identifiers appear in directory names and manifests. Use coded IDs
(`P01`), never names or initials, and keep the linking key wherever your approval
specifies.

---

## 6. Failure modes seen at least once

| symptom | cause | fix |
|---|---|---|
| baseline aborts at <10 windows | bad contact, or `--baseline-seconds` too short | reseat; the default 120 s is sized for this |
| `no library at .../manifest.json` | library not built | `python scripts/build_library.py` |
| session "completes" with no intervention windows | worker crashed (pre-2026-08-16 behaviour) | check for `session FAILED`; update the code |
| audio switching every 1–2 s | controller chatter (pre-hysteresis) | fixed 2026-08-16; verify with `run_tests.py` |
| high rejection with good contact | jaw tension, or mains via the charger | run on battery, remind about the jaw |

---

## 7. Before analysing across participants

Two properties of this data will mislead anyone who assumes otherwise:

**Windows are not independent.** Lag-1 autocorrelation 0.953, decorrelation time 9 s,
so 1043 windows carry an effective sample size of ~25. Never run a test that treats
windows as independent observations.

**The event-locked response is not causal within one arm.** Rung changes are triggered
by z moving, so a positive effect is what a loop with no therapeutic effect also
produces. Only adaptive minus yoked sham is interpretable. See
[results_pilot.md](results_pilot.md).
