# Brain-Music-Therapy

Closed-loop AI music therapy driven by real-time EEG. A Muse 2 headband streams
four-channel EEG over LSL; a frontal arousal index is extracted every second,
normalized against the participant's own resting baseline, and used to steer
continuous music toward a target state.

## The central result

**Live MusicGen generation is slower than realtime on every GPU tested**, including a
datacenter T4 — best case 1.14× realtime where streaming needs < 1.0×. Full
measurements and caveats in [docs/results_latency.md](docs/results_latency.md).

So audio does not come from live generation. It comes from a **precomputed segment
library**, and that turns out to be a complete solution rather than a compromise:
`build_prompt()` is a pure function with a finite range — 5 energy rungs × 4 trend
variants = 20 strings, and it cannot emit anything else. A library covering those 20
covers the controller's entire output space exactly.

The library also responds *faster* than streaming ever could, and not because
selection is quick. Streaming **committed**: once a segment entered the queue it
played to completion, so a prompt change waited up to a full segment no matter how
fast the GPU was. A resident library abandons the current segment mid-playback.

| | streaming | library |
|---|---|---|
| worst-case prompt → audible | 8.0 s | **1.0 s** |
| measured time to first audible change | up to 8.0 s | **13 ms** |
| time to produce 8 s of audio | 15.5–27.7 s | 2.4 µs (selection) |
| underruns | permanent starvation | 0 |

## Quick start

Build the library once (~25 min on a GTX 1650 Ti, needs a GPU):

```bash
python scripts/build_library.py --variants 4
```

Check it covers the controller and plays cleanly (no GPU needed):

```bash
python scripts/verify_library.py
```

Run a session:

```bash
python src/live_music.py --participant P01 --condition adaptive --duration 10
```

Try the whole control loop with no headset and no library:

```bash
python src/live_music.py --mock --headless --duration 1 --baseline-seconds 20
```

## Layout

| path | what it is |
|---|---|
| `src/live_music.py` | the closed loop: LSL → features → prompt → audio |
| `src/eeg_features.py` | windowing, band power, arousal index, baseline normalization |
| `src/music_engine.py` | `build_prompt()` (the controller) and the live-generation engine |
| `src/library_engine.py` | precomputed-library playback, crossfading — **the default audio path** |
| `src/analyze_session.py` | post-hoc analysis, including lagged audio-neural coupling |
| `scripts/build_library.py` | renders every prompt the controller can emit |
| `scripts/verify_library.py` | 14 checks on coverage, mixing, and latency |
| `benchmarks/latency_probe.py` | the latency budget probe; run it on any new machine |
| `notebooks/latency_probe_colab.ipynb` | the same probe on a Colab GPU |
| `docs/results_latency.md` | every benchmark run, and what each one licenses |

## Two things to know before trusting a benchmark number

**Between-run variance dominates.** The same configuration on this laptop varied by
up to **1.96×** across runs, while within-run spread was 1.01–1.11. A single run of
`latency_probe.py` is not a reproducible measurement here. Report the range.

**Never compare across backends.** audiocraft and transformers have different
sampling loops and defaults, so their absolute numbers are not interchangeable.
Every result JSON records `backend` for this reason.

## Open questions

- Ladder rungs 0 and 4 are **unreachable** — `build_prompt` always leads by one rung
  toward a goal that is only ever rung 1 or 3 under the two therapeutic targets. The
  sparsest drone never plays, not even to a participant sitting at that arousal
  level. Whether that is intended is a therapeutic call, so the code is unchanged and
  the library renders them as insurance.
- `fp16-half` has only been measured on a card without tensor cores, where it wins
  6–8%. On a T4 it is the one configuration that could plausibly approach realtime.
- Does the crossfade sound acceptable? `python src/library_engine.py --wav demo.wav`
  renders a scripted arousal trajectory to listen to. This is the one cost of the 8×
  that no metric captures.
