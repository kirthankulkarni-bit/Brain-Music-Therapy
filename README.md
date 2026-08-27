# Brain-Music-Therapy

Closed-loop AI music therapy driven by real-time EEG. A Muse 2 headband streams
four-channel EEG over LSL; a frontal arousal index is extracted every second,
normalized against the participant's own resting baseline, and used to steer
continuous music toward a target state.

## The central result

**Live MusicGen generation is slower than realtime on every GPU tested**, including a
datacenter T4 across three independent runs and all three precision modes — best case
**1.05× realtime** where streaming needs < 1.0×. Full measurements and caveats in
[docs/results_latency.md](docs/results_latency.md).

The mechanism is settled, and it is not the hardware. A T4 runs fp16 on tensor cores
at roughly 8× its fp32 throughput; giving this workload that made it *slower*, so it
was never arithmetic-bound. Batch-1 autoregressive decoding is bound by the sequential
token loop, which no faster GPU and no numeric format addresses.

So audio does not come from live generation. It comes from a **precomputed segment
library**, and that turns out to be a complete solution rather than a compromise:
`build_prompt()` is a pure function with a finite range — at most 5 energy rungs × 4
trend variants = 20 strings, and it cannot emit anything else. A library covering
those 20 covers the controller's entire output space exactly.

In practice the range is much smaller, and the measured numbers belong next to the
design claim: only rungs 1–3 are reachable under the two therapeutic targets, and
PILOT01 used rung 1 for 96% of a 20-minute session. **Do not describe the controller
as five graded levels** — see [docs/results_pilot.md](docs/results_pilot.md).

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

Build the library once — 220 segments, needs a GPU. Resumable, so an interrupted run
continues where it stopped:

```bash
python scripts/build_library.py
```

Renders are allocated by how much a prompt actually plays: 32 for each of the five
suffix-free base prompts, which carry essentially the whole session, and 4 for the
suffixed ones as insurance. A uniform 32 across all 20 would cost three hours of GPU
to produce audio the controller cannot select.

Budget ~25 min on a GTX 1650 Ti *for the generation itself* (79 of 80 segments took
14–35 s, median 17.9 s). The one observed build took **2h44m wall clock**, because
the very first `generate()` call ran for 2h19m before the remaining 79 completed
normally in 24.6 min. Cause not established; the most likely explanation is VRAM
oversubscription on a 4 GB card immediately after another CUDA process released its
context, with Windows WDDM paging to system RAM. The audio was unaffected — that
segment is byte-for-byte normal (8.00 s, RMS 0.1000, all finite). If a build seems
stuck on segment 1, it may genuinely be; it is resumable, so killing and rerunning
costs nothing.

Check everything checkable without hardware — 24 tests, no GPU, no headset:

```bash
python scripts/run_tests.py
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
| `scripts/run_tests.py` | every hardware-free check in one command |
| `scripts/validate_coupling.py` | ground-truth validation of the coupling estimator |
| `scripts/power_analysis.py` | simulated sample size, using the measured autocorrelation |
| `scripts/ladder_policy.py` | compares energy-ladder policies against real z |
| `scripts/make_figures.py` | the paper figures, drawn from session data |
| `benchmarks/latency_probe.py` | the latency budget probe; run it on any new machine |
| `notebooks/latency_probe_colab.ipynb` | the same probe on a Colab GPU |
| `docs/results_latency.md` | every benchmark run, and what each one licenses |
| `docs/results_pilot.md` | the first closed-loop session and the three defects it found |
| `docs/figures/` | generated figures, regenerate with `make_figures.py` |

## Three things to know before trusting a number here

**Between-run variance dominates.** The same configuration on this laptop varied by
up to **1.96×** across runs, while within-run spread was 1.01–1.11. A single run of
`latency_probe.py` is not a reproducible measurement here. Report the range.

**Never compare across backends.** audiocraft and transformers have different
sampling loops and defaults, so their absolute numbers are not interchangeable.
Every result JSON records `backend` for this reason.

**Windows are not independent.** PILOT01's z has a lag-1 autocorrelation of 0.953 and
a 9 s decorrelation time, so 1043 valid windows carry an effective sample size of
**25.3**. Any analysis treating them as independent overstates its evidence by about
**6.4×**. Every inferential number in this project uses either the effective sample
size or a permutation null that preserves the autocorrelation.

## Open questions

- Ladder rungs 0 and 4 are **unreachable** — `build_prompt` always leads by one rung
  toward a goal that is only ever rung 1 or 3 under the two therapeutic targets. The
  sparsest drone never plays, not even to a participant sitting at that arousal
  level. Whether that is intended is a therapeutic call, so the code is unchanged and
  the library renders them as insurance.
- Should the trend suffix be removed? After the hysteresis fix it is calibrated above
  the measured noise ceiling and therefore **cannot fire in a normal session**. A
  branch that cannot fire is arguably dead code, but deleting it is a design decision.
- Independent or crossover design? Detecting a 0.3 z effect needs ~60 participants per
  arm independently, or **8** paired. `scripts/power_analysis.py` has the full table.
- Does the crossfade sound acceptable? `python src/library_engine.py --wav demo.wav`
  renders a scripted arousal trajectory to listen to. This is the one cost of the 8×
  that no metric captures.
