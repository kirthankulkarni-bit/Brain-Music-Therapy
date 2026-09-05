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

Faster models exist — latent consistency models reach hundreds of times realtime — so
slowness alone would be a weak reason to avoid generation. The durable reason is that
generation is **unnecessary**: audio comes from a **precomputed segment library**, which
is a complete solution rather than a compromise because
`build_prompt()` is a pure function with a finite range — **5 energy rungs, and it cannot
emit anything else.** A library covering those 5 covers the controller's entire output
space exactly. Even given an infinitely fast generator, re-synthesising one of five
prompts is strictly worse than selecting a pre-rendered variant. See
[docs/related_work.md](docs/related_work.md).

The range was 20 until 2026-09-05: each rung could carry one of four trend suffixes. The
suffix was removed because the slope it gated on is smaller than the noise of the
estimator measuring it, so no threshold could work — see
[docs/deviations.md](docs/deviations.md). That quartered the prompt space and made this
argument four times stronger.

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

Renders are allocated by how much a prompt actually plays: 32 for each of the five base
prompts, which are now the entire prompt space.

Libraries built before 9/5 also hold 4 renders for each of 15 suffixed prompts — 60 of
220 segments, 8.0 of 29.3 minutes. They still load; those segments are simply never
requested. Rebuild to reclaim the space.

Budget roughly **16 s per segment** on a GTX 1650 Ti. Two builds measured:

| build | segments | wall clock | note |
|---|---|---|---|
| first | 80 | 2h44m | one 2h19m stall on segment 1 |
| second | 140 | **37.8 min** | no stall |

The first build's stall has never been explained. Its very first `generate()` ran for
2h19m while the remaining 79 completed normally in 24.6 min; the audio was unaffected
and is byte-for-byte normal. The leading hypothesis was VRAM oversubscription on a 4 GB
card immediately after another CUDA process released its context, with Windows WDDM
paging to system RAM — and the second build is weak support for it, since it started
with no other CUDA process running and stalled not at all. One clean run is not proof,
so treat it as a hypothesis still.

If a build appears stuck on segment 1, it may genuinely be. It is resumable, so killing
and rerunning costs nothing.

Check everything checkable without hardware — 70 tests, no GPU, no headset:

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
| `scripts/verify_library.py` | 15 checks on coverage, mixing, clipping bounds, and latency |
| `scripts/run_tests.py` | every hardware-free check in one command |
| `scripts/validate_coupling.py` | ground-truth validation of the coupling estimator |
| `scripts/power_analysis.py` | simulated sample size, using the measured autocorrelation |
| `scripts/ladder_policy.py` | compares energy-ladder policies against real z |
| `scripts/controller_replay.py` | replays a recording through the real controller; chatter counts |
| `scripts/make_figures.py` | the paper figures, drawn from session data |
| `scripts/verify_claims.py` | regenerates every number the preprint cites, and checks it |
| `scripts/signal_quality.py` | per-channel: is this electrode measuring cortex, or eyes? |
| `scripts/estimator_sweep.py` | latency vs information rate for candidate estimators |
| `scripts/calibrate_hysteresis.py` | is the trend measurable under a given estimator? (so far, never) |
| `scripts/validate_index_deap.py` | tests log(beta/alpha) against DEAP arousal labels |
| `scripts/compare_indices_deap.py` | seven candidate arousal indices, all reported |
| `benchmarks/latency_probe.py` | the latency budget probe; run it on any new machine |
| `notebooks/latency_probe_colab.ipynb` | the same probe on a Colab GPU |
| `docs/results_latency.md` | every benchmark run, and what each one licenses |
| `docs/results_pilot.md` | the first closed-loop session and the three defects it found |
| `docs/figures/` | generated figures, regenerate with `make_figures.py` |
| `docs/analysis_plan.md` | **frozen pre-registration** — do not edit; see below |
| `docs/deviations.md` | the deviation log — kept outside the frozen file, and why |
| `docs/related_work.md` | the arXiv sweep, and what survives it |
| `docs/finding_channel_validation.md` | the validation/index channel mismatch |
| `docs/finding_analysis_latency.md` | why 5.5 s is a dominated configuration |
| `docs/preprint_draft.md` | sections 1–8 |
| `docs/preprint_outline.md` | what the paper can claim, mapped to evidence |
| `docs/runbook_session.md` | how to run one participant, start to finish |
| `docs/HANDOFF.md` | **read first in a new session** — state, numbers, and the traps |
| `docs/next_session.md` | what the next hardware session must answer, step by step |
| `docs/finding_ladder_hysteresis.md` | why the estimator retuning is not a flag change |

## Pre-registration

The analysis plan was **frozen 2026-08-28, before any participant data existed**.

| | |
|---|---|
| document | [docs/analysis_plan.md](docs/analysis_plan.md) |
| git tag | `preregistration-v1` |
| commit | `e45bd321dbaa048f946fc1d199cfc8a57a05d33e` |
| sha256 (LF-normalised) | `538328a2dac75fc9bab76fecb7f7cfa11ef88db9b08f6cf7e187bd1fe4fe4ce5` |

`scripts/run_tests.py` checks that hash on every run, so an edit after the freeze fails
the suite rather than passing unnoticed. Retrieve the frozen text with:

```bash
git show preregistration-v1:docs/analysis_plan.md
```

Changes after the freeze — including any forced by ethics review — are logged as dated
deviations in §9 of the plan, each with its own commit. The file itself does not move.

**It is a feasibility study.** At n = 10, power to detect the literature-matched 0.15 z
effect is 38%, so no efficacy test is run and no p-value is reported for the primary
contrast. The outputs are an interval and two variance components. See §1 and §4.

## Three things to know before trusting a number here

**Between-run variance dominates.** The same configuration on this laptop varied by
up to **1.96×** across runs, while within-run spread was 1.01–1.11. A single run of
`latency_probe.py` is not a reproducible measurement here. Report the range.

**Never compare across backends.** audiocraft and transformers have different
sampling loops and defaults, so their absolute numbers are not interchangeable.
Every result JSON records `backend` for this reason.

**The analysis path, not the GPU, is the bottleneck — and it is a configuration.**
After the library fix, 85% of the 6.5 s budget is analysis. Measured against labelled
ground truth, the deployed estimator takes 5.67 s to register a state change and yields
1.2 independent observations per minute; **8 of the 9 alternatives beat it on latency AND
information rate simultaneously**. See [docs/finding_analysis_latency.md](docs/finding_analysis_latency.md).

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
- ~~Should the trend suffix be removed?~~ **Removed 2026-09-05.** Not for being unable to
  fire, which would have argued for keeping it as harmless, but because the quantity it
  thresholded is **not measurable**: the largest genuine 60 s drift is 0.0385 z/hop
  against slope-estimator noise of 0.0681. A threshold above the noise can only be
  crossed by noise; one low enough to catch real drift fires constantly. Verified a
  no-op across all 1212 logged windows. See [docs/deviations.md](docs/deviations.md).
- ~~Independent or crossover design?~~ **Settled: crossover, cross-yoked**
  (`analysis_plan.md` §2, frozen). This entry quoted 0.3 z, which was superseded — the
  registered smallest effect of interest is **0.15 z**, matched to a neurofeedback
  meta-analytic g ≈ 0.3 against a between-participant SD of 0.5 z. At that effect it is
  **25 participants per arm** paired, against 61 independent, which is why the study runs
  at n = 10 as an explicit feasibility study with **38% power**. Both numbers are
  asserted by `verify_claims.py`; `scripts/power_analysis.py` has the full table.
- Does the crossfade sound acceptable? `python src/library_engine.py --wav demo.wav`
  renders a scripted arousal trajectory to listen to. This is the one cost of the 8×
  that no metric captures, and PILOT01's audio was chattering, so it has never been
  judged on a representative session.
